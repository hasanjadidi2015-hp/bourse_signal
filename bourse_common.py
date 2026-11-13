# -*- coding: utf-8 -*-
"""
توابع مشترک پروژه: زمان، متن فارسی، اعداد، دیتابیس، لاگ، قفل تک‌نمونه و پشتیبان‌گیری.
قاعده‌ی این فایل: هیچ تابعی نباید خطای کنترل‌نشده به بیرون بدهد (به‌جز قفل که عمداً AlreadyRunning می‌دهد).
"""
import glob
import logging
import logging.handlers
import math
import os
import sqlite3
import sys
import threading
import time
from datetime import datetime, timedelta, timezone

try:
    import config as _config
except Exception:
    _config = None

try:
    from zoneinfo import ZoneInfo
    TEHRAN = ZoneInfo("Asia/Tehran")
except Exception:
    TEHRAN = timezone(timedelta(hours=3, minutes=30))


# ---------------------------------------------------------------- تنظیمات
def cfg(name, default=None):
    """مقدار تنظیم را از config.py می‌خواند؛ اگر نبود یا خراب بود مقدار پیش‌فرض را می‌دهد."""
    try:
        if _config is None:
            return default
        return getattr(_config, name, default)
    except Exception:
        return default


def cfg_num(name, default):
    try:
        return float(cfg(name, default))
    except Exception:
        return float(default)


# ---------------------------------------------------------------- زمان
def now_local():
    """ساعت محلی سیستم (بدون منطقه‌ی زمانی)؛ همان چیزی که تا حالا در دیتابیس ذخیره شده است."""
    return datetime.now()


def clock_skew_seconds():
    """اختلاف ساعت سیستم با ساعت تهران به ثانیه (مثبت یعنی ساعت سیستم جلوتر است)."""
    try:
        tehran = datetime.now(TEHRAN).replace(tzinfo=None)
        return (datetime.now() - tehran).total_seconds()
    except Exception:
        return None


# ---------------------------------------------------------------- متن فارسی و عدد
_FA_MAP = str.maketrans({
    "\u064a": "\u06cc",   # ي عربی -> ی فارسی
    "\u0649": "\u06cc",   # ى -> ی
    "\u0643": "\u06a9",   # ك عربی -> ک فارسی
    "\u0629": "\u0647",   # ة -> ه
    "\u0640": "",         # کشیده
    "\u200e": "", "\u200f": "",
    "\u202a": "", "\u202b": "", "\u202c": "", "\u202d": "", "\u202e": "",
    "\ufeff": "",
})
_DIGITS = str.maketrans("۰۱۲۳۴۵۶۷۸۹٠١٢٣٤٥٦٧٨٩", "01234567890123456789")


def normalize_fa(text):
    """یکسان‌سازی حروف عربی و فارسی و حذف نویسه‌های نامرئی جهت‌دهی."""
    try:
        if text is None:
            return ""
        return " ".join(str(text).translate(_FA_MAP).split())
    except Exception:
        return ""


def fa_key(text):
    """کلید مقایسه‌ی نام‌ها: بدون فاصله و نیم‌فاصله."""
    return normalize_fa(text).replace("\u200c", "").replace(" ", "")


def to_num(v, allow_negative=False):
    """تبدیل امن به عدد؛ NaN، بی‌نهایت، منفی (در صورت ممنوع بودن) و متن نامعتبر = None."""
    try:
        if v is None or isinstance(v, bool):
            return None
        if isinstance(v, str):
            s = v.translate(_DIGITS)
            s = s.replace(",", "").replace("\u066c", "").replace("\u060c", "")
            s = s.replace("\u066b", ".").strip()
            if not s:
                return None
            x = float(s)
        else:
            x = float(v)
        if math.isnan(x) or math.isinf(x):
            return None
        if x < 0 and not allow_negative:
            return None
        return x
    except Exception:
        return None


def pos_or_none(v):
    """فقط عدد مثبت را نگه می‌دارد (قیمت صفر در بورس یعنی «نامشخص»)."""
    x = to_num(v)
    return x if (x is not None and x > 0) else None


def safe_div(a, b, default=None):
    try:
        if a is None or b is None or b == 0:
            return default
        r = a / b
        if math.isnan(r) or math.isinf(r):
            return default
        return r
    except Exception:
        return default


# ---------------------------------------------------------------- دیتابیس
def connect_db(path, timeout=30):
    """اتصال به SQLite با تنظیمات ایمن برای قطعی برق و خواندن هم‌زمان."""
    con = sqlite3.connect(path, timeout=timeout)
    for pragma in ("PRAGMA journal_mode=WAL",
                   "PRAGMA synchronous=FULL",
                   "PRAGMA busy_timeout=%d" % int(timeout * 1000)):
        try:
            con.execute(pragma).fetchall()
        except sqlite3.Error:
            pass
    return con


def table_columns(con, table):
    try:
        return [r[1] for r in con.execute("PRAGMA table_info(%s)" % table)]
    except Exception:
        return []


def table_exists(con, table):
    try:
        row = con.execute("SELECT name FROM sqlite_master WHERE type='table' AND name=?",
                          (table,)).fetchone()
        return row is not None
    except Exception:
        return False


def ensure_column(con, table, column, decl):
    try:
        if column not in table_columns(con, table):
            con.execute("ALTER TABLE %s ADD COLUMN %s %s" % (table, column, decl))
            con.commit()
        return True
    except sqlite3.Error:
        return False


def run_with_retry(fn, tries=5, delay=0.6):
    """اگر دیتابیس قفل بود چند بار دوباره تلاش می‌کند؛ خطای دیگر یا تمام‌شدن تلاش‌ها را بالا می‌دهد."""
    last = None
    for i in range(tries):
        try:
            return fn()
        except sqlite3.OperationalError as e:
            last = e
            msg = str(e).lower()
            if "locked" in msg or "busy" in msg:
                time.sleep(delay * (i + 1))
                continue
            raise
    if last is not None:
        raise last


# ---------------------------------------------------------------- لاگ
class SQLiteLogHandler(logging.Handler):
    """لاگ‌های WARNING به بالا (و پیام‌هایی که to_db=True دارند) را در جدول event_log می‌نویسد."""

    def __init__(self, db_path):
        super().__init__(logging.INFO)
        self.db_path = db_path
        self._con = None
        self._mutex = threading.Lock()
        self._next_try = 0.0

    def _connect(self):
        con = sqlite3.connect(self.db_path, timeout=10, check_same_thread=False)
        try:
            con.execute("PRAGMA busy_timeout=10000")
        except sqlite3.Error:
            pass
        con.execute("CREATE TABLE IF NOT EXISTS event_log "
                    "(ts TEXT, level TEXT, component TEXT, message TEXT)")
        con.commit()
        return con

    def emit(self, record):
        try:
            if not (record.levelno >= logging.WARNING or getattr(record, "to_db", False)):
                return
            if time.time() < self._next_try:
                return
            msg = record.getMessage()
            if record.exc_info and record.exc_info[1] is not None:
                msg += " | %s: %s" % (type(record.exc_info[1]).__name__, record.exc_info[1])
            with self._mutex:
                if self._con is None:
                    self._con = self._connect()
                self._con.execute(
                    "INSERT INTO event_log (ts, level, component, message) VALUES (?,?,?,?)",
                    (now_local().strftime("%Y-%m-%d %H:%M:%S"), record.levelname,
                     record.name.replace("bourse.", ""), msg[:2000]))
                self._con.commit()
        except Exception:
            self._next_try = time.time() + 60
            try:
                if self._con is not None:
                    self._con.close()
            except Exception:
                pass
            self._con = None


def setup_logging(component, db_path=None, log_dir=None, console=True):
    """لاگر پروژه: فایل چرخشی logs/<component>.log + کنسول + (اختیاری) جدول event_log."""
    logger = logging.getLogger("bourse." + component)
    if logger.handlers:
        return logger
    logger.setLevel(logging.INFO)
    logger.propagate = False

    log_dir = log_dir or cfg("LOG_DIR", "logs")
    try:
        os.makedirs(log_dir, exist_ok=True)
        fh = logging.handlers.RotatingFileHandler(
            os.path.join(log_dir, component + ".log"),
            maxBytes=3_000_000, backupCount=5, encoding="utf-8")
        fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
        logger.addHandler(fh)
    except Exception as e:
        print("هشدار: فایل لاگ ساخته نشد:", e)

    if console:
        try:
            sh = logging.StreamHandler(sys.stdout)
            sh.setFormatter(logging.Formatter("%(message)s"))
            logger.addHandler(sh)
        except Exception:
            pass

    if db_path:
        try:
            logger.addHandler(SQLiteLogHandler(db_path))
        except Exception:
            pass
    return logger


# ---------------------------------------------------------------- قفل تک‌نمونه
class AlreadyRunning(Exception):
    pass


class SingleInstance:
    """جلوگیری از اجرای هم‌زمان دو نسخه‌ی یک برنامه. قفل با بسته‌شدن برنامه (حتی با قطعی برق) آزاد می‌شود."""

    def __init__(self, name, lock_dir=None):
        self.path = os.path.join(lock_dir or cfg("LOCK_DIR", "locks"), name + ".lock")
        self._fh = None

    def __enter__(self):
        os.makedirs(os.path.dirname(self.path) or ".", exist_ok=True)
        fh = open(self.path, "a+")
        try:
            if os.name == "nt":
                import msvcrt
                fh.seek(0)
                msvcrt.locking(fh.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(fh.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except OSError:
            fh.close()
            raise AlreadyRunning(self.path)
        self._fh = fh
        return self

    def __exit__(self, *exc):
        try:
            if self._fh is not None:
                if os.name == "nt":
                    import msvcrt
                    self._fh.seek(0)
                    msvcrt.locking(self._fh.fileno(), msvcrt.LK_UNLCK, 1)
                self._fh.close()
        except Exception:
            pass
        self._fh = None
        return False


# ---------------------------------------------------------------- پشتیبان‌گیری
def list_backups(db_path, backup_dir=None):
    try:
        backup_dir = backup_dir or cfg("BACKUP_DIR", "backups")
        base = os.path.splitext(os.path.basename(db_path))[0]
        return sorted(glob.glob(os.path.join(backup_dir, base + "_*.db")))
    except Exception:
        return []


def backup_database(db_path, backup_dir=None, keep=None, logger=None):
    """نسخه‌ی پشتیبان سالم از دیتابیس می‌سازد، سلامتش را می‌سنجد و نسخه‌های قدیمی را پاک می‌کند."""
    def _log(level, msg):
        try:
            if logger is not None:
                logger.log(level, msg, extra={"to_db": True})
            else:
                print(msg)
        except Exception:
            pass

    tmp_path = None
    try:
        if not os.path.exists(db_path):
            _log(logging.WARNING, "BACKUP: فایل دیتابیس پیدا نشد: %s" % db_path)
            return None
        backup_dir = backup_dir or cfg("BACKUP_DIR", "backups")
        keep = int(keep or cfg_num("BACKUP_KEEP", 14))
        os.makedirs(backup_dir, exist_ok=True)

        base = os.path.splitext(os.path.basename(db_path))[0]
        stamp = now_local().strftime("%Y%m%d_%H%M%S")
        dst_path = os.path.join(backup_dir, "%s_%s.db" % (base, stamp))
        tmp_path = dst_path + ".tmp"

        src = sqlite3.connect(db_path, timeout=30)
        try:
            dst = sqlite3.connect(tmp_path)
            try:
                src.backup(dst)
                dst.commit()
                check = dst.execute("PRAGMA quick_check").fetchone()
            finally:
                dst.close()
        finally:
            src.close()

        if not check or check[0] != "ok":
            _log(logging.ERROR, "BACKUP: نسخه‌ی پشتیبان سالم نبود و حذف شد (%s)" % (check,))
            try:
                os.remove(tmp_path)
            except Exception:
                pass
            return None

        os.replace(tmp_path, dst_path)
        for old in list_backups(db_path, backup_dir)[:-keep]:
            try:
                os.remove(old)
            except Exception:
                pass
        size_mb = os.path.getsize(dst_path) / 1e6
        _log(logging.INFO, "BACKUP: ساخته شد %s (%.1f MB)" % (dst_path, size_mb))
        return dst_path
    except Exception as e:
        _log(logging.ERROR, "BACKUP: خطا: %s: %s" % (type(e).__name__, e))
        try:
            if tmp_path and os.path.exists(tmp_path):
                os.remove(tmp_path)
        except Exception:
            pass
        return None