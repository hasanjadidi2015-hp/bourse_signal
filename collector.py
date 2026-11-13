# -*- coding: utf-8 -*-
"""
collector.py (نسخه‌ی ۲، مقاوم در برابر خطا)

جمع‌آوری دفتر سفارش و قیمت سه نماد از TSETMC هر POLL_SECONDS ثانیه.

اجرای عادی (فقط ساعت معاملات):
    newenv\\Scripts\\python.exe collector.py

تست خارج از ساعت بازار (روی دیتابیس جدا، داده‌ی اصلی دست نمی‌خورد):
    newenv\\Scripts\\python.exe collector.py --test
"""
import argparse
import os
import random
import sqlite3
import sys
import time

import requests

import bourse_common as bc

SYMBOLS = {bc.normalize_fa(k): str(v).strip()
           for k, v in (bc.cfg("SYMBOLS", {}) or {}).items()}
DB_PATH = bc.cfg("DB_PATH", "signal_data.db")
POLL_SECONDS = bc.cfg_num("POLL_SECONDS", 10)
MARKET_OPEN = str(bc.cfg("MARKET_OPEN", "08:45"))
MARKET_CLOSE = str(bc.cfg("MARKET_CLOSE", "12:30"))
MARKET_WEEKDAYS = list(bc.cfg("MARKET_WEEKDAYS", [5, 6, 0, 1, 2]))

URL_BOOK = "https://cdn.tsetmc.com/api/BestLimits/{}"
URL_PRICE = "https://cdn.tsetmc.com/api/ClosingPrice/GetClosingPriceInfo/{}"
HEADERS = {
    "User-Agent": ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                   "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0 Safari/537.36"),
    "Accept": "application/json, text/plain, */*",
    "Referer": "https://www.tsetmc.com/",
}

CLOSED_SLEEP = 30           # ثانیه‌ی انتظار خارج از ساعت بازار
HEARTBEAT_SEC = 300         # هر ۵ دقیقه یک خط «زنده‌ام» در لاگ دیتابیس
OUTAGE_STREAK = 3           # بعد از این‌همه چرخه‌ی کاملاً ناموفق «قطعی شبکه» اعلام می‌شود
WARN_INTERVAL = 60          # جلوگیری از تکرار پیام هشدار یکسان (ثانیه)
PRICE_STALE_AFTER = "09:05"  # بعد از این ساعت، قیمت مربوط به روز قبل «کهنه» است


# ---------------------------------------------------------------- دیتابیس
SCHEMA = [
    """CREATE TABLE IF NOT EXISTS book (
        ts TEXT, symbol TEXT, level INTEGER,
        bid_price REAL, bid_vol REAL, bid_cnt REAL,
        ask_price REAL, ask_vol REAL, ask_cnt REAL)""",
    """CREATE TABLE IF NOT EXISTS price (
        ts TEXT, symbol TEXT, last_price REAL, closing_price REAL,
        day_high REAL, day_low REAL, yesterday_price REAL,
        cum_volume REAL, cum_trades REAL, market_cap REAL)""",
    """CREATE TABLE IF NOT EXISTS collector_cycles (
        ts TEXT, symbol TEXT, book_ok INTEGER, book_levels INTEGER,
        price_ok INTEGER, latency_ms INTEGER, quality TEXT, error TEXT)""",
    """CREATE TABLE IF NOT EXISTS event_log (
        ts TEXT, level TEXT, component TEXT, message TEXT)""",
    "CREATE INDEX IF NOT EXISTS idx_book_sym_ts ON book(symbol, ts)",
    "CREATE INDEX IF NOT EXISTS idx_price_sym_ts ON price(symbol, ts)",
    "CREATE INDEX IF NOT EXISTS idx_cycles_sym_ts ON collector_cycles(symbol, ts)",
]
EXTRA_PRICE_COLUMNS = [
    ("instr_state", "TEXT"),
    ("instr_state_title", "TEXT"),
    ("d_even", "INTEGER"),
    ("h_even", "INTEGER"),
]


def open_db(path):
    con = bc.connect_db(path)
    with con:
        for stmt in SCHEMA:
            con.execute(stmt)
    for col, decl in EXTRA_PRICE_COLUMNS:
        bc.ensure_column(con, "price", col, decl)
    return con


def open_db_safe(path, log):
    for attempt in range(1, 4):
        try:
            return open_db(path)
        except Exception as e:
            log.error("DB_OPEN_FAIL تلاش %d: %s: %s" % (attempt, type(e).__name__, e))
            time.sleep(2 * attempt)
    return None


# ---------------------------------------------------------------- شبکه
class Http:
    """درخواست با Session، تلاش مجدد با فاصله‌ی افزایشی و مهار هر نوع خطا."""

    def __init__(self):
        self.session = None
        self.reset()

    def reset(self):
        try:
            if self.session is not None:
                self.session.close()
        except Exception:
            pass
        self.session = requests.Session()
        self.session.headers.update(HEADERS)

    def get_json(self, url, tries=3):
        """برمی‌گرداند: (داده، None) یا (None، متن خطا)."""
        last_err = "unknown"
        for attempt in range(1, tries + 1):
            code = None
            try:
                r = self.session.get(url, timeout=(5, 10))
                code = r.status_code
                if code == 200:
                    text = (r.text or "").strip()
                    if not text:
                        last_err = "empty body"
                    elif text[0] not in "{[":
                        last_err = "non-json body"
                    else:
                        try:
                            return r.json(), None
                        except ValueError as e:
                            last_err = "invalid json: %s" % str(e)[:80]
                else:
                    last_err = "http %s" % code
            except requests.exceptions.Timeout:
                last_err = "timeout"
            except requests.exceptions.ConnectionError:
                last_err = "connection error"
                self.reset()
            except requests.exceptions.RequestException as e:
                last_err = "request error: %s" % type(e).__name__
            except Exception as e:
                last_err = "unexpected: %s: %s" % (type(e).__name__, str(e)[:80])
            if attempt < tries:
                delay = min(8.0, 1.5 * (2 ** (attempt - 1))) + random.uniform(0, 0.5)
                if code in (403, 429):
                    delay *= 2
                time.sleep(delay)
        return None, last_err


# ---------------------------------------------------------------- تحلیل پاسخ‌ها
def parse_book(payload):
    """(ردیف‌ها، خطا). هر ردیف: (level, bid_price, bid_vol, bid_cnt, ask_price, ask_vol, ask_cnt)."""
    levels = None
    if isinstance(payload, dict):
        levels = (payload.get("bestLimitsInfo") or payload.get("bestLimits")
                  or payload.get("bestLimitsData"))
    elif isinstance(payload, list):
        levels = payload
    if not isinstance(levels, list) or not levels:
        return [], "no_levels"

    rows = {}
    for lv in levels:
        if not isinstance(lv, dict):
            continue
        n = bc.to_num(lv.get("number"))
        if n is None or n < 1 or n > 20:
            continue
        n = int(n)
        if n in rows:
            continue
        rows[n] = (n,
                   bc.to_num(lv.get("pMeDem")), bc.to_num(lv.get("qTitMeDem")),
                   bc.to_num(lv.get("zOrdMeDem")),
                   bc.to_num(lv.get("pMeOf")), bc.to_num(lv.get("qTitMeOf")),
                   bc.to_num(lv.get("zOrdMeOf")))
    if not rows:
        return [], "no_valid_levels"
    return [rows[k] for k in sorted(rows)], None


def book_valid(rows):
    """دفتر وقتی معتبر است که سطح اول دست‌کم یک قیمت مثبت داشته باشد."""
    if not rows:
        return False
    first = rows[0]
    return (first[1] or 0) > 0 or (first[4] or 0) > 0


def book_quality(rows):
    first = rows[0]
    bid, ask = first[1] or 0, first[4] or 0
    if bid > 0 and ask > 0:
        if bid > ask:
            return "crossed"
        if bid == ask:
            return "locked"
        return "ok"
    return "one_sided"


def parse_price(payload):
    info = None
    if isinstance(payload, dict):
        info = payload.get("closingPriceInfo")
        if not isinstance(info, dict):
            info = payload
    if not isinstance(info, dict):
        return None
    state = info.get("instrumentState")
    state = state if isinstance(state, dict) else {}
    code = state.get("cEtaval")
    title = state.get("cEtavalTitle")
    d_even = bc.to_num(info.get("dEven"))
    h_even = bc.to_num(info.get("hEven"))
    return {
        "last": bc.pos_or_none(info.get("pDrCotVal")),
        "closing": bc.pos_or_none(info.get("pClosing")),
        "high": bc.pos_or_none(info.get("priceMax")),
        "low": bc.pos_or_none(info.get("priceMin")),
        "yesterday": bc.pos_or_none(info.get("priceYesterday")),
        "cum_volume": bc.to_num(info.get("qTotTran5J")),
        "cum_trades": bc.to_num(info.get("zTotTran")),
        "market_cap": bc.to_num(info.get("qTotCap")),
        "state": str(code).strip() if code is not None else None,
        "state_title": bc.normalize_fa(title) if title is not None else None,
        "d_even": int(d_even) if d_even is not None else None,
        "h_even": int(h_even) if h_even is not None else None,
    }


# ---------------------------------------------------------------- یک نماد
def process_symbol(http, con, name, code, now_fn):
    t0 = time.monotonic()
    ts = now_fn().strftime("%Y-%m-%d %H:%M:%S")
    info = {"ts": ts, "symbol": name, "book_ok": 0, "levels": 0, "price_ok": 0,
            "quality": "", "error": "", "bid": None, "ask": None, "last": None,
            "stale_price": False, "write_error": None}
    book_rows = []
    price = None

    try:
        data, err = http.get_json(URL_BOOK.format(code))
        if err:
            info["error"] += "book:%s;" % err
        else:
            rows, perr = parse_book(data)
            if perr:
                info["error"] += "book:%s;" % perr
            elif not book_valid(rows):
                info["error"] += "book:empty;"
            else:
                book_rows = rows
                info["book_ok"] = 1
                info["levels"] = len(rows)
                info["quality"] = book_quality(rows)
                info["bid"] = rows[0][1]
                info["ask"] = rows[0][4]
    except Exception as e:
        info["error"] += "book:crash %s;" % type(e).__name__

    try:
        data, err = http.get_json(URL_PRICE.format(code))
        if err:
            info["error"] += "price:%s;" % err
        else:
            p = parse_price(data)
            if p is None:
                info["error"] += "price:bad_shape;"
            elif p["last"] is None:
                info["error"] += "price:no_last;"
            else:
                price = p
                info["price_ok"] = 1
                info["last"] = p["last"]
                today_int = int(ts[:4] + ts[5:7] + ts[8:10])
                if (ts[11:16] >= PRICE_STALE_AFTER and p["d_even"] is not None
                        and p["d_even"] != today_int):
                    info["stale_price"] = True
    except Exception as e:
        info["error"] += "price:crash %s;" % type(e).__name__

    info["latency_ms"] = int((time.monotonic() - t0) * 1000)

    def _write():
        with con:
            if book_rows:
                con.executemany(
                    "INSERT INTO book (ts, symbol, level, bid_price, bid_vol, bid_cnt, "
                    "ask_price, ask_vol, ask_cnt) VALUES (?,?,?,?,?,?,?,?,?)",
                    [(ts, name) + r for r in book_rows])
            if price is not None:
                con.execute(
                    "INSERT INTO price (ts, symbol, last_price, closing_price, day_high, "
                    "day_low, yesterday_price, cum_volume, cum_trades, market_cap, "
                    "instr_state, instr_state_title, d_even, h_even) "
                    "VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?)",
                    (ts, name, price["last"], price["closing"], price["high"],
                     price["low"], price["yesterday"], price["cum_volume"],
                     price["cum_trades"], price["market_cap"], price["state"],
                     price["state_title"], price["d_even"], price["h_even"]))
            con.execute(
                "INSERT INTO collector_cycles (ts, symbol, book_ok, book_levels, "
                "price_ok, latency_ms, quality, error) VALUES (?,?,?,?,?,?,?,?)",
                (ts, name, info["book_ok"], info["levels"], info["price_ok"],
                 info["latency_ms"], info["quality"], info["error"][:300]))

    try:
        bc.run_with_retry(_write)
    except Exception as e:
        info["write_error"] = "%s: %s" % (type(e).__name__, str(e)[:120])
    return info


def market_open(now):
    if now.weekday() not in MARKET_WEEKDAYS:
        return False
    t = now.strftime("%H:%M")
    return MARKET_OPEN <= t <= MARKET_CLOSE


# ---------------------------------------------------------------- حلقه‌ی اصلی
class Runner:
    def __init__(self, args, db_path, log):
        self.args = args
        self.db_path = db_path
        self.log = log
        self.http = Http()
        self.con = None
        self.was_open = None
        self.fail_streak = 0
        self.outage_start = None
        self.crashes = 0
        self.write_fails = 0
        self.cycles = 0
        self.ok_cycles = 0
        self.last_heartbeat = time.time()
        self.last_warn = {}

    def warn_limited(self, key, msg):
        now = time.time()
        if now - self.last_warn.get(key, 0) >= WARN_INTERVAL:
            self.last_warn[key] = now
            self.log.warning(msg)

    def reopen_db(self):
        try:
            if self.con is not None:
                self.con.close()
        except Exception:
            pass
        self.con = open_db_safe(self.db_path, self.log)
        return self.con is not None

    def on_market_state(self, is_open):
        if is_open and self.was_open is not True:
            self.log.info("MARKET_OPEN شروع جمع‌آوری", extra={"to_db": True})
        elif (not is_open) and self.was_open is True:
            self.log.info("MARKET_CLOSED پایان ساعت معاملات؛ پشتیبان‌گیری...",
                          extra={"to_db": True})
            bc.backup_database(self.db_path, logger=self.log)
            try:
                self.con.execute("PRAGMA wal_checkpoint(TRUNCATE)").fetchall()
            except Exception:
                pass
        elif (not is_open) and self.was_open is None:
            self.log.info("بازار بسته است. منتظر ساعت معاملات...")
        self.was_open = is_open

    def one_cycle(self):
        infos = []
        for name, code in SYMBOLS.items():
            try:
                info = process_symbol(self.http, self.con, name, code, bc.now_local)
            except Exception as e:
                self.log.exception("SYMBOL_CRASH %s" % name)
                info = {"ts": "", "symbol": name, "book_ok": 0, "price_ok": 0,
                        "quality": "", "error": "crash %s" % type(e).__name__,
                        "bid": None, "ask": None, "last": None,
                        "stale_price": False, "write_error": None}
            infos.append(info)
            self.report_symbol(info)

        any_ok = any(i["book_ok"] or i["price_ok"] for i in infos)
        self.cycles += 1
        if any_ok:
            self.ok_cycles += 1
            if self.outage_start is not None:
                self.log.warning("NETWORK_UP اتصال برقرار شد؛ مدت قطعی %.0f ثانیه"
                                 % (time.time() - self.outage_start))
            self.fail_streak = 0
            self.outage_start = None
        else:
            self.fail_streak += 1
            if self.fail_streak == OUTAGE_STREAK and self.outage_start is None:
                self.outage_start = time.time()
                self.log.warning("NETWORK_DOWN %d چرخه‌ی پشت‌سرهم بدون هیچ داده" % self.fail_streak)

        if any(i.get("write_error") for i in infos):
            self.write_fails += 1
            if self.write_fails >= 10:
                self.log.error("DB_REOPEN خطای پشت‌سرهم در نوشتن؛ اتصال دیتابیس دوباره باز می‌شود")
                self.reopen_db()
                self.write_fails = 0
        else:
            self.write_fails = 0
        return infos

    def report_symbol(self, info):
        base = "%s %s: خرید۱=%s فروش۱=%s آخرین=%s" % (
            info["ts"], info["symbol"], info["bid"], info["ask"], info["last"])
        problems = []
        if info["error"]:
            problems.append("خطا: " + info["error"])
        if info["quality"] in ("crossed", "locked", "one_sided"):
            problems.append("دفتر " + info["quality"])
        if info["stale_price"]:
            problems.append("قیمت مربوط به روز قبل است")
        if info["write_error"]:
            problems.append("خطای ذخیره: " + info["write_error"])

        if info["error"] or info["write_error"]:
            self.warn_limited("err:" + info["symbol"], base + " | " + " | ".join(problems))
        elif info["stale_price"]:
            self.warn_limited("stale:" + info["symbol"], base + " | " + " | ".join(problems))
        else:
            self.log.info(base + ((" | " + " | ".join(problems)) if problems else ""))

    def heartbeat(self):
        if time.time() - self.last_heartbeat >= HEARTBEAT_SEC:
            self.last_heartbeat = time.time()
            self.log.info("HEARTBEAT چرخه‌ها=%d موفق=%d" % (self.cycles, self.ok_cycles),
                          extra={"to_db": True})

    def loop(self):
        max_cycles = self.args.cycles if self.args.test else None
        while True:
            try:
                now = bc.now_local()
                is_open = True if self.args.test else market_open(now)
                if not self.args.test:
                    self.on_market_state(is_open)
                if not is_open:
                    time.sleep(CLOSED_SLEEP)
                    continue

                if self.con is None and not self.reopen_db():
                    time.sleep(10)
                    continue

                self.one_cycle()
                self.crashes = 0
                self.heartbeat()

                if max_cycles is not None and self.cycles >= max_cycles:
                    return
                time.sleep(POLL_SECONDS if not self.args.test else 2)
                if self.fail_streak:
                    time.sleep(min(30, 5 * self.fail_streak))
            except KeyboardInterrupt:
                raise
            except Exception:
                self.crashes += 1
                self.log.exception("CYCLE_CRASH خطای پیش‌بینی‌نشده؛ ادامه می‌دهیم (بار %d)" % self.crashes)
                time.sleep(min(60, 5 * self.crashes))
                self.reopen_db()

    def run(self):
        skew = bc.clock_skew_seconds()
        self.log.info("COLLECTOR_START پایتون %s | دیتابیس %s | نمادها %s"
                      % (sys.version.split()[0], self.db_path, list(SYMBOLS)),
                      extra={"to_db": True})
        if not SYMBOLS:
            self.log.error("هیچ نمادی در config.py تعریف نشده است.")
            return 1
        if skew is not None and abs(skew) > 120:
            self.log.warning("CLOCK_SKEW ساعت سیستم %.0f ثانیه با ساعت تهران اختلاف دارد؛ "
                             "ساعت ویندوز را بررسی کن." % skew)
        if not self.reopen_db():
            self.log.error("دیتابیس باز نشد؛ برنامه متوقف شد.")
            return 1
        try:
            self.loop()
        except KeyboardInterrupt:
            self.log.info("COLLECTOR_STOP توقف با Ctrl+C", extra={"to_db": True})
        finally:
            try:
                if self.con is not None:
                    self.con.close()
            except Exception:
                pass
        return 0


def print_test_summary(db_path):
    try:
        con = sqlite3.connect(db_path)
        print("\n=== خلاصه‌ی تست (%s) ===" % db_path)
        for table in ("book", "price", "collector_cycles"):
            n = con.execute("SELECT COUNT(*) FROM %s" % table).fetchone()[0]
            print("  %s: %d ردیف" % (table, n))
        rows = con.execute(
            "SELECT symbol, SUM(book_ok), SUM(price_ok), COUNT(*), ROUND(AVG(latency_ms)) "
            "FROM collector_cycles GROUP BY symbol").fetchall()
        for r in rows:
            print("  %s: دفتر موفق=%s قیمت موفق=%s از %s بار | میانگین تأخیر=%s ms" % r)
        last = con.execute(
            "SELECT symbol, last_price, instr_state, instr_state_title FROM price "
            "ORDER BY ts DESC LIMIT 3").fetchall()
        for r in last:
            print("  آخرین قیمت %s: %s | وضعیت نماد: %s %s" % r)
        con.close()
    except Exception as e:
        print("خلاصه‌ی تست ساخته نشد:", e)


def parse_args():
    ap = argparse.ArgumentParser(description="جمع‌آوری دفتر سفارش و قیمت")
    ap.add_argument("--test", action="store_true",
                    help="تست خارج از ساعت بازار روی دیتابیس جدا (test_data.db)")
    ap.add_argument("--cycles", type=int, default=3, help="تعداد چرخه در حالت تست")
    ap.add_argument("--no-lock", action="store_true", help="غیرفعال‌کردن قفل تک‌نمونه")
    return ap.parse_args()


def main():
    args = parse_args()
    if args.test:
        db_path = os.path.join(os.path.dirname(DB_PATH) or ".", "test_data.db")
    else:
        db_path = DB_PATH
    log = bc.setup_logging("collector_test" if args.test else "collector", db_path=db_path)

    def go():
        code = Runner(args, db_path, log).run()
        if args.test:
            print_test_summary(db_path)
        return code

    if args.no_lock:
        return go()
    try:
        with bc.SingleInstance("collector_test" if args.test else "collector"):
            return go()
    except bc.AlreadyRunning:
        print("collector از قبل در یک پنجره‌ی دیگر در حال اجراست. "
              "آن پنجره را ببند، یا برای نادیده‌گرفتن قفل از --no-lock استفاده کن.")
        return 2
    except Exception as e:
        print("خطای غیرمنتظره در راه‌اندازی:", type(e).__name__, e)
        return 1


if __name__ == "__main__":
    sys.exit(main())