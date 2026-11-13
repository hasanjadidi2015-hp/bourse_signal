# -*- coding: utf-8 -*-
"""
after_market.py: همه‌ی کارهای بعد از بازار با یک دستور.

ترتیب: پشتیبان‌گیری ← daily_bars.py ← signal_features.py ← (اختیاری) train_eval.py
هر مرحله جدا اجرا می‌شود؛ اگر یکی خطا بدهد بقیه ادامه پیدا می‌کنند و در پایان خلاصه چاپ می‌شود.

اجرا (بعد از Ctrl+C زدن روی collector و dashboard):
    newenv\\Scripts\\python.exe after_market.py
    newenv\\Scripts\\python.exe after_market.py --train      ← با آموزش/ارزیابی مدل (هفته‌ای یک بار)
"""
import argparse
import os
import subprocess
import sys
import time

import bourse_common as bc

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH = bc.cfg("DB_PATH", "signal_data.db")
STEP_TIMEOUT = 1800     # ثانیه؛ بیشتر از این یعنی برنامه گیر کرده است


def collector_running():
    """True اگر collector هنوز باز است، False اگر نیست، None اگر نشد فهمید."""
    try:
        with bc.SingleInstance("collector"):
            return False
    except bc.AlreadyRunning:
        return True
    except Exception:
        return None


def run_script(script, log):
    path = os.path.join(BASE_DIR, script)
    if not os.path.exists(path):
        return "پیدا نشد", 0.0
    t0 = time.time()
    try:
        r = subprocess.run([sys.executable, path], cwd=BASE_DIR, timeout=STEP_TIMEOUT)
        status = "موفق" if r.returncode == 0 else "خطا (کد %s)" % r.returncode
    except subprocess.TimeoutExpired:
        status = "بیش از %d ثانیه طول کشید و متوقف شد" % STEP_TIMEOUT
    except Exception as e:
        status = "اجرا نشد: %s" % type(e).__name__
    return status, time.time() - t0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", action="store_true", help="train_eval.py هم اجرا شود")
    ap.add_argument("--skip-backup", action="store_true")
    args = ap.parse_args()

    log = bc.setup_logging("after_market", db_path=DB_PATH)
    log.info("AFTER_MARKET_START %s" % bc.now_local().strftime("%Y-%m-%d %H:%M"),
             extra={"to_db": True})

    running = collector_running()
    if running:
        log.warning("collector هنوز در حال اجراست. اگر بازار تمام شده، اول در پنجره‌ی آن Ctrl+C بزن؛ "
                    "ادامه می‌دهم ولی کندل امروز ممکن است ناقص باشد.")

    results = []

    if not args.skip_backup:
        t0 = time.time()
        path = bc.backup_database(DB_PATH, logger=log)
        results.append(("پشتیبان‌گیری", "موفق" if path else "ناموفق", time.time() - t0))

    steps = [("daily_bars.py", "کندل‌های روزانه"), ("signal_features.py", "ساخت ویژگی‌ها")]
    if args.train:
        steps.append(("train_eval.py", "ارزیابی مدل"))

    for script, title in steps:
        print("\n" + "=" * 10 + " %s (%s) " % (title, script) + "=" * 10)
        status, secs = run_script(script, log)
        results.append((title, status, secs))
        level = "INFO" if status == "موفق" else "WARNING"
        getattr(log, level.lower())("AFTER_MARKET_STEP %s: %s (%.0f ثانیه)" % (script, status, secs),
                                     extra={"to_db": True})

    print("\n" + "=" * 10 + " خلاصه " + "=" * 10)
    ok = True
    for title, status, secs in results:
        print("  %-18s %-28s %5.0f ثانیه" % (title, status, secs))
        ok = ok and status == "موفق"
    if running:
        print("  توجه: collector هنوز باز بود.")
    print("\nهمه‌چیز موفق بود." if ok else "\nبعضی مراحل موفق نبودند؛ متن خطا را بالاتر ببین و برایم بفرست.")
    log.info("AFTER_MARKET_END %s" % ("OK" if ok else "WITH_ERRORS"), extra={"to_db": True})
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())