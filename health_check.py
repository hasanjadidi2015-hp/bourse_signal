# -*- coding: utf-8 -*-
"""
health_check.py: بررسی سلامت داده و سیستم (فقط می‌خواند، چیزی را تغییر نمی‌دهد).

اجرا:
    newenv\\Scripts\\python.exe health_check.py            ← ۷ روز اخیر
    newenv\\Scripts\\python.exe health_check.py --days 14

خروجی روی صفحه چاپ می‌شود و در health_report.txt هم ذخیره می‌شود.
"""
import argparse
import os
import sqlite3
import sys
from datetime import datetime, timedelta

import numpy as np
import pandas as pd

import bourse_common as bc

DB_PATH = bc.cfg("DB_PATH", "signal_data.db")
TRADING_START = "09:00"
MARKET_CLOSE = str(bc.cfg("MARKET_CLOSE", "12:30"))
GAP_SEC = 60
STALL_MIN = 10
REPORT_FILE = "health_report.txt"

lines = []
problems = []


def say(msg=""):
    print(msg)
    lines.append(msg)


def problem(msg):
    problems.append(msg)


def section(title, fn, *args):
    say("")
    say("=" * 8 + " " + title + " " + "=" * 8)
    try:
        fn(*args)
    except Exception as e:
        say("  خطا در این بخش: %s: %s" % (type(e).__name__, e))
        problem("بخش «%s» اجرا نشد: %s" % (title, type(e).__name__))


def pct(a, b):
    return "%.1f%%" % (100.0 * a / b) if b else "-"


# ---------------------------------------------------------------- بخش‌ها
def sec_environment(con):
    say("  پایتون: %s | pandas: %s" % (sys.version.split()[0], pd.__version__))
    size = os.path.getsize(DB_PATH) / 1e6 if os.path.exists(DB_PATH) else 0
    say("  دیتابیس: %s (%.1f MB)" % (DB_PATH, size))
    mode = con.execute("PRAGMA journal_mode").fetchone()[0]
    say("  حالت ژورنال: %s %s" % (mode, "(مناسب)" if mode == "wal" else "(collector نسخه‌ی ۲ آن را wal می‌کند)"))
    chk = con.execute("PRAGMA quick_check").fetchone()[0]
    say("  بررسی یکپارچگی: %s" % chk)
    if chk != "ok":
        problem("یکپارچگی دیتابیس ok نیست: %s" % chk)
    skew = bc.clock_skew_seconds()
    if skew is None:
        say("  ساعت: مرجع، ساعت نمایش‌داده‌شده‌ی ویندوز است (مقایسه با بورس در بخش «ساعت» پایین‌تر)")
    else:
        say("  اختلاف ساعت UTC داخلی سیستم با تهران: %.0f ثانیه" % skew)


def sec_tables(con):
    for t in ("book", "price", "collector_cycles", "event_log"):
        if not bc.table_exists(con, t):
            say("  %s: جدول وجود ندارد" % t)
            if t in ("book", "price"):
                problem("جدول %s وجود ندارد" % t)
            continue
        n, a, b = con.execute("SELECT COUNT(*), MIN(ts), MAX(ts) FROM %s" % t).fetchone()
        say("  %-17s %8d ردیف | از %s تا %s" % (t, n, a, b))
    cols = bc.table_columns(con, "price")
    for c in ("instr_state", "instr_state_title", "d_even", "h_even"):
        if c not in cols:
            say("  ستون price.%s هنوز نیست (بعد از اولین اجرای collector نسخه‌ی ۲ ساخته می‌شود)" % c)


def load_frames(con, days):
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
    book = pd.read_sql_query(
        "SELECT ts, symbol, bid_price, ask_price FROM book WHERE level = 1 AND ts >= ?",
        con, params=(since,), parse_dates=["ts"])
    price = pd.read_sql_query(
        "SELECT ts, symbol, last_price, cum_volume FROM price WHERE ts >= ?",
        con, params=(since,), parse_dates=["ts"])
    return book, price


def longest_stall(p):
    """طولانی‌ترین بازه‌ای (دقیقه) که حجم تجمعی در ساعت معاملات تغییر نکرده است."""
    p = p.dropna(subset=["cum_volume"]).sort_values("ts")
    p = p[p.ts.dt.strftime("%H:%M") >= TRADING_START]
    p = p[p.ts.dt.strftime("%H:%M") <= MARKET_CLOSE]
    if len(p) < 2:
        return 0.0
    run = p.cum_volume.ne(p.cum_volume.shift()).cumsum()
    best = 0.0
    for _, g in p.groupby(run):
        best = max(best, (g.ts.max() - g.ts.min()).total_seconds() / 60.0)
    return best


def sec_daily(con, days):
    book, price = load_frames(con, days)
    if book.empty:
        say("  در این بازه داده‌ی دفتر سفارش نیست.")
        return
    book["day"] = book.ts.dt.normalize()
    price["day"] = price.ts.dt.normalize()
    book["hm"] = book.ts.dt.strftime("%H:%M")
    say("  نماد   | روز        | عکس  | اولین-آخرین   | وقفه>%ds (بزرگ‌ترین) | پیش‌گشایش | crossed/locked/یک‌طرفه (٪ بعد از ۹) | توقف حجم" % GAP_SEC)

    tot = {"pre": 0, "pre_cross": 0, "post": 0, "post_cross": 0}
    for (name, day), g in book.groupby(["symbol", "day"]):
        g = g.drop_duplicates("ts").sort_values("ts")
        n = len(g)
        gaps = g.ts.diff().dt.total_seconds().dropna()
        big = gaps[gaps > GAP_SEC]
        post = g[g.hm >= TRADING_START]
        pre = g[g.hm < TRADING_START]

        both = (post.bid_price > 0) & (post.ask_price > 0)
        crossed = int((both & (post.bid_price > post.ask_price)).sum())
        locked = int((both & (post.bid_price == post.ask_price)).sum())
        one = int((~both).sum())

        pboth = (pre.bid_price > 0) & (pre.ask_price > 0)
        pcross = int((pboth & (pre.bid_price >= pre.ask_price)).sum())
        tot["pre"] += len(pre)
        tot["pre_cross"] += pcross
        tot["post"] += len(post)
        tot["post_cross"] += crossed + locked

        pp = price[(price.symbol == name) & (price.day == day)]
        stall = longest_stall(pp) if len(pp) else 0.0
        last_t = g.ts.max().strftime("%H:%M")
        flag = ""
        if last_t < "12:20":
            flag += " [روز ناقص]"
            problem("%s %s: داده تا %s بیشتر نیست (روز ناقص)" % (name, day.date(), last_t))
        if stall >= STALL_MIN:
            problem("%s %s: حجم %.0f دقیقه ثابت بود (توقف ناظر یا بی‌معامله)" % (name, day.date(), stall))
        say("  %-6s | %s | %4d | %s-%s | %3d (%4.0fs)        | %4d      | %s/%s/%s | %.0f دقیقه%s" % (
            name, day.date(), n, g.ts.min().strftime("%H:%M"), last_t, len(big),
            big.max() if len(big) else 0, len(pre),
            pct(crossed, len(post)), pct(locked, len(post)), pct(one, len(post)), stall, flag))

    say("")
    say("  جمع‌بندی پیش‌گشایش: %d عکس قبل از %s؛ crossed/locked در آن‌ها: %s (بعد از ۹: %s)" % (
        tot["pre"], TRADING_START, pct(tot["pre_cross"], tot["pre"]), pct(tot["post_cross"], tot["post"])))
    if tot["pre"] > 0:
        say("  توضیح: عکس‌های پیش‌گشایش جمع‌آوری و در داشبورد نمایش داده می‌شوند، ولی signal_features نسخه‌ی ۳ "
            "آن‌ها را از ویژگی‌ها و آموزش حذف می‌کند (برچسب ۱۵ دقیقه‌ای‌شان از باز شدن بازار عبور می‌کند).")


def sec_collector(con, days):
    if not bc.table_exists(con, "collector_cycles"):
        say("  جدول collector_cycles هنوز نیست (collector نسخه‌ی ۲ را حداقل یک بار اجرا کن).")
        return
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
    df = pd.read_sql_query(
        "SELECT ts, symbol, book_ok, price_ok, latency_ms, quality, error FROM collector_cycles WHERE ts >= ?",
        con, params=(since,))
    if df.empty:
        say("  در این بازه چرخه‌ای ثبت نشده است.")
        return
    for name, g in df.groupby("symbol"):
        n = len(g)
        say("  %-6s: %d درخواست | دفتر موفق %s | قیمت موفق %s | تأخیر میانگین %.0f ms (p95 %.0f)" % (
            name, n, pct(int(g.book_ok.sum()), n), pct(int(g.price_ok.sum()), n),
            g.latency_ms.mean(), g.latency_ms.quantile(0.95)))
        if g.book_ok.mean() < 0.97 or g.price_ok.mean() < 0.97:
            problem("%s: نرخ موفقیت درخواست کمتر از ۹۷٪ است" % name)
    q = df[df.quality.isin(["crossed", "locked", "one_sided"])].groupby(["symbol", "quality"]).size()
    if len(q):
        say("  کیفیت دفتر سفارش (غیر از ok): " + "; ".join("%s/%s=%d" % (a, b, c) for (a, b), c in q.items()))
    errs = df[df.error.fillna("") != ""].error.str.split(";").explode()
    errs = errs[errs.str.strip() != ""].str.strip().value_counts().head(6)
    if len(errs):
        say("  پرتکرارترین خطاها:")
        for k, v in errs.items():
            say("    %5d × %s" % (v, k))


def sec_events(con, days):
    if not bc.table_exists(con, "event_log"):
        say("  جدول event_log هنوز نیست.")
        return
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
    lv = con.execute("SELECT level, COUNT(*) FROM event_log WHERE ts >= ? GROUP BY level", (since,)).fetchall()
    say("  تعداد رویدادها: " + (", ".join("%s=%d" % r for r in lv) if lv else "هیچ"))
    for key in ("NETWORK_DOWN", "NETWORK_UP", "CYCLE_CRASH", "SYMBOL_CRASH", "DB_REOPEN", "CLOCK_SKEW"):
        n = con.execute("SELECT COUNT(*) FROM event_log WHERE ts >= ? AND message LIKE ?",
                        (since, key + "%")).fetchone()[0]
        if n:
            say("  %s: %d بار" % (key, n))
            if key in ("CYCLE_CRASH", "SYMBOL_CRASH", "NETWORK_DOWN", "DB_REOPEN"):
                problem("%s: %d بار در %d روز اخیر" % (key, n, days))
    rows = con.execute("SELECT ts, level, substr(message,1,110) FROM event_log "
                       "WHERE ts >= ? AND level IN ('WARNING','ERROR') ORDER BY ts DESC LIMIT 8", (since,)).fetchall()
    if rows:
        say("  آخرین هشدارها و خطاها:")
        for r in rows:
            say("    %s %s %s" % r)


def sec_clock(con, days):
    """ساعت ثبت ما را با زمان آخرین به‌روزرسانی خود بورس (h_even) مقایسه می‌کند."""
    if "h_even" not in bc.table_columns(con, "price"):
        say("  ستون h_even هنوز نیست؛ بعد از اولین روز با collector نسخه‌ی ۲ این بخش پر می‌شود.")
        return
    since = (datetime.now() - timedelta(days=days)).strftime("%Y-%m-%d 00:00:00")
    df = pd.read_sql_query(
        "SELECT ts, symbol, d_even, h_even FROM price "
        "WHERE ts >= ? AND h_even IS NOT NULL AND d_even IS NOT NULL", con,
        params=(since,), parse_dates=["ts"])
    if df.empty:
        say("  هنوز ردیفی با h_even ثبت نشده است.")
        return
    hm = df.ts.dt.strftime("%H:%M")
    day_int = df.ts.dt.strftime("%Y%m%d").astype(int)
    df = df[(df.d_even == day_int) & (hm >= "09:05") & (hm <= "12:25")].copy()
    if df.empty:
        say("  در ساعت معاملات (۹:۰۵ تا ۱۲:۲۵) ردیفی برای مقایسه نیست.")
        return
    h = df.h_even.astype(int)
    stamp = (df.ts.dt.strftime("%Y-%m-%d") + " " + (h // 10000).astype(str).str.zfill(2) + ":"
             + ((h // 100) % 100).astype(str).str.zfill(2) + ":" + (h % 100).astype(str).str.zfill(2))
    lag = (df.ts - pd.to_datetime(stamp, errors="coerce")).dt.total_seconds()
    say("  ساعت ثبت ما منهای زمان آخرین به‌روزرسانی بورس (ثانیه):")
    for name, g in lag.groupby(df.symbol):
        g = g.dropna()
        if len(g):
            say("    %-6s: میانه %7.0f | ۱۰٪ %7.0f | ۹۰٪ %7.0f | تعداد %d" % (
                name, g.median(), g.quantile(0.1), g.quantile(0.9), len(g)))
    say("  تفسیر: برای نماد پرمعامله (اهرم) میانه باید نزدیک صفر تا چند دقیقه باشد. "
        "اگر حدود ۱۸۰۰ یا ۳۶۰۰ بود، ساعت ویندوز نسبت به بورس اختلاف دارد.")


def sec_backups():
    files = bc.list_backups(DB_PATH)
    if not files:
        say("  هیچ نسخه‌ی پشتیبانی وجود ندارد.")
        problem("هیچ نسخه‌ی پشتیبان از دیتابیس وجود ندارد")
        return
    last = files[-1]
    age_h = (datetime.now().timestamp() - os.path.getmtime(last)) / 3600
    say("  %d نسخه‌ی پشتیبان | آخرین: %s (%.1f ساعت پیش، %.1f MB)" % (
        len(files), os.path.basename(last), age_h, os.path.getsize(last) / 1e6))
    if age_h > 72:
        problem("آخرین نسخه‌ی پشتیبان بیش از ۷۲ ساعت پیش ساخته شده است")


def sec_files():
    for f in ("config.py", "bourse_common.py", "collector.py", "dashboard.py", "daily_bars.py",
              "signal_features.py", "train_eval.py", "after_market.py"):
        say("  %-20s %s" % (f, ("%d بایت" % os.path.getsize(f)) if os.path.exists(f) else "پیدا نشد"))
    for f in ("test_run.py", "features.py"):
        if os.path.exists(f):
            say("  هشدار: فایل قدیمی %s هنوز هست (به old_tests ببر)" % f)
            problem("فایل قدیمی %s در پوشه‌ی اصلی مانده است" % f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=7)
    args = ap.parse_args()

    say("گزارش سلامت پروژه | %s | %d روز اخیر" % (datetime.now().strftime("%Y-%m-%d %H:%M"), args.days))
    if not os.path.exists(DB_PATH):
        say("دیتابیس %s پیدا نشد." % DB_PATH)
        return 1
    con = sqlite3.connect(DB_PATH, timeout=30)
    try:
        section("محیط و دیتابیس", sec_environment, con)
        section("جدول‌ها", sec_tables, con)
        section("وضعیت روزانه‌ی هر نماد", sec_daily, con, args.days)
        section("عملکرد collector", sec_collector, con, args.days)
        section("رویدادها", sec_events, con, args.days)
        section("ساعت (مقایسه با بورس)", sec_clock, con, args.days)
        section("نسخه‌های پشتیبان", sec_backups)
        section("فایل‌های پروژه", sec_files)
    finally:
        con.close()

    say("")
    say("=" * 8 + " نتیجه " + "=" * 8)
    if problems:
        say("موارد نیازمند توجه (%d):" % len(problems))
        for i, p in enumerate(problems, 1):
            say("  %d) %s" % (i, p))
    else:
        say("مورد نگران‌کننده‌ای پیدا نشد.")
    try:
        with open(REPORT_FILE, "w", encoding="utf-8-sig") as f:
            f.write("\n".join(lines))
        print("\nگزارش در %s هم ذخیره شد." % REPORT_FILE)
    except Exception as e:
        print("ذخیره‌ی گزارش ممکن نشد:", e)
    return 0


if __name__ == "__main__":
    sys.exit(main())