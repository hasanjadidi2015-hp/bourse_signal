# -*- coding: utf-8 -*-
"""
dashboard.py (نسخه‌ی ۲)
داشبورد زنده: هر نماد یک کارت با قیمت، نمودار، وضعیت صف و (فقط اگر مدل معتبر باشد) پیش‌بینی.

اجرا:
    newenv\\Scripts\\python.exe dashboard.py
بعد فایل dashboard.html را در مرورگر باز کن (هر ۱۰ ثانیه خودش تازه می‌شود).
"""
import contextlib
import html as html_lib
import io
import json
import os
import pickle
import sqlite3
import sys
import time
from datetime import datetime, timedelta

import pandas as pd

import bourse_common as bc

try:
    with contextlib.redirect_stdout(io.StringIO()):
        import signal_features as sf
except Exception:
    sf = None

SYMBOLS = [bc.normalize_fa(k) for k in (bc.cfg("SYMBOLS", {}) or {})]
DB_PATH = bc.cfg("DB_PATH", "signal_data.db")
MARKET_OPEN = str(bc.cfg("MARKET_OPEN", "08:45"))
MARKET_CLOSE = str(bc.cfg("MARKET_CLOSE", "12:30"))
MARKET_WEEKDAYS = list(bc.cfg("MARKET_WEEKDAYS", [5, 6, 0, 1, 2]))

OUT_HTML = "dashboard.html"
STATUS_FILE = "model_status.json"
MODEL_FILE = "model.pkl"
REFRESH_SEC = 10
HEAVY = 0.33
TRADING_START = "09:00"      # قبل از این ساعت پیش‌گشایش است
FRESH_LIMIT_SEC = 90         # بیشتر از این بدون داده = «داده‌ی تازه نیست»
STALL_MINUTES = 10           # این‌قدر حجم ثابت = «بدون معامله / توقف؟»
HIST_BOOK = 12
CHART_W, CHART_H = 520, 170

CSS = """
body { font-family: Tahoma, sans-serif; background: #111; color: #eee; padding: 20px; margin: 0; }
h1 { color: #6cf; margin-top: 0; }
.grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(420px, 1fr)); gap: 16px; }
.card { background: #1b1b1b; border: 1px solid #333; border-radius: 10px; padding: 14px; }
.head { display: flex; justify-content: space-between; align-items: center; }
.name { font-size: 20px; font-weight: bold; }
.price { font-size: 30px; font-weight: bold; margin: 6px 0; }
.chg { font-size: 14px; margin-right: 8px; }
.up { color: #4caf50; } .down { color: #f44336; } .flat { color: #ccc; }
.badge { padding: 4px 10px; border-radius: 6px; font-size: 13px; }
.b-buy { background: #1b3d1f; color: #4caf50; }
.b-sell { background: #4a1c1c; color: #f44336; }
.b-flat { background: #2a2a2a; color: #ccc; }
.b-warn { background: #4a3a10; color: #ffb300; }
.row { display: flex; justify-content: space-between; margin: 6px 0; font-size: 14px; }
.hist { direction: ltr; font-size: 12px; color: #aaa; word-break: break-all; }
.nochart { height: 170px; display: flex; align-items: center; justify-content: center;
           color: #777; background: #181818; border-radius: 6px; }
.pred { margin-top: 12px; padding: 10px; border: 1px dashed #444; border-radius: 8px; font-size: 14px; }
.invalid { color: #ffb300; }
.pbar { background: #2a2a2a; height: 10px; border-radius: 5px; overflow: hidden; margin: 8px 0; }
.pfill { height: 100%; } .pf-buy { background: #4caf50; } .pf-flat { background: #666; }
.banner { padding: 10px 14px; border-radius: 8px; margin-bottom: 16px; font-size: 14px; }
.banner.ok { background: #1b3d1f; color: #a5d6a7; }
.banner.warn { background: #4a3a10; color: #ffd54f; }
.note { color: #999; font-size: 13px; margin-top: 18px; line-height: 1.8; }
.sys { color: #888; font-size: 12px; margin-top: 12px; }
"""


def esc(x):
    return html_lib.escape(str(x), quote=True)


def fmt(x, nd=0):
    try:
        if x is None:
            return "-"
        return "{:,.{nd}f}".format(float(x), nd=nd)
    except Exception:
        return "-"


def market_open(now):
    if now.weekday() not in MARKET_WEEKDAYS:
        return False
    return MARKET_OPEN <= now.strftime("%H:%M") <= MARKET_CLOSE


# ---------------------------------------------------------------- مدل
def load_model():
    try:
        if not os.path.exists(STATUS_FILE):
            return None, None
        with open(STATUS_FILE, encoding="utf-8") as f:
            st = json.load(f)
        mdl = None
        if st.get("valid") and os.path.exists(MODEL_FILE):
            with open(MODEL_FILE, "rb") as f:
                mdl = pickle.load(f)
        return st, mdl
    except Exception:
        return None, None


def model_banner(st, mdl):
    try:
        if st is None:
            return ("<div class='banner warn'>مدل هنوز آموزش ندیده است. "
                    "پیش‌بینی نمایش داده نمی‌شود.</div>")
        if not st.get("valid"):
            return ("<div class='banner warn'><b>مدل هنوز معتبر نیست.</b> %s</div>"
                    % esc(st.get("message", "")))
        if mdl is None:
            return ("<div class='banner warn'>فایل مدل پیدا نشد؛ train_eval.py را دوباره اجرا کن.</div>")
        return ("<div class='banner ok'><b>مدل معتبر است.</b> %s | سیگنال‌های آزمون %s | برد %s%% | "
                "میانگین خالص %s%% | آخرین آموزش %s</div>"
                % (esc(st.get("mode", "")), esc(st.get("trades", "?")), esc(st.get("win_pct", "?")),
                   esc(st.get("net_pct", "?")), esc(st.get("updated", "?"))))
    except Exception:
        return "<div class='banner warn'>وضعیت مدل خوانده نشد.</div>"


# ---------------------------------------------------------------- داده
def read_book(con, name, now):
    """آخرین عکس‌های دفتر سفارش (فقط ۲۰ دقیقه‌ی اخیر) برای محاسبه‌ی I و تاریخچه."""
    since = (now - timedelta(minutes=20)).strftime("%Y-%m-%d %H:%M:%S")
    rows = con.execute(
        "SELECT ts, SUM(bid_vol), SUM(ask_vol) FROM book WHERE symbol = ? AND ts >= ? "
        "GROUP BY ts ORDER BY ts DESC LIMIT ?", (name, since, HIST_BOOK)).fetchall()
    if not rows:
        last = con.execute("SELECT MAX(ts) FROM book WHERE symbol = ?", (name,)).fetchone()
        return {"empty": True, "last_ts": last[0] if last else None}

    ts, bv, av = rows[0]
    bv, av = bv or 0, av or 0
    l1 = con.execute("SELECT bid_price, ask_price FROM book WHERE symbol = ? AND ts = ? AND level = 1",
                     (name, ts)).fetchone()
    tot = bv + av
    hist = []
    for _, b, a in reversed(rows):
        t = (b or 0) + (a or 0)
        hist.append((b - a) / t if t > 0 and b is not None and a is not None else None)
    return {"empty": False, "ts": ts, "bv": bv, "av": av,
            "i5": (bv - av) / tot if tot > 0 else None, "hist": hist,
            "bid": l1[0] if l1 else None, "ask": l1[1] if l1 else None}


def read_prices_today(con, name, today):
    cols = bc.table_columns(con, "price")
    extra = ", instr_state_title" if "instr_state_title" in cols else ", NULL"
    return con.execute(
        "SELECT ts, last_price, yesterday_price, cum_volume%s FROM price "
        "WHERE symbol = ? AND ts LIKE ? ORDER BY ts" % extra, (name, today + "%")).fetchall()


def stalled_minutes(prices, now):
    """چند دقیقه است که حجم تجمعی تغییر نکرده؟"""
    try:
        vols = [(r[0], r[3]) for r in prices if r[3] is not None]
        if len(vols) < 2:
            return 0.0
        last_vol = vols[-1][1]
        since_ts = vols[-1][0]
        for ts, v in reversed(vols):
            if v != last_vol:
                break
            since_ts = ts
        start = datetime.strptime(since_ts, "%Y-%m-%d %H:%M:%S")
        return max(0.0, (now - start).total_seconds() / 60.0)
    except Exception:
        return 0.0


def status_of(book, prices, now):
    is_open = market_open(now)
    hm = now.strftime("%H:%M")
    if book is None or book.get("empty"):
        return ("بازار بسته" if not is_open else "داده ندارد", "b-flat" if not is_open else "b-warn")

    try:
        age = (now - datetime.strptime(book["ts"], "%Y-%m-%d %H:%M:%S")).total_seconds()
    except Exception:
        age = None
    if is_open and age is not None and age > FRESH_LIMIT_SEC:
        return ("داده‌ی تازه نیست (collector؟)", "b-warn")
    if not is_open:
        return ("بازار بسته", "b-flat")

    title = prices[-1][4] if prices and len(prices[-1]) > 4 else None
    if title and bc.normalize_fa(title) != "مجاز":
        return (esc(title), "b-warn")
    if hm < TRADING_START:
        return ("پیش‌گشایش", "b-flat")
    if not prices:
        return ("بدون داده‌ی قیمت امروز", "b-warn")
    if stalled_minutes(prices, now) >= STALL_MINUTES:
        return ("بدون معامله %d+ دقیقه (توقف؟)" % STALL_MINUTES, "b-warn")

    if book["bv"] == 0 or book["av"] == 0:
        return ("صف یک‌طرفه", "b-warn")
    i5 = book["i5"]
    if i5 is None:
        return ("نامشخص", "b-warn")
    if i5 > HEAVY:
        return ("فشار خرید", "b-buy")
    if i5 < -HEAVY:
        return ("فشار فروش", "b-sell")
    return ("متعادل", "b-flat")


# ---------------------------------------------------------------- پیش‌بینی
def predict(con, name, mdl, now):
    if sf is None:
        return None, "ماژول ویژگی‌ها پیدا نشد"
    if now.strftime("%H:%M") < TRADING_START:
        return None, "پیش‌گشایش"
    try:
        start = now.strftime("%Y-%m-%d") + " 00:00:00"
        book = pd.read_sql_query("SELECT * FROM book WHERE symbol = ? AND ts >= ?", con,
                                 params=(name, start), parse_dates=["ts"])
        price = pd.read_sql_query("SELECT * FROM price WHERE symbol = ? AND ts >= ?", con,
                                  params=(name, start), parse_dates=["ts"])
        feat = sf.build_symbol(name, book, price)
    except Exception as e:
        return None, "خطا در ویژگی‌ها: %s" % type(e).__name__
    if feat is None or len(feat) == 0:
        return None, "امروز هنوز معامله‌ای نشده"

    row = feat.iloc[-1]
    age = (pd.Timestamp(now) - pd.Timestamp(row["ts"])).total_seconds()
    if age > 120:
        return None, "داده‌ی تازه نیست"
    try:
        X = pd.DataFrame([row[mdl["features"]].astype(float).values], columns=mdl["features"])
        if X.isna().any().any():
            return None, "بعضی ویژگی‌ها هنوز کامل نیستند"
        prob = float(mdl["model"].predict_proba(X)[0, 1])
        mid, sig = float(row["mid"]), float(mdl["sigma"])
    except Exception as e:
        return None, "خطا در مدل: %s" % type(e).__name__
    return {"prob": prob, "threshold": float(mdl["threshold"]), "lo": mid * (1 - sig),
            "hi": mid * (1 + sig)}, None


def pred_block(pred, reason, st, mdl):
    if not st or not st.get("valid") or mdl is None:
        return ("<div class='pred invalid'><b>مدل هنوز معتبر نیست.</b> "
                "پیش‌بینی نمایش داده نمی‌شود.</div>")
    if pred is None:
        return "<div class='pred'>پیش‌بینی در این لحظه ممکن نیست: %s</div>" % esc(reason)
    pct = round(pred["prob"] * 100)
    on = pred["prob"] >= pred["threshold"]
    return (
        "<div class='pred'><div>احتمال رشد ۱۵ دقیقه‌ی بعد: <b>%d%%</b> — <span class='%s'>%s</span> "
        "(آستانه %.0f%%)</div><div class='pbar'><div class='pfill %s' style='width:%d%%'></div></div>"
        "<div class='hist'>محدوده‌ی تقریبی حرکت ۱۵ دقیقه: %s تا %s</div></div>"
        % (pct, "up" if on else "flat", "سیگنال فعال" if on else "بدون سیگنال",
           pred["threshold"] * 100, "pf-buy" if on else "pf-flat", max(0, min(100, pct)),
           fmt(pred["lo"]), fmt(pred["hi"])))


# ---------------------------------------------------------------- نمودار
def svg_chart(points, ref, band=None):
    if len(points) < 2:
        return "<div class='nochart'>داده‌ی کافی برای نمودار امروز نیست</div>"
    w, h, pad = CHART_W, CHART_H, 10
    vals = list(points) + ([ref] if ref else []) + (list(band) if band else [])
    lo, hi = min(vals), max(vals)
    if hi == lo:
        hi = lo + 1

    def x_of(i):
        return pad + i * (w - 2 * pad) / (len(points) - 1)

    def y_of(v):
        return pad + (hi - v) * (h - 2 * pad - 14) / (hi - lo)

    band_svg = ""
    if band:
        yl, yh = y_of(band[0]), y_of(band[1])
        band_svg = ("<rect x='%d' y='%.1f' width='%d' height='%.1f' fill='#6cf' fill-opacity='0.15'/>"
                    % (pad, min(yl, yh), w - 2 * pad, abs(yl - yh)))
    ref_line = ""
    if ref:
        y = y_of(ref)
        ref_line = ("<line x1='%d' x2='%d' y1='%.1f' y2='%.1f' stroke='#888' stroke-dasharray='4 4'/>"
                    "<text x='%d' y='%.1f' fill='#999' font-size='11' text-anchor='end'>قیمت دیروز %s</text>"
                    % (pad, w - pad, y, y, w - pad, y - 4, fmt(ref)))
    path = " ".join("%s%.1f,%.1f" % ("M" if i == 0 else "L", x_of(i), y_of(v))
                    for i, v in enumerate(points))
    color = "#4caf50" if points[-1] >= points[0] else "#f44336"
    return ("<svg viewBox='0 0 %d %d' width='100%%' height='%d' style='background:#181818;border-radius:6px'>"
            "%s%s<path d='%s' fill='none' stroke='%s' stroke-width='2'/>"
            "<circle cx='%.1f' cy='%.1f' r='3.5' fill='%s'/>"
            "<text x='%d' y='12' fill='#aaa' font-size='11'>%s</text>"
            "<text x='%d' y='%d' fill='#aaa' font-size='11'>%s</text></svg>"
            % (w, h, h, band_svg, ref_line, path, color, x_of(len(points) - 1),
               y_of(points[-1]), color, pad, fmt(hi), pad, h - 4, fmt(lo)))


# ---------------------------------------------------------------- کارت
def card_html(con, name, st, mdl, now):
    today = now.strftime("%Y-%m-%d")
    book = read_book(con, name, now)
    prices = read_prices_today(con, name, today)
    label, badge = status_of(book, prices, now)

    last = prices[-1][1] if prices else None
    yest = prices[-1][2] if prices else None
    chg_html, chg_cls = "-", "flat"
    if last is not None and yest:
        chg = (last - yest) / yest * 100
        chg_cls = "up" if chg > 0 else ("down" if chg < 0 else "flat")
        chg_html = "%+.2f%%" % chg

    pred, reason, band = None, None, None
    if st is not None and st.get("valid") and mdl is not None:
        pred, reason = predict(con, name, mdl, now)
        if pred is not None:
            band = (pred["lo"], pred["hi"])

    series = [r[1] for r in prices if r[1] is not None]
    chart = svg_chart(series, yest, band)

    if book and not book.get("empty"):
        i5 = "-" if book["i5"] is None else "%+.2f" % book["i5"]
        hist = " ".join("-" if h is None else "%+.2f" % h for h in book["hist"])
        last_ts, bid, ask = book["ts"], fmt(book["bid"]), fmt(book["ask"])
    else:
        i5, hist, bid, ask = "-", "-", "-", "-"
        last_ts = (book or {}).get("last_ts") or "-"

    return """
<div class='card'>
  <div class='head'><div class='name'>%s</div><span class='badge %s'>%s</span></div>
  <div class='price'>%s <span class='chg %s'>%s</span></div>
  %s
  %s
  <div class='row'><span>بهترین خرید</span><span>%s</span></div>
  <div class='row'><span>بهترین فروش</span><span>%s</span></div>
  <div class='row'><span>عدم تعادل صف (I)</span><span>%s</span></div>
  <div class='row'><span>آخرین عکس صف</span><span>%s</span></div>
  <div class='hist'>I قدیمی ← جدید: %s</div>
</div>""" % (esc(name), badge, label, fmt(last), chg_cls, chg_html, chart,
             pred_block(pred, reason, st, mdl), bid, ask, i5, esc(last_ts), esc(hist))


def error_card(name, err):
    return ("<div class='card'><div class='head'><div class='name'>%s</div>"
            "<span class='badge b-warn'>خطا در نمایش</span></div>"
            "<div class='hist'>%s</div></div>" % (esc(name), esc(err)))


def system_footer(con, now):
    parts = []
    try:
        size = os.path.getsize(DB_PATH) / 1e6
        parts.append("حجم دیتابیس: %.0f MB" % size)
    except Exception:
        pass
    try:
        if bc.table_exists(con, "collector_cycles"):
            since = (now - timedelta(minutes=10)).strftime("%Y-%m-%d %H:%M:%S")
            row = con.execute(
                "SELECT COUNT(*), SUM(CASE WHEN book_ok=0 OR price_ok=0 THEN 1 ELSE 0 END), "
                "ROUND(AVG(latency_ms)) FROM collector_cycles WHERE ts >= ?", (since,)).fetchone()
            if row and row[0]:
                parts.append("۱۰ دقیقه‌ی اخیر: %d درخواست، %d ناموفق، تأخیر میانگین %s ms"
                             % (row[0], row[1] or 0, int(row[2] or 0)))
    except Exception:
        pass
    skew = bc.clock_skew_seconds()
    if skew is not None and abs(skew) > 120:
        parts.append("هشدار: ساعت سیستم %.0f ثانیه با ساعت تهران اختلاف دارد" % skew)
    return "<div class='sys'>%s</div>" % esc(" | ".join(parts)) if parts else ""


def build_html(con, log):
    now = bc.now_local()
    st, mdl = load_model()
    cards = []
    for name in SYMBOLS:
        try:
            cards.append(card_html(con, name, st, mdl, now))
        except Exception as e:
            log.exception("CARD_ERROR %s" % name)
            cards.append(error_card(name, "%s: %s" % (type(e).__name__, e)))
    return """<!DOCTYPE html>
<html lang="fa" dir="rtl">
<head>
<meta charset="utf-8">
<meta http-equiv="refresh" content="%d">
<title>داشبورد زنده</title>
<style>%s</style>
</head>
<body>
<h1>داشبورد زنده بورس</h1>
<p>آخرین بروزرسانی: %s (هر %d ثانیه)</p>
%s
<div class='grid'>%s</div>
<div class='note'>
نمودار: خط رنگی قیمت آخرین معامله‌ی امروز، خط نقطه‌چین قیمت دیروز و ناحیه‌ی آبی محدوده‌ی تقریبی حرکت ۱۵ دقیقه.<br>
احتمال فقط وقتی نشان داده می‌شود که مدل در آزمون روزهای جدا سود خالص مثبت داشته باشد.<br>
I بین −۱ تا +۱ است؛ بالای %s یعنی سفارش خرید سنگین‌تر و زیر %s یعنی فروش سنگین‌تر.<br>
این صفحه توصیه‌ی خرید یا فروش نیست.
</div>
%s
</body>
</html>""" % (REFRESH_SEC, CSS, now.strftime("%Y-%m-%d %H:%M:%S"), REFRESH_SEC,
              model_banner(st, mdl), "".join(cards), HEAVY, -HEAVY, system_footer(con, now))


def write_atomic(path, text):
    """اول در فایل موقت می‌نویسد، بعد جابه‌جا می‌کند؛ مرورگر هیچ‌وقت نیمه‌نوشته نمی‌خواند."""
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        f.write(text)
    for attempt in range(3):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            time.sleep(0.3)          # ویندوز: مرورگر لحظه‌ای فایل را باز نگه داشته است
    with open(path, "w", encoding="utf-8") as f:      # آخرین راه: نوشتن مستقیم
        f.write(text)
    try:
        os.remove(tmp)
    except Exception:
        pass


def main():
    log = bc.setup_logging("dashboard")
    print("داشبورد شروع شد. برای توقف Ctrl+C را بزن.")
    print("فایل خروجی:", OUT_HTML)
    fails = 0
    try:
        while True:
            con = None
            try:
                con = sqlite3.connect(DB_PATH, timeout=10)
                try:
                    con.execute("PRAGMA busy_timeout=10000")
                except sqlite3.Error:
                    pass
                page = build_html(con, log)
                write_atomic(OUT_HTML, page)
                if fails:
                    log.info("DASHBOARD_RECOVERED بعد از %d خطا" % fails)
                fails = 0
                print(bc.now_local().strftime("%H:%M:%S"), "به‌روز شد")
            except Exception:
                fails += 1
                log.exception("DASHBOARD_ERROR (بار %d)؛ آخرین صفحه‌ی سالم حفظ شد" % fails)
            finally:
                try:
                    if con is not None:
                        con.close()
                except Exception:
                    pass
            time.sleep(REFRESH_SEC if fails == 0 else min(60, REFRESH_SEC * (1 + fails)))
    except KeyboardInterrupt:
        print("\nداشبورد متوقف شد.")


if __name__ == "__main__":
    sys.exit(main())