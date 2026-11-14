# -*- coding: utf-8 -*-
"""
signal_features.py (نسخه‌ی ۳)

ساخت features.csv از دیتابیس. تغییرها نسبت به نسخه‌ی ۲:
  - عکس‌های پیش‌گشایش (قبل از ۹:۰۰) کاملاً حذف می‌شوند، چون دفتر آن‌ها قیمت‌های «تقریبی» دارد
    و برچسب ۱۵ دقیقه‌ی بعدشان از باز شدن بازار عبور می‌کند.
  - لحظه‌هایی که حجم تجمعی ۱۰ دقیقه ثابت مانده (توقف ناظر یا بی‌معامله) از آموزش حذف می‌شوند.
  - vol_pace با زمان واقعی معامله (از ۹:۰۰) محاسبه می‌شود، نه از ۸:۴۵.
  - گزارش «روز کامل» (۹:۱۰ تا ۱۲:۲۰) جدا از «روز فعال» چاپ می‌شود.
  - خطا در یک نماد، بقیه را از کار نمی‌اندازد و فایل خروجی نیمه‌نوشته نمی‌ماند.
"""
import os
import sqlite3
import numpy as np
import pandas as pd
import config

print("شروع ساخت ویژگی‌ها...")

HORIZON_MIN = 15
COST = 0.003
SPREAD_MAX = 5.0         # شکاف خرید و فروش بیشتر از این درصد = داده‌ی نامعتبر
RET_MAX = 0.05           # بازده ۱۵ دقیقه‌ای بیشتر از ۵٪ = داده‌ی نامعتبر
OUT_FILE = "features.csv"
BASE_DAYS = 20           # تعداد روز برای مبنای حجم
MIN_BASE = 5             # حداقل روز برای اینکه مبنا معتبر باشد
DAILY_DIR = "daily"
TRADING_START = "09:00"  # قبل از این ساعت پیش‌گشایش است
STALL_ROWS = 60          # حدود ۱۰ دقیقه (هر عکس ≈ ۱۰ ثانیه)
FULL_FIRST = "09:10"     # روز «کامل» یعنی اولین داده تا این ساعت ...
FULL_LAST = "12:20"      # ... و آخرین داده از این ساعت به بعد باشد

MARKET_OPEN = str(getattr(config, "MARKET_OPEN", "08:45"))
MARKET_CLOSE = str(getattr(config, "MARKET_CLOSE", "12:30"))
WEEKDAYS = list(getattr(config, "MARKET_WEEKDAYS", [5, 6, 0, 1, 2]))

STATS = {}


def to_minutes(hm):
    return int(hm[:2]) * 60 + int(hm[3:5])


def minutes_from_start(index):
    """دقیقه‌ی سپری‌شده از شروع معامله (۹:۰۰) و کل دقیقه‌های معامله."""
    sm, cm = to_minutes(TRADING_START), to_minutes(MARKET_CLOSE)
    tm = np.asarray(index.hour * 60 + index.minute)
    total = cm - sm
    return np.clip(tm - sm, 1, total), total


def load_daily(name):
    path = os.path.join(DAILY_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    try:
        d = pd.read_csv(path, parse_dates=["date"])
        return d.drop_duplicates("date").set_index("date")["volume"].sort_index()
    except Exception as e:
        print(f"  هشدار: فایل {path} خوانده نشد ({type(e).__name__}); مبنای حجم برای این نماد ساخته نمی‌شود.")
        return None


def baseline_map(daily_vol, days):
    out = {}
    if daily_vol is None:
        return out
    for day in days:
        prev = daily_vol[daily_vol.index < day].tail(BASE_DAYS)
        if len(prev) >= MIN_BASE:
            out[day] = float(prev.median())
    return out


def build_symbol(name, book, price):
    b = book[book.symbol == name].drop_duplicates(["ts", "level"])
    p = price[price.symbol == name].drop_duplicates("ts")
    if b.empty:
        return None

    agg = b.groupby("ts")[["bid_vol", "ask_vol"]].sum()
    l1 = b[b.level == 1].set_index("ts")[["bid_price", "ask_price"]]
    s = agg.join(l1, how="left").sort_index()
    s.index.name = "ts"

    if not p.empty:
        pp = p.set_index("ts")[["last_price", "cum_volume"]].sort_index()
        pp["pday"] = pp.index.normalize()
        pp_full = pp.reindex(s.index.union(pp.index)).ffill().reindex(s.index)
        s = s.join(pp_full)
    else:
        s["last_price"] = np.nan
        s["cum_volume"] = np.nan
        s["pday"] = pd.NaT

    s["day"] = s.index.normalize()
    s["cum_volume"] = s.cum_volume.where(s.pday == s.day)

    # پوشش هر روز و روزهای فعال (قبل از هر حذفی)
    hm_all = np.asarray(s.index.strftime("%H:%M"))
    cover = pd.DataFrame({"day": s.day.values, "hm": hm_all}).groupby("day").hm.agg(["min", "max"])
    rng = s.groupby("day")["cum_volume"].agg(lambda x: x.max() - x.min())
    active = {d for d in rng[rng > 0].index if d.weekday() in WEEKDAYS}
    full = {d for d in active
            if d in cover.index and cover.loc[d, "min"] <= FULL_FIRST and cover.loc[d, "max"] >= FULL_LAST}

    # حذف پیش‌گشایش و ساعت‌های خارج از معامله
    in_week = np.isin(np.asarray(s.index.weekday), WEEKDAYS)
    in_active = np.asarray(s.day.isin(list(active)))
    pre_mask = in_week & (hm_all >= MARKET_OPEN) & (hm_all < TRADING_START) & in_active
    trade_mask = in_week & (hm_all >= TRADING_START) & (hm_all <= MARKET_CLOSE)
    st = {"days": s.day.nunique(), "active": len(active), "full": len(full),
          "preopen": int(pre_mask.sum()), "spread": 0, "nomid": 0, "ret": 0, "stall": 0, "base": 0}
    STATS[name] = st
    s = s[trade_mask].copy()
    if s.empty:
        return None

    # مبنای حجم و سرعت معامله (نسبت به زمان واقعی معامله)
    base_map = baseline_map(load_daily(name), s.day.unique())
    st["base"] = len(base_map)
    s["vol_base"] = s.day.map(base_map)
    mins, total = minutes_from_start(s.index)
    s["vol_ratio"] = s.cum_volume / s.vol_base
    s["vol_pace"] = s.vol_ratio / (mins / total)

    tot = s.bid_vol + s.ask_vol
    s["I5"] = (s.bid_vol - s.ask_vol) / tot.replace(0, np.nan)

    # قیمت میانی؛ شکاف غیرعادی یعنی قیمت میانی معتبر نیست
    two = (s.bid_price > 0) & (s.ask_price > 0)
    raw_mid = ((s.bid_price + s.ask_price) / 2).where(two)
    raw_spread = ((s.ask_price - s.bid_price) / raw_mid * 100).where(two)
    bad = (raw_spread > SPREAD_MAX).fillna(False)
    st["spread"] = int(bad.sum())
    mid_ff = raw_mid.groupby(s.day).ffill(limit=3)
    s["mid"] = mid_ff.where(~bad)
    s["spread_pct"] = raw_spread.where(~bad)

    s["traded_per_snap"] = s.groupby("day")["cum_volume"].diff().clip(lower=0)

    # لحظه‌ی «حجم ثابت»: ۱۰ دقیقه هیچ معامله‌ای نشده (فقط از روی گذشته تشخیص داده می‌شود)
    s["stall"] = (s.groupby("day")["cum_volume"].diff(STALL_ROWS) == 0).fillna(False)

    g = s.groupby("day")
    s["I5_ma6"] = g["I5"].transform(lambda x: x.rolling(6, min_periods=3).mean())
    s["I5_chg3"] = g["I5"].diff(3)
    s["ret_1m"] = s.mid / g["mid"].shift(6) - 1
    s["ret_3m"] = s.mid / g["mid"].shift(18) - 1
    s["hour"] = s.index.hour + s.index.minute / 60

    before = len(s)
    s = s[s.mid.notna()].copy()
    st["nomid"] = before - len(s)

    base = s[["mid"]].reset_index().rename(columns={"ts": "t"})
    base["target"] = base["t"] + pd.Timedelta(minutes=HORIZON_MIN)
    fut = base[["t", "mid"]].rename(columns={"t": "t2", "mid": "fut_mid"})
    m = pd.merge_asof(
        base.sort_values("target")[["t", "target", "mid"]],
        fut.sort_values("t2"),
        left_on="target", right_on="t2", direction="forward",
        tolerance=pd.Timedelta("60s"),
    ).set_index("t")
    s["fwd_ret"] = (m["fut_mid"] / m["mid"] - 1).reindex(s.index)

    big = (s.fwd_ret.abs() > RET_MAX).fillna(False)
    st["ret"] = int(big.sum())
    s = s[~big].copy()

    # حذف لحظه‌های توقف (بعد از محاسبه‌ی برچسب‌ها تا بقیه‌ی ردیف‌ها دست نخورند)
    stalled = s.stall & s.day.isin(list(active))
    st["stall"] = int(stalled.sum())
    s = s[~s.stall].copy()

    has = s.fwd_ret.notna()
    s["y_up"] = np.where(has, (s.fwd_ret > COST).astype(float), np.nan)
    s["y_down"] = np.where(has, (s.fwd_ret < -COST).astype(float), np.nan)

    s = s[s.day.isin(list(active))].drop(columns=["stall"]).copy()
    s["symbol"] = name
    return s.reset_index()


def main():
    con = sqlite3.connect(config.DB_PATH, timeout=30)
    try:
        book = pd.read_sql_query("SELECT * FROM book", con, parse_dates=["ts"])
        price = pd.read_sql_query("SELECT * FROM price", con, parse_dates=["ts"])
    finally:
        con.close()
    print("ردیف دفتر سفارش:", len(book), "| ردیف قیمت:", len(price))

    parts = []
    for name in config.SYMBOLS:
        try:
            r = build_symbol(name, book, price)
            if r is not None and len(r):
                parts.append(r)
        except Exception as e:
            print(f"  خطا در ساخت ویژگی‌های {name}: {type(e).__name__}: {e}")

    print("\nروزهای داده، فعال، کامل (از ≤%s تا ≥%s) و روزهای با مبنای حجم:" % (FULL_FIRST, FULL_LAST))
    for name in config.SYMBOLS:
        st = STATS.get(name)
        if st:
            print(f"  {name}: داده={st['days']}  فعال={st['active']}  کامل={st['full']}  مبنای حجم={st['base']}")

    print("\nداده‌ی کنار گذاشته‌شده (برای هر نماد):")
    for name in config.SYMBOLS:
        st = STATS.get(name)
        if st:
            print(f"  {name}: پیش‌گشایش={st['preopen']}  بدون قیمت میانی={st['nomid']}  "
                  f"شکاف>{SPREAD_MAX}%={st['spread']}  بازده>{RET_MAX * 100:.0f}%={st['ret']}  "
                  f"توقف حجم={st['stall']}")

    if not parts:
        print("\nداده‌ی فعالی برای ساخت ویژگی پیدا نشد.")
        return

    out = pd.concat(parts, ignore_index=True)
    tmp = OUT_FILE + ".tmp"
    out.to_csv(tmp, index=False, encoding="utf-8-sig")
    os.replace(tmp, OUT_FILE)

    print("\nفایل ساخته شد:", OUT_FILE)
    print("تعداد ردیف به تفکیک نماد (فقط ساعت معامله‌ی ۹:۰۰ تا ۱۲:۳۰ و روزهای فعال):")
    print(out.groupby("symbol").size().to_string())
    print("تعداد ردیف‌هایی که هدف دارند:", int(out.y_up.notna().sum()))
    print("تعداد روزهای فعال کل:", out.day.nunique())

    lab = out.dropna(subset=["fwd_ret"])
    print("\nتوزیع برچسب‌ها (افق %d دقیقه، آستانه %.1f%%):" % (HORIZON_MIN, COST * 100))
    for name, gdf in lab.groupby("symbol"):
        up = (gdf.fwd_ret > COST).mean()
        dn = (gdf.fwd_ret < -COST).mean()
        print(f"  {name}: n={len(gdf)}  P(رشد)={up:.3f}  P(افت)={dn:.3f}  "
              f"میانگین بازده={gdf.fwd_ret.mean() * 100:.3f}%")


if __name__ == "__main__":
    main()