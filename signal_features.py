# -*- coding: utf-8 -*-
import os
import sqlite3
import numpy as np
import pandas as pd
import config

print("شروع ساخت ویژگی‌ها...")

HORIZON_MIN = 15
COST = 0.003
SPREAD_MAX = 5.0        # شکاف خرید و فروش بیشتر از این درصد = داده‌ی نامعتبر
RET_MAX = 0.05          # بازده ۱۵ دقیقه‌ای بیشتر از ۵٪ = داده‌ی نامعتبر
OUT_FILE = "features.csv"
BASE_DAYS = 20          # تعداد روز برای مبنای حجم
MIN_BASE = 5            # حداقل روز برای اینکه مبنا معتبر باشد
DAILY_DIR = "daily"
DAY_STATS = {}
BASE_STATS = {}
CLEAN_STATS = {}


def market_mask(index):
    wd = np.asarray(index.weekday)
    hm = np.asarray(index.strftime("%H:%M"))
    return (
        np.isin(wd, config.MARKET_WEEKDAYS)
        & (hm >= config.MARKET_OPEN)
        & (hm <= config.MARKET_CLOSE)
    )


def minutes_from_open(index):
    om = int(config.MARKET_OPEN[:2]) * 60 + int(config.MARKET_OPEN[3:])
    cm = int(config.MARKET_CLOSE[:2]) * 60 + int(config.MARKET_CLOSE[3:])
    tm = np.asarray(index.hour * 60 + index.minute)
    total = cm - om
    return np.clip(tm - om, 1, total), total


def load_daily(name):
    path = os.path.join(DAILY_DIR, f"{name}.csv")
    if not os.path.exists(path):
        return None
    d = pd.read_csv(path, parse_dates=["date"])
    return d.drop_duplicates("date").set_index("date")["volume"].sort_index()


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

    rng = s.groupby("day")["cum_volume"].agg(lambda x: x.max() - x.min())
    active = set(rng[rng > 0].index)
    DAY_STATS[name] = (s.day.nunique(), len(active))

    base_map = baseline_map(load_daily(name), s.day.unique())
    BASE_STATS[name] = len(base_map)
    s["vol_base"] = s.day.map(base_map)
    mins, total = minutes_from_open(s.index)
    s["vol_ratio"] = s.cum_volume / s.vol_base
    s["vol_pace"] = s.vol_ratio / (mins / total)

    tot = s.bid_vol + s.ask_vol
    s["I5"] = (s.bid_vol - s.ask_vol) / tot.replace(0, np.nan)

    # ۱. پاک‌سازی شکاف غیرعادی: اگر خرید و فروش خیلی از هم دور باشند، میانگین قیمت معتبر نیست
    two = (s.bid_price > 0) & (s.ask_price > 0)
    raw_mid = ((s.bid_price + s.ask_price) / 2).where(two)
    raw_spread = ((s.ask_price - s.bid_price) / raw_mid * 100).where(two)
    bad = (raw_spread > SPREAD_MAX).fillna(False)
    spread_dropped = int(bad.sum())

    mid_ff = raw_mid.groupby(s.day).ffill(limit=3)
    s["mid"] = mid_ff.where(~bad)
    s["spread_pct"] = raw_spread.where(~bad)

    s["traded_per_snap"] = s.groupby("day")["cum_volume"].diff().clip(lower=0)

    g = s.groupby("day")
    s["I5_ma6"] = g["I5"].transform(lambda x: x.rolling(6, min_periods=3).mean())
    s["I5_chg3"] = g["I5"].diff(3)
    s["ret_1m"] = s.mid / g["mid"].shift(6) - 1
    s["ret_3m"] = s.mid / g["mid"].shift(18) - 1
    s["hour"] = s.index.hour + s.index.minute / 60

    s = s[s.mid.notna()].copy()

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

    # ۲. پاک‌سازی بازده غیرعادی: حرکت بیشتر از ۵٪ در ۱۵ دقیقه برای این نمادها معقول نیست
    big = (s.fwd_ret.abs() > RET_MAX).fillna(False)
    ret_dropped = int(big.sum())
    s = s[~big].copy()

    CLEAN_STATS[name] = (spread_dropped, ret_dropped)

    has = s.fwd_ret.notna()
    s["y_up"] = np.where(has, (s.fwd_ret > COST).astype(float), np.nan)
    s["y_down"] = np.where(has, (s.fwd_ret < -COST).astype(float), np.nan)

    s = s[market_mask(s.index) & s.day.isin(list(active))].copy()
    s["symbol"] = name
    return s.reset_index()


def main():
    con = sqlite3.connect(config.DB_PATH)
    book = pd.read_sql_query("SELECT * FROM book", con, parse_dates=["ts"])
    price = pd.read_sql_query("SELECT * FROM price", con, parse_dates=["ts"])
    con.close()
    print("ردیف دفتر سفارش:", len(book), "| ردیف قیمت:", len(price))

    parts = []
    for name in config.SYMBOLS:
        r = build_symbol(name, book, price)
        if r is not None and len(r):
            parts.append(r)

    print("\nروزهای داده، روزهای فعال و روزهای مبنای حجم به تفکیک نماد:")
    for name in config.SYMBOLS:
        total, active = DAY_STATS.get(name, (0, 0))
        base = BASE_STATS.get(name, 0)
        print(f"  {name}: روزهای داده={total}  روزهای فعال={active}  "
              f"روزهای با مبنای حجم={base}")

    print("\nداده‌ی حذف‌شده به دلیل غیرعادی بودن (برای هر نماد):")
    for name in config.SYMBOLS:
        sp, rt = CLEAN_STATS.get(name, (0, 0))
        print(f"  {name}: شکاف خرید-فروش بیش از {SPREAD_MAX}%={sp}  |  "
              f"بازده ۱۵ دقیقه بیش از {RET_MAX * 100:.0f}%={rt}")

    if not parts:
        print("\nداده‌ی فعالی برای ساخت ویژگی پیدا نشد.")
        return

    out = pd.concat(parts, ignore_index=True)
    out.to_csv(OUT_FILE, index=False, encoding="utf-8-sig")

    print("\nفایل ساخته شد:", OUT_FILE)
    print("تعداد ردیف به تفکیک نماد (فقط ساعت معاملات و روزهای فعال):")
    print(out.groupby("symbol").size().to_string())
    print("تعداد ردیف‌هایی که هدف دارند:", int(out.y_up.notna().sum()))
    print("تعداد روزهای فعال کل:", out.day.nunique())

    lab = out.dropna(subset=["fwd_ret"])
    print("\nتوزیع برچسب‌ها (افق %d دقیقه، آستانه %.1f%%):" % (HORIZON_MIN, COST * 100))
    for name, g in lab.groupby("symbol"):
        up = (g.fwd_ret > COST).mean()
        dn = (g.fwd_ret < -COST).mean()
        print(f"  {name}: n={len(g)}  P(رشد)={up:.3f}  P(افت)={dn:.3f}  "
              f"میانگین بازده={g.fwd_ret.mean() * 100:.3f}%")


if __name__ == "__main__":
    main()