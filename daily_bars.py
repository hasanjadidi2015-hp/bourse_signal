# -*- coding: utf-8 -*-
import os
import sqlite3
from datetime import datetime

import numpy as np
import pandas as pd
import config

# فایل تاریخچه‌ی هر نماد (باید دقیقاً همین نام را در پوشه‌ی پروژه داشته باشد)
HISTORY_FILES = {
    "اهرم": "CharismaAhromi_ETF.csv",
    "وبملت": "S_Mellat Bank.csv",
    "شستا": "Social Sec Inv.csv",
}
OUT_DIR = "daily"
COLS = ["date", "open", "high", "low", "close", "volume", "source"]
COMPLETE_AFTER = "12:20"   # روزی که داده‌اش قبل از این ساعت تمام شده، کامل حساب نمی‌شود


def read_any(path):
    last_err = None
    for enc in ("utf-8-sig", "utf-16", "cp1256"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception as e:
            last_err = e
    raise last_err


def pick(df, options):
    cols = {str(c).strip().strip("<>").upper(): c for c in df.columns}
    for o in options:
        if o in cols:
            return cols[o]
    raise KeyError(f"ستون پیدا نشد: {options}. ستون‌های فایل: {list(df.columns)}")


def load_history(path):
    df = read_any(path)
    c_date = pick(df, ["DTYYYYMMDD", "DATE", "TRADE_DATE", "DATETIME"])
    c_open = pick(df, ["OPEN"])
    c_high = pick(df, ["HIGH"])
    c_low = pick(df, ["LOW"])
    c_close = pick(df, ["CLOSE"])
    c_vol = pick(df, ["VOL", "VOLUME"])

    raw = df[c_date].astype(str).str.strip()
    dates = pd.to_datetime(raw, format="%Y%m%d", errors="coerce")
    if dates.isna().all():
        dates = pd.to_datetime(raw, errors="coerce")

    out = pd.DataFrame({
        "date": dates.dt.normalize(),
        "open": pd.to_numeric(df[c_open], errors="coerce"),
        "high": pd.to_numeric(df[c_high], errors="coerce"),
        "low": pd.to_numeric(df[c_low], errors="coerce"),
        "close": pd.to_numeric(df[c_close], errors="coerce"),
        "volume": pd.to_numeric(df[c_vol], errors="coerce"),
        "source": "history",
    })
    out = out.dropna(subset=["date", "close"])
    out = out[out.volume > 0]
    return out.drop_duplicates("date").sort_values("date").reset_index(drop=True)


def fix_bar(o, h, l, c):
    """قیمت‌های صفر یا ناسازگار را اصلاح می‌کند. اگر داده کافی نباشد None برمی‌گرداند."""
    def ok(v):
        return v is not None and not np.isnan(v) and v > 0

    vals = [v for v in (o, h, l, c) if ok(v)]
    if len(vals) < 2:
        return None
    o = o if ok(o) else c
    c = c if ok(c) else o
    h = h if ok(h) else max(o, c)
    l = l if ok(l) else min(o, c)
    h = max(h, o, c, l)
    l = min(l, o, c, h)
    return o, h, l, c


def live_daily(con, name, now):
    df = pd.read_sql_query(
        "SELECT ts, last_price, closing_price, day_high, day_low, cum_volume "
        "FROM price WHERE symbol = ? AND last_price > 0 ORDER BY ts",
        con, params=(name,), parse_dates=["ts"],
    )
    if df.empty:
        return pd.DataFrame(columns=COLS)

    df["date"] = df.ts.dt.normalize()
    rows = []
    for d, g in df.groupby("date"):
        last_time = g.ts.max().strftime("%H:%M")
        if last_time < COMPLETE_AFTER:
            print(f"  {name}: روز {d:%Y-%m-%d} ناقص است (آخرین داده ساعت {last_time}) و کنار گذاشته شد.")
            continue

        vol = g.cum_volume.max()
        if pd.isna(vol) or vol <= 0:
            continue
        closes = g.closing_price.replace(0, np.nan)
        fixed = fix_bar(
            g.last_price.iloc[0],
            g.day_high.replace(0, np.nan).max(),
            g.day_low.replace(0, np.nan).min(),
            closes.iloc[-1] if closes.notna().any() else g.last_price.iloc[-1],
        )
        if fixed is None:
            continue
        o, h, l, c = fixed
        rows.append({"date": d, "open": o, "high": h, "low": l,
                     "close": c, "volume": vol, "source": "live"})

    out = pd.DataFrame(rows, columns=COLS)

    # روز امروز را فقط بعد از پایان بازار ثبت کن
    if now.strftime("%H:%M") < config.MARKET_CLOSE:
        out = out[out.date < pd.Timestamp(now.date())]
    return out


def build(name, con, now):
    live = live_daily(con, name, now)
    hist_path = HISTORY_FILES.get(name)

    if hist_path and os.path.exists(hist_path):
        hist = load_history(hist_path)
        extra = live[~live.date.isin(hist.date)]
        full = pd.concat([hist, extra], ignore_index=True)

        ov = live[["date", "volume", "close"]].merge(
            hist[["date", "volume", "close"]], on="date", suffixes=("_live", "_hist"))
        same_vol = int((ov.volume_live == ov.volume_hist).sum())
        same_close = int((ov.close_live - ov.close_hist).abs().lt(1).sum())
        print(f"  {name}: همپوشانی با تاریخچه={len(ov)} روز | "
              f"حجم یکسان={same_vol} | بسته‌شدن یکسان={same_close}")
    else:
        if hist_path:
            print(f"  {name}: فایل تاریخچه‌ی '{hist_path}' پیدا نشد؛ فقط داده‌ی زنده استفاده می‌شود.")
        hist = pd.DataFrame(columns=COLS)
        extra = live
        full = live

    full = full.sort_values("date").reset_index(drop=True)
    os.makedirs(OUT_DIR, exist_ok=True)
    out_path = os.path.join(OUT_DIR, f"{name}.csv")
    full[COLS].to_csv(out_path, index=False, encoding="utf-8-sig", date_format="%Y-%m-%d")

    print(f"{name}: تاریخچه={len(hist)} روز | زنده‌ی جدید={len(extra)} روز | "
          f"کل={len(full)} روز")
    if len(full):
        last = full.iloc[-1]
        print(f"  آخرین روز: {last['date']:%Y-%m-%d}  باز={last['open']}  "
              f"بالا={last['high']}  پایین={last['low']}  بسته={last['close']}  "
              f"حجم={last['volume']:,.0f}  منبع={last['source']}")
    print(f"  فایل: {out_path}")


def main():
    now = datetime.now()
    print("ساخت کندل‌های روزانه...")
    con = sqlite3.connect(config.DB_PATH)
    try:
        for name in config.SYMBOLS:
            try:
                build(name, con, now)
            except Exception as e:
                print(f"{name}: خطا: {e}")
    finally:
        con.close()
    print("\nتمام شد.")


if __name__ == "__main__":
    main()