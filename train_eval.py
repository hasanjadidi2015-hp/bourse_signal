# -*- coding: utf-8 -*-
import json
import pickle
from datetime import datetime

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("شروع ارزیابی مدل (نسخه ۲)...")

FEATURES = ["I5", "I5_ma6", "I5_chg3", "spread_pct", "ret_1m", "ret_3m", "hour", "vol_pace"]
ROUNDTRIP = 0.003              # کارمزد و لغزش تقریبی رفت و برگشت
MIN_TRAIN_DAYS = 5             # حداقل روز برای آموزش اولین مدل
MIN_TRADES = 30                # حداقل تعداد سیگنال مستقل
MIN_SIGNAL_DAYS = 3            # حداقل روزی که سیگنال داشته باشد
MIN_POSITIVE_SHARE = 0.5       # حداقل سهم روزهایی که میانگین خالص مثبت دارند
GAP_SEC = 900                  # فاصله‌ی حداقل بین دو سیگنال یک نماد (۱۵ دقیقه)
REPORT_FILE = "model_report.txt"
STATUS_FILE = "model_status.json"
MODEL_FILE = "model.pkl"

ABS_THRESHOLDS = [0.50, 0.55, 0.60]     # آستانه‌ی ثابت احتمال
PCT_CUTOFFS = [0.02, 0.05, 0.10]        # درصد بالاترین احتمال‌ها (نسبت به آموزش)

lines = []


def say(msg=""):
    print(msg)
    lines.append(msg)


def save_status(valid, **info):
    data = {"valid": valid, "updated": datetime.now().strftime("%Y-%m-%d %H:%M"), **info}
    with open(STATUS_FILE, "w", encoding="utf-8") as f:
        json.dump(data, f, ensure_ascii=False, indent=1)


def write_report():
    with open(REPORT_FILE, "w", encoding="utf-8-sig") as f:
        f.write("\n".join(lines))
    print(f"\nگزارش در {REPORT_FILE} و وضعیت در {STATUS_FILE} ذخیره شد.")


def spaced(sel):
    """از هر نماد فقط سیگنال‌هایی را نگه می‌دارد که حداقل ۱۵ دقیقه از هم فاصله دارند."""
    keep = []
    for _, g in sel.groupby("symbol"):
        chosen = []
        for _, row in g.sort_values("p", ascending=False).iterrows():
            t = row["ts"]
            if all(abs((t - c).total_seconds()) >= GAP_SEC for c in chosen):
                chosen.append(t)
                keep.append(row)
    if not keep:
        return sel.iloc[0:0]
    return pd.DataFrame(keep)


df = pd.read_csv("features.csv", parse_dates=["ts"])
df = df.dropna(subset=FEATURES + ["y_up", "fwd_ret"])
df["day"] = df.ts.dt.normalize()
days = sorted(df.day.unique())

say(f"تعداد روزهای معاملاتی قابل استفاده: {len(days)}  |  ردیف قابل استفاده: {len(df)}")
for name, g in df.groupby("symbol"):
    say(f"  {name}: ردیف={len(g)}  روز={g.day.nunique()}")

if len(days) < MIN_TRAIN_DAYS + 1:
    say(f"داده کافی نیست. حداقل {MIN_TRAIN_DAYS + 1} روز معاملاتی لازم است.")
    save_status(False, message=f"داده کافی نیست (حداقل {MIN_TRAIN_DAYS + 1} روز لازم است)",
                days=len(days))
    write_report()
    raise SystemExit

folds = []
for i, d in enumerate(days):
    if i < MIN_TRAIN_DAYS:
        continue
    tr = df[df.day < d]
    te = df[df.day == d]
    if te.empty or tr.y_up.nunique() < 2:
        continue
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(tr[FEATURES], tr.y_up)
    tr_p = model.predict_proba(tr[FEATURES])[:, 1]
    te = te.assign(p=model.predict_proba(te[FEATURES])[:, 1])
    folds.append((te, tr_p))

if not folds:
    say("هیچ روز آزمونی ساخته نشد.")
    save_status(False, message="مدل هنوز معتبر نیست: روز آزمونی ساخته نشد", days=len(days))
    write_report()
    raise SystemExit

tested = len(folds)
say(f"روزهای آزمون (خارج از نمونه): {tested}")
say(f"مبنا (هیچ معامله‌ای نکن): بازده خالص = 0")
say(f"مبنا (خرید همیشگی): میانگین بازده خالص = "
    f"{(df.fwd_ret.mean() - ROUNDTRIP) * 100:.3f}%")
say("")


def run_candidate(kind, value):
    """kind='abs' یعنی آستانه‌ی ثابت؛ kind='pct' یعنی درصد بالاترین احتمال‌ها."""
    parts = []
    for te, tr_p in folds:
        thr = value if kind == "abs" else float(np.quantile(tr_p, 1 - value))
        sel = te[te.p >= thr]
        sig = spaced(sel)
        if len(sig):
            parts.append(sig.assign(net=sig.fwd_ret - ROUNDTRIP))
    if not parts:
        return None
    sig = pd.concat(parts, ignore_index=True)
    day_means = sig.groupby("day").net.mean()
    return {
        "n": len(sig),
        "win": float((sig.net > 0).mean()),
        "mean": float(sig.net.mean()),
        "signal_days": int(len(day_means)),
        "positive_share": float((day_means > 0).mean()),
        "day_table": day_means,
        "sig": sig,
    }


candidates = [("ثابت", f"{v:.2f}", "abs", v) for v in ABS_THRESHOLDS]
candidates += [("درصد", f"بالاترین {int(v * 100)}٪", "pct", v) for v in PCT_CUTOFFS]

say("نتایج هر تنظیم (سیگنال‌ها با فاصله‌ی ۱۵ دقیقه، فقط خرید):")
results = []
for kind_fa, label, kind, value in candidates:
    r = run_candidate(kind, value)
    if r is None:
        say(f"  {kind_fa} {label}: سیگنالی نبود")
        continue
    results.append((kind_fa, label, kind, value, r))
    say(f"  {kind_fa} {label}: سیگنال={r['n']}  برد={r['win'] * 100:.1f}%  "
        f"میانگین خالص={r['mean'] * 100:.3f}%  روزهای سیگنال={r['signal_days']}  "
        f"روزهای مثبت={r['positive_share'] * 100:.0f}%")

valid_results = [x for x in results
                 if x[4]["n"] >= MIN_TRADES
                 and x[4]["mean"] > 0
                 and x[4]["signal_days"] >= MIN_SIGNAL_DAYS
                 and x[4]["positive_share"] >= MIN_POSITIVE_SHARE]

say("")
if not valid_results:
    best_any = max(results, key=lambda x: x[4]["mean"]) if results else None
    reason = "هیچ تنظیمی هر سه شرط را همزمان نداشت (سیگنال کافی، میانگین خالص مثبت، و روزهای مثبت کافی)"
    say("نتیجه: " + reason)
    if best_any:
        say(f"بهترین تنظیم از نظر میانگین: {best_any[0]} {best_any[1]} با میانگین خالص "
            f"{best_any[4]['mean'] * 100:.3f}% و {best_any[4]['n']} سیگنال (معتبر نیست).")
    save_status(False, message="مدل هنوز معتبر نیست: " + reason,
                days=len(days), tested_days=int(tested))
else:
    kind_fa, label, kind, value, r = max(valid_results, key=lambda x: x[4]["mean"])
    say(f"بهترین تنظیم معتبر: {kind_fa} {label} با {r['n']} سیگنال، "
        f"میانگین خالص {r['mean'] * 100:.3f}%، برد {r['win'] * 100:.1f}%")

    say("\nنتیجه‌ی روز به روز برای این تنظیم:")
    for d, v in r["day_table"].items():
        say(f"  {pd.Timestamp(d):%Y-%m-%d}: میانگین خالص={v * 100:.3f}%")

    final = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    final.fit(df[FEATURES], df.y_up)
    final_tr_p = final.predict_proba(df[FEATURES])[:, 1]
    thr_final = value if kind == "abs" else float(np.quantile(final_tr_p, 1 - value))
    sigma = float(df.fwd_ret.std())
    with open(MODEL_FILE, "wb") as f:
        pickle.dump({"model": final, "threshold": thr_final, "sigma": sigma,
                     "features": FEATURES}, f)

    save_status(
        True,
        message="در آزمون روزهای جدا، این تنظیم بعد از کارمزد سود خالص مثبت داشت",
        threshold=thr_final, mode=f"{kind_fa} {label}", trades=int(r["n"]),
        win_pct=round(r["win"] * 100, 1), net_pct=round(r["mean"] * 100, 3),
        positive_days_pct=round(r["positive_share"] * 100, 1),
        sigma_pct=round(sigma * 100, 3), days=len(days), tested_days=int(tested),
    )
    say("\nنتیجه: مدل معتبر است، ولی هنوز اثبات قطعی نیست. با داده‌ی بیشتر دوباره بررسی کن.")

write_report()