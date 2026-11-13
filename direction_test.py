# -*- coding: utf-8 -*-
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

print("آزمون جهت ۱۵ دقیقه‌ی بعد (بدون کارمزد)...")

FEATURES = ["I5", "I5_ma6", "I5_chg3", "spread_pct", "ret_1m", "ret_3m", "hour", "vol_pace"]
MIN_TEST_ROWS = 300   # کمتر از این تعداد، نتیجه قابل اعتماد نیست

df = pd.read_csv("features.csv", parse_dates=["ts"])
df = df.dropna(subset=FEATURES + ["fwd_ret"])
df["day"] = df.ts.dt.normalize()
df["y"] = (df.fwd_ret > 0).astype(int)     # ۱ = بالا، ۰ = پایین
days = sorted(df.day.unique())

print(f"ردیف قابل استفاده: {len(df)}  |  روز: {len(days)}")
for name, g in df.groupby("symbol"):
    print(f"  {name}: ردیف={len(g)}  روز={g.day.nunique()}")

if len(days) < 2:
    print("حداقل ۲ روز معاملاتی لازم است.")
    raise SystemExit

parts = []
for i in range(1, len(days)):
    tr = df[df.day < days[i]]           # فقط گذشته
    te = df[df.day == days[i]]          # روز بعد
    if te.empty or tr.y.nunique() < 2:
        continue
    model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
    model.fit(tr[FEATURES], tr.y)
    parts.append(te.assign(pred=model.predict(te[FEATURES])))

if not parts:
    print("آزمونی ساخته نشد.")
    raise SystemExit

res = pd.concat(parts)


def report(label, g):
    acc = (g.pred == g.y).mean()
    up_share = g.y.mean()
    naive = max(up_share, 1 - up_share)   # اگر همیشه جهت غالب را حدس بزنیم
    flag = "" if len(g) >= MIN_TEST_ROWS else "  (نمونه کم، قابل اعتماد نیست)"
    print(f"{label}: n={len(g)}  دقت مدل={acc * 100:.1f}%  "
          f"حدس ساده={naive * 100:.1f}%  تفاوت={(acc - naive) * 100:+.1f}%{flag}")
    return acc, naive


print("\nنتیجه به تفکیک نماد:")
for name, g in res.groupby("symbol"):
    report(f"  {name}", g)

print("\nنتیجه کل:")
acc, naive = report("  همه", res)

print("\nنتیجه:")
if len(res) < MIN_TEST_ROWS:
    print("  داده هنوز کم است. این عدد فقط یک نشانه است، نه نتیجه‌ی قابل اتکا.")
elif acc > naive + 0.02:
    print("  مدل از حدس ساده بهتر است، ولی باید با داده‌ی بیشتر تکرار شود.")
else:
    print("  مدل از حدس ساده بهتر نیست. با این ویژگی‌ها هنوز سیگنال جهتی دیده نمی‌شود.")