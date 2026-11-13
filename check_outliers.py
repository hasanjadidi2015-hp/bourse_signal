# -*- coding: utf-8 -*-
import pandas as pd

df = pd.read_csv("features.csv", parse_dates=["ts"])
df = df.dropna(subset=["fwd_ret"])
df["r"] = df.fwd_ret * 100

print("توزیع بازده ۱۵ دقیقه (درصد):")
print(df.r.describe(percentiles=[0.01, 0.05, 0.5, 0.95, 0.99]).round(3).to_string())

print("\n۱۰ بازده‌ی بزرگ مثبت:")
cols = ["ts", "symbol", "mid", "r", "spread_pct", "ret_1m"]
print(df.nlargest(10, "r")[cols].to_string())

print("\n۱۰ بازده‌ی بزرگ منفی:")
print(df.nsmallest(10, "r")[cols].to_string())

print(f"\nتعداد بازده‌های بیشتر از ۵٪ (قدر مطلق): {int((df.r.abs() > 5).sum())}")

trim = df[(df.r > df.r.quantile(0.01)) & (df.r < df.r.quantile(0.99))]
print(f"میانگین کل بازده: {df.r.mean():.3f}%")
print(f"میانگین بدون ۱٪ بالا و پایین: {trim.r.mean():.3f}%")