# -*- coding: utf-8 -*-
import sqlite3
from datetime import date

import config

TODAY = date.today().strftime("%Y-%m-%d") + "%"

con = sqlite3.connect(config.DB_PATH)

print("=== تعداد ردیف‌ها (کل) ===")
for table in ("book", "price"):
    rows = con.execute(
        f"SELECT symbol, COUNT(*) FROM {table} GROUP BY symbol"
    ).fetchall()
    print(table, rows)

print("\n=== تعداد ردیف‌های امروز ===")
for table in ("book", "price"):
    rows = con.execute(
        f"SELECT symbol, COUNT(*) FROM {table} WHERE ts LIKE ? GROUP BY symbol",
        (TODAY,),
    ).fetchall()
    print(table, rows)

print("\n=== آخرین ۵ قیمت امروز برای هر نماد ===")
for name in config.SYMBOLS:
    print(f"\n{name}:")
    rows = con.execute(
        "SELECT ts, last_price, cum_volume FROM price "
        "WHERE symbol = ? AND ts LIKE ? ORDER BY ts DESC LIMIT 5",
        (name, TODAY),
    ).fetchall()
    for r in rows:
        print("  ", r)

print("\n=== تعداد قیمت‌های متفاوت امروز (اگر ۱ باشد یعنی قیمت ثابت مانده) ===")
for name in config.SYMBOLS:
    n = con.execute(
        "SELECT COUNT(DISTINCT last_price) FROM price "
        "WHERE symbol = ? AND ts LIKE ?",
        (name, TODAY),
    ).fetchone()[0]
    print(f"{name}: {n}")

print("\n=== تعداد تغییرهای قیمت امروز (چند بار قیمت عوض شد) ===")
for name in config.SYMBOLS:
    rows = con.execute(
        "SELECT last_price FROM price WHERE symbol = ? AND ts LIKE ? ORDER BY ts",
        (name, TODAY),
    ).fetchall()
    changes = sum(1 for a, b in zip(rows, rows[1:]) if a[0] != b[0])
    print(f"{name}: {changes}")

con.close()