# -*- coding: utf-8 -*-
import sqlite3
from datetime import date

import config

today = str(date.today()) + "%"
con = sqlite3.connect(config.DB_PATH)

rows = con.execute(
    "SELECT symbol, COUNT(*), MIN(ts), MAX(ts) FROM book WHERE ts LIKE ? GROUP BY symbol",
    (today,),
).fetchall()

print("داده‌ی امروز در دفتر سفارش:")
if not rows:
    print("  هیچ داده‌ای برای امروز ثبت نشده است.")
for r in rows:
    print(" ", r)

con.close()