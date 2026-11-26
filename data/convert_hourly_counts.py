"""Convert the 7-day hourly count table (one TIME/NO column pair per day) into a
continuous hourly time series that Circadian Phyphox Studio loads directly.

Interpretation of the source table (row position, not the printed AM/PM, defines the hour):
  * Row k (k = 1..23) is hour k:00 of that day's date, so the series starts at 01:00.
  * Row 24 ("12:00 AM", or "12:00 PM" on Day 7) follows 11 PM, so it is midnight
    closing the day: 00:00 of the next date.
  * Rows 6-7 are printed "6:00 PM"/"7:00 PM" but sit between 5 AM and 8 AM (the real
    6 PM / 7 PM rows come later), so they are 06:00 / 07:00.
  * Dates are taken from the headers. 31 Jan and 1 Feb were not recorded, so those
    hours are left out; the analysis treats them as a gap rather than zeros.

Usage:  python data/convert_hourly_counts.py
"""

from __future__ import annotations

import re
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

HERE = Path(__file__).resolve().parent
RAW = HERE / "raw_hourly_counts_2023-01.txt"
OUT = HERE / "activity_hourly_2023-01-26_to_02-03.csv"


def main() -> None:
    lines = [ln.rstrip("\n") for ln in RAW.read_text(encoding="utf-8").splitlines()]
    dates = [datetime.strptime(d, "%d/%m/%y") for d in re.findall(r"\((\d{2}/\d{2}/\d{2})\)", lines[0])]
    data_rows = [ln.split("\t") for ln in lines if re.match(r"^\d{1,2}:\d{2} [AP]M\t", ln)]
    assert len(data_rows) == 24, f"expected 24 hourly rows, found {len(data_rows)}"

    records = []
    for d_i, day in enumerate(dates):
        for row_i, cells in enumerate(data_rows, start=1):
            label, count = cells[2 * d_i].strip(), cells[2 * d_i + 1].strip()
            printed_hour = int(label.split(":")[0]) % 12 + (12 if label.endswith("PM") else 0)
            expected_hour = row_i % 24
            if printed_hour != expected_hour:
                print(f"  {day:%d %b} row {row_i:>2}: printed '{label}' -> using {expected_hour:02d}:00")
            records.append({"timestamp": day + timedelta(hours=row_i), "activity": int(count)})

    df = pd.DataFrame(records).sort_values("timestamp")
    assert df["timestamp"].is_unique
    df["timestamp"] = df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} hourly rows ({df['timestamp'].iloc[0]} -> {df['timestamp'].iloc[-1]}) to {OUT}")


if __name__ == "__main__":
    main()
