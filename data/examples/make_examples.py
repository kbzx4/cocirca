"""Generate small, realistic example files in common chronobiology export layouts.

The layouts follow the published / documented formats of each device or tool; the values are synthetic
(a 24 h rhythm with noise). Used for import tests in both the Python app and the Rust desktop app.

    python data/examples/make_examples.py
"""

from __future__ import annotations

import math
import random
from datetime import datetime, timedelta
from pathlib import Path

OUT = Path(__file__).resolve().parent
rng = random.Random(7)


def circ(t_h: float, acro: float = 14.0, tau: float = 24.0) -> float:
    """0..1 daily drive peaking at `acro` with a rest phase at night."""
    x = 0.5 + 0.5 * math.cos(2 * math.pi * (t_h - acro) / tau)
    return x ** 1.5


def write(name: str, text: str, encoding: str = "utf-8") -> None:
    (OUT / name).write_text(text, encoding=encoding, newline="")
    print(f"wrote {name}")


def actigraph() -> None:
    start = datetime(2024, 3, 4, 8, 0)
    lines = [
        "------------ Data File Created By ActiGraph GT3X+ ActiLife v6.13.4 Firmware v3.2.1 date format dd/MM/yyyy at 30 Hz  Filter Normal -----------\r\n",
        "Serial Number: MOS2E22180542\r\n",
        f"Start Time {start:%H:%M:%S}\r\n",
        f"Start Date {start.day:02d}/{start.month:02d}/{start.year}\r\n",
        "Epoch Period (hh:mm:ss) 00:01:00\r\n",
        "Download Time 09:12:44\r\n",
        "Download Date 11/03/2024\r\n",
        "Current Memory Address: 0\r\n",
        "Current Battery Voltage: 4.11     Mode = 61\r\n",
        "--------------------------------------------------\r\n",
        "Date,Time,Axis1,Axis2,Axis3,Steps,Lux,Inclinometer Off,Inclinometer Standing,Inclinometer Sitting,Inclinometer Lying,Vector Magnitude\r\n",
    ]
    for i in range(7 * 1440):
        t = start + timedelta(minutes=i)
        th = t.hour + t.minute / 60
        a = max(0, int(rng.gauss(900 * circ(th), 200 * circ(th) + 5)))
        lux = int(max(0, 600 * circ(th, 13) + rng.gauss(0, 40))) if 7 <= th <= 21 else rng.randint(0, 3)
        vm = round(math.sqrt(a * a + (0.7 * a) ** 2 + (0.5 * a) ** 2))
        lines.append(f"{t:%d/%m/%Y},{t:%H:%M:%S},{a},{int(0.7 * a)},{int(0.5 * a)},{a // 40},{lux},0,{int(a > 50)},{int(a <= 50)},0,{vm}\r\n")
    write("actigraph_actilife_60s.csv", "".join(lines))


def actiware() -> None:
    start = datetime(2023, 10, 2, 12, 0)
    out = [
        '"Actiware Export File"\n',
        '"Version:","6.3.0"\n',
        '"Identity:","S017"\n',
        '"Epoch Length:","30 sec"\n',
        "\n",
        '"-------------------- Epoch-by-Epoch Data -------------------"\n',
        '"Line","Date","Time","Off-Wrist Status","Activity","Marker","White Light","Red Light","Green Light","Blue Light","Sleep/Wake","Interval Status"\n',
    ]
    for i in range(6 * 2880):
        t = start + timedelta(seconds=30 * i)
        th = t.hour + t.minute / 60
        off = 1 if (3 <= (i // 2880) <= 3 and 14 <= th < 16) else 0  # took the watch off on day 4
        act = 0 if off else max(0, int(rng.gauss(300 * circ(th, 15), 80 * circ(th, 15) + 2)))
        wl = 0 if off else round(max(0.0, 800 * circ(th, 13) + rng.gauss(0, 30)), 2)
        status = "EXCLUDED" if off else ("ACTIVE" if 7 <= th < 23 else "REST")
        tm = t.strftime("%I:%M:%S %p").lstrip("0")
        out.append(f'"{i + 1}","{t.month}/{t.day}/{t.year}","{tm}","{off}","{act}","0","{wl}","{wl * 0.4:.2f}","{wl * 0.5:.2f}","{wl * 0.3:.2f}","{int(act > 20)}","{status}"\n')
    write("actiware_export.csv", "".join(out))


def acttrust() -> None:
    start = datetime(2024, 1, 15, 9, 0)
    out = [
        "+-------------------------------------------------+\n",
        "|                Condor Instruments               |\n",
        "|                    ActTrust                     |\n",
        "+-------------------------------------------------+\n",
        "DEVICE_ID : AT0012345\n",
        "SUBJECT_NAME : P042\n",
        "+-------------------------------------------------+\n",
        "DATE/TIME;MS;EVENT;TEMPERATURE;EXT TEMPERATURE;ORIENTATION;PIM;PIMn;TAT;TATn;ZCM;ZCMn;LIGHT;AMB LIGHT;RED LIGHT;GREEN LIGHT;BLUE LIGHT;IR LIGHT;UVA LIGHT;UVB LIGHT;STATE\n",
    ]
    for i in range(5 * 1440):
        t = start + timedelta(minutes=i)
        th = t.hour + t.minute / 60
        pim = max(0, int(rng.gauss(4000 * circ(th, 15), 600 * circ(th, 15) + 3)))
        temp = 33.0 + 1.6 * (1 - circ(th, 15)) + rng.gauss(0, 0.15)  # wrist temp peaks at night
        light = max(0.0, 900 * circ(th, 13) + rng.gauss(0, 20)) if 7 <= th <= 21 else 0.0
        f = lambda v: f"{v:.2f}".replace(".", ",")  # noqa: E731
        out.append(f"{t:%d/%m/%Y %H:%M:%S};0;0;{f(temp)};{f(22 + rng.gauss(0, .3))};0;{pim};{pim // 60};{pim // 30};{pim // 90};{pim // 8};{pim // 400};{f(light)};{f(light * .8)};{f(light * .3)};{f(light * .4)};{f(light * .2)};0;0;0;0\n")
    write("acttrust_condor.txt", "".join(out), encoding="cp1252")


def awd() -> None:
    lines = ["P007\n", "05-Feb-2024\n", "08:30\n", "4\n", "41\n", "V3021\n", "F\n"]
    for i in range(7 * 1440):
        th = (8.5 + i / 60) % 24
        lines.append(f"{max(0, int(rng.gauss(400 * circ(th, 15), 90 * circ(th, 15) + 1)))} , M\n")
    write("actiwatch_P007.awd", "".join(lines))


def dam() -> None:
    start = datetime(2024, 6, 3, 8, 0)
    rows = []
    taus = [23.6 + 0.05 * k for k in range(32)]  # flies free-running in DD with slightly different periods
    for i in range(6 * 288):
        t = start + timedelta(minutes=5 * i)
        el = i * 5 / 60
        light = 1 if (8 <= t.hour < 20 and el < 48) else 0  # 2 days LD, then DD
        counts = []
        for k in range(32):
            if k in (5, 17):  # empty tubes
                counts.append(0)
                continue
            drive = circ(el + 8, 17, taus[k]) + 0.6 * circ(el + 8, 7, taus[k])  # morning + evening peaks
            counts.append(max(0, int(rng.gauss(12 * drive, 3 * drive + 0.3))))
        rows.append("\t".join([str(i + 1), f"{t.day} {t:%b} {t:%y}", f"{t:%H:%M:%S}", "1", "5", "0", "0", "0", "0", str(light)] + [str(c) for c in counts]))
    write("Monitor7.txt", "\n".join(rows) + "\n")


def hobo() -> None:
    start = datetime(2024, 7, 1, 0, 0)
    out = ['"Plot Title: 20481533"\n', '"#","Date Time, GMT+02:00","Temp, °C (LGR S/N: 20481533, SEN S/N: 20481533)","Intensity, Lux (LGR S/N: 20481533, SEN S/N: 20481533)"\n']
    for i in range(10 * 144):
        t = start + timedelta(minutes=10 * i)
        th = t.hour + t.minute / 60
        temp = 21 + 4 * circ(th, 15) + rng.gauss(0, 0.2)
        lux = max(0.0, 20000 * circ(th, 13) ** 2 + rng.gauss(0, 200)) if 5 <= th <= 21 else 0.0
        out.append(f'{i + 1},{t:%m/%d/%y %I:%M:%S %p},{temp:.3f},{lux:.1f}\n')
    write("hobo_pendant.csv", "".join(out), encoding="cp1252")


def ibutton() -> None:
    start = datetime(2024, 2, 10, 20, 0)
    out = ["1-Wire/iButton Part Number: DS1922L\n", "1-Wire/iButton Registration Number: 4D000000253C3E41\n", "Date/Time,Unit,Value\n"]
    for i in range(6 * 144):
        t = start + timedelta(minutes=10 * i)
        th = t.hour + t.minute / 60
        out.append(f"{t:%d/%m/%y %I:%M:%S %p},C,{33.2 + 1.8 * (1 - circ(th, 16)) + rng.gauss(0, 0.12):.3f}\n")
    write("ibutton_wrist_temp.csv", "".join(out))


def plate_reader() -> None:
    wells = [f"{r}{c}" for r in "AB" for c in range(1, 7)]
    out = ["Time (h)," + ",".join(wells) + "\n"]
    for i in range(5 * 24 * 2):
        th = i / 2
        vals = []
        for k, _ in enumerate(wells):
            tau = 24.5 + 0.3 * k
            amp = 3000 * math.exp(-th / 70)
            vals.append(f"{5000 + 1500 * math.exp(-th / 30) + amp * math.cos(2 * math.pi * (th - 6 - k) / tau) + rng.gauss(0, 80):.0f}")
        out.append(f"{th}," + ",".join(vals) + "\n")
    write("per2luc_platereader.csv", "".join(out))


def gene_table() -> None:
    zts = list(range(0, 48, 4))
    out = ["ID," + ",".join(f"ZT{z:02d}" for z in zts) + "\n"]
    genes = [("Per1", 12, 3.0), ("Per2", 14, 4.0), ("Cry1", 18, 2.0), ("Bmal1", 0, 3.5), ("Nr1d1", 8, 5.0), ("Dbp", 10, 6.0), ("Gapdh", 0, 0.0), ("Actb", 0, 0.0)]
    for name, acro, amp in genes:
        vals = [f"{8 + amp * math.cos(2 * math.pi * (z - acro) / 24) + rng.gauss(0, 0.3):.3f}" for z in zts]
        out.append(name + "," + ",".join(vals) + "\n")
    write("liver_timecourse_jtk.csv", "".join(out))


def fitbit() -> None:
    start = datetime(2024, 4, 1)
    recs = []
    for i in range(3 * 1440):
        t = start + timedelta(minutes=i)
        th = t.hour + t.minute / 60
        recs.append(f'{{"dateTime":"{t:%m/%d/%y %H:%M:%S}","value":{{"bpm":{int(58 + 30 * circ(th, 15) + rng.gauss(0, 3))},"confidence":2}}}}')
    write("heart_rate-2024-04-01.json", "[" + ",".join(recs) + "]")


if __name__ == "__main__":
    for f in (actigraph, actiware, acttrust, awd, dam, hobo, ibutton, plate_reader, gene_table, fitbit):
        f()
