"""Import tests: every example layout in data/examples plus inline fixtures for date / time edge cases."""

from __future__ import annotations

import gzip
import io
import sys
import unittest
import zipfile
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from circadian_app.formats import ImportOptions, Source, combine, detect_order, expand, load_any, parse_source, subject_to_dataset  # noqa: E402
from circadian_app.pipeline import CircadianAnalysisConfig, run_full_analysis  # noqa: E402

EX = ROOT / "data" / "examples"


def parse(name: str, text: str, **opts):
    return parse_source(Source(name, text.encode("utf-8")), ImportOptions(**opts))


class TestExampleFiles(unittest.TestCase):
    EXPECT = {
        "actigraph_actilife_60s.csv": ("ActiGraph ActiLife CSV", 1, {"activity", "light"}),
        "actiware_export.csv": ("Philips Respironics Actiware", 1, {"activity", "light"}),
        "acttrust_condor.txt": ("Condor ActTrust", 1, {"activity", "light", "temperature"}),
        "actiwatch_P007.awd": ("Actiwatch / CamNtech AWD", 1, {"activity"}),
        "Monitor7.txt": ("TriKinetics DAM monitor", 30, {"activity", "light"}),
        "hobo_pendant.csv": ("HOBO logger (HOBOware CSV)", 1, {"temperature", "light"}),
        "ibutton_wrist_temp.csv": ("iButton / Thermochron (OneWireViewer)", 1, {"temperature"}),
        "per2luc_platereader.csv": ("Multi-sample table (one subject per column)", 12, {"signal"}),
        "liver_timecourse_jtk.csv": ("Time-course table (ZT / CT columns)", 8, {"signal"}),
        "heart_rate-2024-04-01.json": ("Fitbit JSON", 1, {"heart_rate"}),
    }

    @classmethod
    def setUpClass(cls) -> None:
        if not (EX / "Monitor7.txt").exists():
            import runpy

            runpy.run_path(str(EX / "make_examples.py"), run_name="__main__")

    def test_every_example_is_recognised_and_analysable(self) -> None:
        cfg = CircadianAnalysisConfig(epoch_minutes=30)
        for fname, (fmt, n_subj, roles) in self.EXPECT.items():
            with self.subTest(fname):
                subjects, _, parsed = load_any(fname, (EX / fname).read_bytes())
                self.assertEqual(parsed[0].format, fmt)
                self.assertEqual(len(subjects), n_subj)
                self.assertTrue(roles <= {c.role for c in subjects[0].channels})
                span_h = (subjects[0].t[-1] - subjects[0].t[0]).total_seconds() / 3600
                self.assertGreater(span_h, 23)
                rep = run_full_analysis(subject_to_dataset(subjects[0]), cfg).primary
                self.assertTrue(np.isfinite(rep.cosinor_single.amplitude))

    def test_temperature_as_main_signal(self) -> None:
        subjects, _, _ = load_any("acttrust_condor.txt", (EX / "acttrust_condor.txt").read_bytes())
        ds = subject_to_dataset(subjects[0], "Temperature")
        rep = run_full_analysis(ds, CircadianAnalysisConfig(epoch_minutes=30)).primary
        self.assertEqual((rep.channel_name, rep.unit_label), ("Temperature", "°C"))
        self.assertTrue(1.5 < rep.cosinor_single.acrophase_hours < 4.5)  # wrist temperature peaks at night


class TestLayouts(unittest.TestCase):
    def test_dates_and_times(self) -> None:
        self.assertEqual(detect_order(["10/02/24 23:50", "11/02/24 00:00", "12/02/24 00:10"])[0], "dmy")
        self.assertEqual(detect_order(["02/10/24 23:50", "02/11/24 00:00", "02/12/24 00:10"])[0], "mdy")
        p = parse("b.csv", "Date;Time;Activity;Light\n26.01.2023;23:00;1,5;10\n26.01.2023;23:30;2,5;20\n27.01.2023;00:00;3,5;30\n")
        s = p.subjects[0]
        self.assertEqual(str(s.t[2]), "2023-01-27 00:00:00")
        self.assertEqual(s.channel("activity").values[0], 1.5)

    def test_actiware_off_wrist_mask(self) -> None:
        t = ('"Actiware Export File"\n"-------------------- Epoch-by-Epoch Data -------------------"\n'
             '"Line","Date","Time","Off-Wrist Status","Activity","White Light","Interval Status"\n'
             '"1","3/20/2019","12:00:00 PM","0","120","300.5","ACTIVE"\n"2","3/20/2019","12:00:30 PM","1","0","0","EXCLUDED"\n"3","3/20/2019","12:01:00 PM","0","95","280","ACTIVE"\n')
        s = parse("aw.csv", t).subjects[0]
        self.assertEqual(str(s.t[0]), "2019-03-20 12:00:00")
        self.assertTrue(np.isnan(s.channel("activity").values[1]))

    def test_numeric_times(self) -> None:
        self.assertEqual(int(np.diff(parse("p.csv", '"Time (s)","Illuminance (lx)"\n0.0,10\n0.5,11\n1.0,12\n').subjects[0].t.asi8)[0]), 500_000_000)
        s = parse("x.csv", "Date serial,Activity\n44927.0,1\n44927.5,2\n44928.0,3\n").subjects[0]
        self.assertEqual(str(s.t[1]), "2023-01-01 12:00:00")
        s = parse("zt.csv", "ZT,Melatonin (pg/mL)\n0,5\n4,3\n8,2\n12,10\n16,60\n20,40\n").subjects[0]
        self.assertEqual(s.channels[0].unit, "pg/mL")
        self.assertEqual((s.t[1] - s.t[0]).total_seconds(), 4 * 3600)
        s = parse("sl.csv", "time,seconds_elapsed,z,y,x\n1700000000000000000,0,9.8,0.1,0.2\n1700000000500000000,0.5,9.7,0.1,0.2\n1700000001000000000,1.0,9.8,0.2,0.1\n").subjects[0]
        self.assertIsNotNone(s.channel("accel_x"))

    def test_rr_polar_and_archives(self) -> None:
        p = parse("000.txt", "800\n" * 150)
        self.assertEqual(p.format, "RR / inter-beat intervals")
        self.assertAlmostEqual(float(p.subjects[0].channel("heart_rate").values[0]), 75.0)
        polar = "Name,Sport,Date,Start time,Duration\nJo,RUNNING,15-01-2020,08:00:00,01:00:00\nSample rate,Time,HR (bpm),Speed (km/h)\n1,00:00:00,80,0\n1,00:00:01,82,1.2\n1,00:00:02,85,2.0\n"
        s = parse("polar.csv", polar).subjects[0]
        self.assertEqual(str(s.t[2]), "2020-01-15 08:00:02")
        buf = io.BytesIO()
        with zipfile.ZipFile(buf, "w") as zf:
            zf.writestr("rr/s1.txt", "800\n" * 150)
            zf.writestr("rr/s2.txt", "700\n" * 150)
            zf.writestr("rr/LICENSE.txt", "CC-BY")
        subjects, notes, _ = load_any("rr.zip", buf.getvalue())
        self.assertEqual(len(subjects), 2)
        gz = gzip.compress(b"timestamp,lux\n2020-01-01 00:00,5\n2020-01-01 00:01,6\n2020-01-01 00:02,7\n")
        self.assertEqual(len(expand("a.csv.gz", gz)), 1)

    def test_raw_accel_units_and_eda(self) -> None:
        rows = "".join(f"2020-02-13 15:28:{50 + i // 32:02d}.{(i % 32) * 31250:06d},{-2 + i % 3},1,64\n" for i in range(64))
        p = parse("ACC_001.csv", "datetime, acc_x, acc_y, acc_z\n" + rows)
        z = p.subjects[0].channel("accel_z")
        self.assertEqual(z.unit, "g")
        self.assertAlmostEqual(float(z.values[0]), 1.0)
        self.assertTrue(any("1/64 g" in n for n in p.notes))
        s = parse("a.csv", "time,ax,ay,az\n2020-01-01 00:00:00,0,0,9.8\n2020-01-01 00:00:01,0,0,9.7\n2020-01-01 00:00:02,0,0,9.9\n").subjects[0]
        self.assertAlmostEqual(float(s.channel("accel_z").values[0]), 9.8)
        c = parse("EDA_001.csv", "datetime, eda\n2020-02-13 15:28:50.000,0.1\n2020-02-13 15:28:50.250,0.2\n2020-02-13 15:28:50.500,0.3\n").subjects[0].channels[0]
        self.assertEqual((c.name, c.unit), ("EDA", "µS"))

    def test_timestamped_ibi_in_seconds(self) -> None:
        # BIG IDEAs / Empatica re-export: datetime + IBI in seconds, plus an extra column; one artifact
        rows = "".join(f"2020-02-13 15:33:{i:02d}.000000,{3.5 if i == 30 else 0.8},33.{i % 10}\n" for i in range(60))
        p = parse("IBI_001.csv", "datetime, ibi, temp\n" + rows)
        s = p.subjects[0]
        hr = s.channel("heart_rate")
        self.assertEqual((len(s.t), len(hr.values), len(s.channel("temperature").values)), (60, 60, 60))
        self.assertAlmostEqual(float(hr.values[0]), 75.0)
        self.assertTrue(np.isnan(hr.values[30]))
        self.assertEqual(hr.name, "Heart rate (IBI)")

    def test_merge_sensors(self) -> None:
        light = parse("Light.csv", '"Time (s)","Illuminance (lx)"\n0,1\n1,2\n2,3\n')
        acc = parse("Acc.csv", '"Time (s)","Acceleration x (m/s^2)","Acceleration y (m/s^2)","Acceleration z (m/s^2)"\n0,0,0,9.8\n1,0,0,9.8\n2,0,0,9.8\n')
        subjects, _ = combine([light, acc], "auto")
        self.assertEqual(len(subjects), 1)
        self.assertEqual({c.role for c in subjects[0].channels}, {"light", "accel_x", "accel_y", "accel_z"})
        # clock-time files parsed at different resolutions (whole seconds vs. sub-second) must stay on the calendar
        hr = parse("HR.csv", "datetime, hr\n" + "".join(f"2020-02-{13 + d} {h:02d}:00:00,{70 + h}\n" for d in range(3) for h in range(24)))
        tp = parse("TEMP.csv", "datetime, temp\n" + "".join(f"2020-02-{13 + d} {h:02d}:00:00.250,33.{h % 10}\n" for d in range(3) for h in range(24)))
        (m,), _ = combine([hr, tp], "auto")
        self.assertEqual((str(m.t[0]), str(m.t[-1])), ("2020-02-13 00:00:00", "2020-02-15 23:00:00.250000"))
        rep = run_full_analysis(subject_to_dataset(m), CircadianAnalysisConfig(epoch_minutes=60))
        self.assertGreater(rep.binned.n_days, 2.9)


if __name__ == "__main__":
    unittest.main()
