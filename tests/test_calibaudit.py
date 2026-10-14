#!/usr/bin/env python3
"""Tests for calibaudit on synthetic data whose calibration is known. Run: python3 tests/test_calibaudit.py"""
import csv
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
import calibaudit as ca  # noqa: E402


def synth(t_true=1.0, days=40, per_day=150, seed=1, lo=0.55, hi=0.995, link="logit", day_shock=0.0):
    """Forecast p uniform on [lo, hi]; the truth is p tempered by t_true, so a perfect audit recovers T = t_true.
    day_shock shifts every row of a day by the same logit offset (correlated rows, the reason to cluster by day)."""
    rng, rows = random.Random(seed), []
    for d in range(days):
        shock = rng.gauss(0, day_shock)
        for _ in range(per_day):
            p = round(rng.uniform(lo, hi), 4)
            q = ca.sigmoid(ca.logit(ca.temper(p, t_true, link)) + shock)
            rows.append((p, 1 if rng.random() < q else 0, f"2026-01-{d + 1:02d}" if d < 31 else f"2026-02-{d - 30:02d}"))
    return rows


class Scores(unittest.TestCase):
    def test_brier_logloss_by_hand(self):
        self.assertAlmostEqual(ca.brier([0.9, 0.2], [1, 0]), (0.01 + 0.04) / 2)
        self.assertAlmostEqual(ca.logloss([0.9, 0.2], [1, 0]), -(math.log(0.9) + math.log(0.8)) / 2)

    def test_ece_by_hand(self):
        # ten rows at p = 0.9 with eight hits: one bin, gap 0.1
        self.assertAlmostEqual(ca.ece([0.9] * 10, [1] * 8 + [0] * 2, k=10, mode="mass"), 0.1)
        self.assertAlmostEqual(ca.ece([0.9] * 10, [1] * 8 + [0] * 2, k=10, mode="width"), 0.1)

    def test_equal_mass_bins_have_equal_counts(self):
        sizes = [len(b) for b in ca.bins_of([i / 1000 for i in range(1000)], 10, "mass")]
        self.assertEqual(sizes, [100] * 10)

    def test_mass_bins_keep_ties_together(self):
        ps = [0.9] * 30 + [0.95] * 50 + [0.97] * 20
        bins = ca.bins_of(ps, 10, "mass")
        self.assertEqual(sorted(len(b) for b in bins), [20, 30, 50])
        self.assertTrue(all(len({ps[i] for i in b}) == 1 for b in bins))

    def test_squeezed_forecasts_fill_all_mass_bins_but_not_width_bins(self):
        ps = [0.93 + 0.069 * i / 999 for i in range(1000)]
        self.assertEqual(len(ca.bins_of(ps, 10, "mass")), 10)
        self.assertLessEqual(len(ca.bins_of(ps, 10, "width")), 2)


class Temperature(unittest.TestCase):
    def test_calibrated_forecast_needs_no_temperature(self):
        rep = ca.audit(synth(1.0), samples=400)
        self.assertTrue(0.85 <= rep["temperature"] <= 1.18, rep["temperature"])
        self.assertLess(rep["raw"]["ece"], 0.02)
        lo, hi = rep["oof_minus_raw"]["brier"]["ci95"]
        self.assertGreaterEqual(hi, 0, "tempering a calibrated forecast must not look like a clear improvement")

    def test_overconfident_forecast_recovers_t_and_separates(self):
        rep = ca.audit(synth(2.0), samples=400)
        self.assertTrue(1.7 <= rep["temperature"] <= 2.35, rep["temperature"])
        self.assertTrue(all(1.6 <= t <= 2.5 for t in rep["fold_temperatures"]), rep["fold_temperatures"])
        self.assertLess(rep["oof_minus_raw"]["brier"]["ci95"][1], 0)
        self.assertTrue(rep["oof_minus_raw"]["brier"]["separated"])
        self.assertLess(rep["oof"]["ece"], rep["raw"]["ece"])
        self.assertTrue(rep["forward"]["test_minus_raw"]["brier"]["separated"])

    def test_mean_gap_interval(self):
        over = ca.audit(synth(2.0, days=20, per_day=100), samples=300)["mean_p_minus_rate"]
        fair = ca.audit(synth(1.0, days=20, per_day=100), samples=300)["mean_p_minus_rate"]
        self.assertGreater(over["ci95"][0], 0)
        self.assertLessEqual(fair["ci95"][0], 0)
        self.assertGreaterEqual(fair["ci95"][1], 0)

    def test_underconfident_forecast(self):
        rep = ca.audit(synth(0.5), samples=200)
        self.assertTrue(0.4 <= rep["temperature"] <= 0.62, rep["temperature"])

    def test_probit_link_recovers_the_spread_factor(self):
        # p = Phi(z) while the real spread is 1.5x larger: the truth is Phi(z / 1.5)
        rep = ca.audit(synth(1.5, link="probit", lo=0.6, hi=0.999), samples=200, link="probit")
        self.assertTrue(1.3 <= rep["temperature"] <= 1.75, rep["temperature"])

    def test_temper_identity_and_direction(self):
        for link in ca.LINKS:
            self.assertAlmostEqual(ca.temper(0.9, 1.0, link), 0.9, places=9)
            self.assertLess(ca.temper(0.9, 2.0, link), 0.9)
            self.assertGreater(ca.temper(0.9, 0.5, link), 0.9)
            self.assertAlmostEqual(ca.temper(0.5, 3.0, link), 0.5, places=9)


class Days(unittest.TestCase):
    def test_folds_never_split_a_day(self):
        days = [f"d{i % 13}" for i in range(500)]
        fold = ca.day_folds(days, 5, 0)
        self.assertEqual(set(fold), set(days))
        self.assertEqual(set(fold.values()), set(range(5)))

    def test_out_of_fold_never_sees_its_own_day(self):
        rows = synth(2.0, days=10, per_day=80, seed=5)
        ps, ys, days = [r[0] for r in rows], [r[1] for r in rows], [r[2] for r in rows]
        cal, _, _ = ca.out_of_fold(ps, ys, days, 5, 0)
        flipped = [1 - y if d == days[0] else y for y, d in zip(ys, days)]
        cal2, _, _ = ca.out_of_fold(ps, flipped, days, 5, 0)
        own = [i for i, d in enumerate(days) if d == days[0]]
        self.assertEqual([cal[i] for i in own], [cal2[i] for i in own])
        self.assertNotEqual(cal, cal2)

    def test_too_few_days_for_folds(self):
        with self.assertRaises(ValueError):
            ca.day_folds(["a", "b", "c"], 5, 0)

    def test_day_clusters_widen_the_interval_when_days_are_correlated(self):
        rows = synth(1.0, days=40, per_day=100, day_shock=0.8, seed=3)
        ys, days = [r[1] for r in rows], [r[2] for r in rows]
        ones = [1.0] * len(ys)
        by_day = ca.day_bootstrap(days, [float(y) for y in ys], ones, 1000, 0)["ci95"]
        by_row = ca.day_bootstrap([str(i) for i in range(len(ys))], [float(y) for y in ys], ones, 1000, 0)["ci95"]
        self.assertGreater(by_day[1] - by_day[0], 2 * (by_row[1] - by_row[0]))

    def test_bootstrap_is_seeded(self):
        rows = synth(1.0, days=12, per_day=30)
        a = ca.day_bootstrap([r[2] for r in rows], [float(r[1]) for r in rows], [1.0] * len(rows), 300, 7)
        b = ca.day_bootstrap([r[2] for r in rows], [float(r[1]) for r in rows], [1.0] * len(rows), 300, 7)
        self.assertEqual(a, b)


class Cut(unittest.TestCase):
    def setUp(self):
        # 100 rows at 0.99 with 1 error, 100 rows at 0.95 with 10 errors
        self.values = [0.99] * 100 + [0.95] * 100
        self.correct = [False] + [True] * 99 + [False] * 10 + [True] * 90

    def test_conservative_cut(self):
        # at 0.99: (1+1)/(100+1) = 0.0198 holds 5%; adding 0.95: (11+1)/(200+1) = 0.0597 does not
        self.assertEqual(ca.select_cut(self.values, self.correct, 0.05), 0.99)

    def test_conservative_rule_refuses_a_short_clean_run(self):
        # 20 rows, no errors: plain 0/20 clears 4%, conservative (0+1)/(20+1) = 4.8% does not
        self.assertEqual(ca.select_cut([0.99] * 20, [True] * 20, 0.04, conservative=False), 0.99)
        self.assertIsNone(ca.select_cut([0.99] * 20, [True] * 20, 0.04))

    def test_plain_cut_goes_lower(self):
        self.assertEqual(ca.select_cut(self.values, self.correct, 0.06, conservative=False), 0.95)

    def test_zero_errors_report_the_rule_of_three(self):
        line = ca._err({"accepted": 150, "errors": 0, "error": {"value": 0.0, "ci95": [0.0, 0.0]}})
        self.assertIn("2.00%", line)
        self.assertNotIn("[0.00%", line)

    def test_impossible_budget(self):
        self.assertIsNone(ca.select_cut(self.values, self.correct, 0.001))

    def test_ties_are_judged_as_a_whole(self):
        # three rows tied at 0.97, one of them wrong: the level is judged on all three, not on its first two
        self.assertIsNone(ca.select_cut([0.97, 0.97, 0.97], [True, True, False], 0.2, conservative=False))

    def test_gate_scales_accept_the_same_rows(self):
        rows = synth(1.6, days=20, per_day=60, lo=0.6, hi=0.999)
        ps = [r[0] for r in rows]
        correct = [(p >= 0.5) == bool(y) for p, (_, y, _) in zip(ps, rows)]
        t = ca.fit_temperature(ps, [r[1] for r in rows])
        accepted = []
        for gate in ca.GATES:
            vals = ca.gate_values(ps, gate, t)
            cut = ca.select_cut(vals, correct, 0.08)
            accepted.append(sum(v >= cut for v in vals))
            if gate == "z":
                self.assertTrue(1 < cut < 4, cut)
        self.assertEqual(len(set(accepted)), 1, accepted)

    def test_cut_fitted_on_one_scale_and_applied_on_another_is_wrong(self):
        # the laya lesson: a number chosen on the calibrated scale admits far more when compared against raw p
        rows = synth(2.0, days=20, per_day=80, lo=0.6, hi=0.999)
        ps, ys = [r[0] for r in rows], [r[1] for r in rows]
        correct = [(p >= 0.5) == bool(y) for p, y in zip(ps, ys)]
        t = ca.fit_temperature(ps, ys)
        cal_cut = ca.select_cut(ca.gate_values(ps, "calibrated", t), correct, 0.08)
        right = sum(v >= cal_cut for v in ca.gate_values(ps, "calibrated", t))
        wrong = sum(p >= cal_cut for p in ps)
        self.assertGreater(wrong, right)


class Cli(unittest.TestCase):
    def write(self, rows, header=("p", "outcome", "day")):
        fd, name = tempfile.mkstemp(suffix=".csv")
        with open(fd, "w", newline="") as f:
            w = csv.writer(f)
            w.writerow(header)
            w.writerows(rows)
        self.addCleanup(Path(name).unlink)
        return name

    def run_cli(self, *args):
        return subprocess.run([sys.executable, str(ROOT / "calibaudit.py"), *args], capture_output=True, text=True)

    def test_end_to_end_and_deterministic(self):
        path = self.write(synth(1.5, days=15, per_day=40))
        fd, out = tempfile.mkstemp(suffix=".json")
        os.close(fd)
        self.addCleanup(Path(out).unlink)
        a = self.run_cli(path, "--samples", "200", "--gate", "z", "--json", out)
        b = self.run_cli(path, "--samples", "200", "--gate", "z")
        self.assertEqual(a.returncode, 0, a.stderr)
        self.assertEqual(a.stdout, b.stdout)
        for word in ("Brier", "reliability", "z >=", "out-of-fold", "forward"):
            self.assertIn(word, a.stdout)
        rep = json.loads(Path(out).read_text())
        self.assertEqual(rep["n"], 600)
        self.assertEqual(len(rep["reliability"]), 10)

    def test_bad_inputs_exit_2(self):
        cases = [self.write([(0.9, 1, "d1")], header=("p", "y", "day")), self.write([(1.2, 1, "d1")]),
                 self.write([(0.9, 2, "d1")]), self.write([(0.9, 1, "")]), self.write([("x", 1, "d1")]),
                 self.write([(0.9, 1, "d1")] * 10)]   # last: one day, fewer than 5 folds
        for path in cases:
            r = self.run_cli(path)
            self.assertEqual(r.returncode, 2, (path, r.stdout))
            self.assertIn("calibaudit:", r.stderr)


if __name__ == "__main__":
    unittest.main(verbosity=1)
