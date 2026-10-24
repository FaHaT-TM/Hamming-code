"""Тесты экспериментального слоя: воспроизводимость, подсчёт ошибок, сохранение."""

import csv
import os
import sys
import tempfile
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import convolutional as conv  # noqa: E402
from src import hamming, metrics, plotting, simulation  # noqa: E402
from src import viterbi  # noqa: E402


def small_plan(**kwargs):
    base = dict(
        seed=1234,
        ebno_db=(2.0, 4.0),
        n_info_uncoded=20_000,
        n_info_hamming=20_000,
        n_info_convolutional=20_000,
        packet_length=2_000,
        chunk_bits=8_000,
        packets_per_batch=2,
    )
    base.update(kwargs)
    return simulation.Plan(**base)


class TestPlan(unittest.TestCase):
    def test_r_eff_values(self):
        plan = simulation.Plan()
        r = plan.r_eff
        self.assertEqual(r["uncoded"], 1.0)
        self.assertAlmostEqual(r["hamming"], 4 / 7, places=15)
        self.assertAlmostEqual(r["convolutional"], 10_000 / 20_004, places=15)

    def test_default_plan_values(self):
        """Фиксированные до запуска объёмы должны совпадать со спецификацией."""
        plan = simulation.Plan()
        self.assertEqual(plan.ebno_db, tuple(float(x) for x in range(11)))
        self.assertEqual(plan.n_info_uncoded, 2_000_000)
        self.assertEqual(plan.n_info_hamming, 2_000_000)
        self.assertEqual(plan.n_info_convolutional, 1_000_000)
        self.assertEqual(plan.packet_length, 10_000)
        self.assertEqual(plan.seed, 20261024)


class TestPointResult(unittest.TestCase):
    def test_pt_noise_free_hamming_zero_errors(self):
        """При большом Eb/N0 и малом объёме ошибки могут отсутствовать — проверяем структуру."""
        plan = small_plan(ebno_db=(20.0,))
        res = simulation.run_point("hamming", 20.0, plan)
        self.assertEqual(res.n_errors, 0)
        self.assertEqual(res.ber, 0.0)
        self.assertEqual(res.ci_type, "conservative_one_sided_95_no_errors")
        self.assertAlmostEqual(res.upper95, 1 - 0.05 ** (1 / res.n_units), places=15)
        self.assertEqual(res.n_info_bits, plan.n_info_hamming)
        self.assertEqual(res.n_units, plan.n_info_hamming // 4)

    def test_volumes_and_counters(self):
        plan = small_plan()
        for scheme in ("uncoded", "hamming", "convolutional"):
            res = simulation.run_point(scheme, 3.0, plan)
            self.assertEqual(res.n_info_bits, plan.n_info(scheme))
            self.assertEqual(res.ber, res.n_errors / res.n_info_bits)
            self.assertGreater(res.n_errors, 0)
            self.assertEqual(res.n_tx_bits > 0, True)
            if scheme == "hamming":
                self.assertEqual(res.n_tx_bits, res.n_info_bits * 7 // 4)
                self.assertEqual(res.n_units, res.n_info_bits // 4)
            if scheme == "convolutional":
                self.assertEqual(res.n_units, res.n_info_bits // plan.packet_length)
                self.assertEqual(res.n_tx_bits,
                                 res.n_units * 2 * (plan.packet_length + conv.TAIL))
                self.assertLessEqual(res.n_units_with_errors, res.n_units)

    def test_sigma2_and_rate_stored(self):
        res = simulation.run_point("hamming", 5.0, small_plan(ebno_db=(5.0,)))
        self.assertAlmostEqual(res.r_eff, 4 / 7, places=15)
        self.assertAlmostEqual(res.sigma2, 1 / (2 * (4 / 7) * 10 ** 0.5), places=12)


class TestReproducibility(unittest.TestCase):
    def test_same_seed_same_result(self):
        plan = small_plan()
        for scheme in ("uncoded", "hamming", "convolutional"):
            a = simulation.run_point(scheme, 3.0, plan)
            b = simulation.run_point(scheme, 3.0, plan)
            self.assertEqual(a.n_errors, b.n_errors, msg=scheme)
            self.assertEqual(a.ber, b.ber, msg=scheme)

    def test_different_seed_different_result(self):
        a = simulation.run_point("uncoded", 3.0, small_plan(seed=1))
        b = simulation.run_point("uncoded", 3.0, small_plan(seed=2))
        self.assertNotEqual(a.n_errors, b.n_errors)

    def test_streams_are_independent_of_order(self):
        """Поток точки не зависит от того, какие ещё точки считались."""
        plan = small_plan()
        first = simulation.run_point("hamming", 4.0, plan)
        simulation.run_point("hamming", 2.0, plan)
        again = simulation.run_point("hamming", 4.0, plan)
        self.assertEqual(first.n_errors, again.n_errors)

    def test_hamming_ber_close_to_theory(self):
        """Монте-Карло согласуется с точной теорией в пределах статистической погрешности."""
        ebno = 3.0
        plan = small_plan(ebno_db=(ebno,), n_info_hamming=400_000)
        res = simulation.run_point("hamming", ebno, plan)
        theory = metrics.hamming_ber_theory(ebno)
        # консервативный интервал по блокам должен накрывать точное значение
        self.assertLess(res.ci_low, theory)
        self.assertGreater(res.ci_high, theory)


class TestOrdering(unittest.TestCase):
    def test_ber_decreases_with_ebno(self):
        plan = small_plan(ebno_db=(0.0, 3.0, 6.0))
        for scheme in ("uncoded", "hamming", "convolutional"):
            bers = [simulation.run_point(scheme, x, plan).ber for x in plan.ebno_db]
            self.assertGreater(bers[0], bers[1], msg=scheme)
            self.assertGreater(bers[1], bers[2], msg=scheme)

    def test_run_experiment_covers_all_points(self):
        plan = small_plan()
        results = simulation.run_experiment(plan, verbose=False)
        self.assertEqual(len(results), 3 * len(plan.ebno_db))


class TestCsv(unittest.TestCase):
    def test_write_and_read_roundtrip(self):
        plan = small_plan()
        results = simulation.run_experiment(plan, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "res.csv")
            simulation.write_csv(results, path)

            with open(path, newline="", encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), len(results))
            required = {"scheme", "ebno_db", "n_info_bits", "n_errors", "ber",
                        "n_units", "n_units_with_errors", "sum_z2", "seed",
                        "r_nominal", "r_eff", "sigma2", "ci_type", "status",
                        "length", "tail_bits", "elapsed_s"}
            self.assertTrue(required.issubset(set(rows[0].keys())))

            # значения не округлены
            first = results[0]
            row = next(r for r in rows if r["scheme"] == first.scheme
                       and float(r["ebno_db"]) == first.ebno_db)
            self.assertAlmostEqual(float(row["ber"]), first.ber, places=18)
            self.assertEqual(int(row["n_errors"]), first.n_errors)

            # читается обратно для графиков
            back = plotting.read_csv(path)
            self.assertEqual(len(back), len(results))
            self.assertEqual(back[0]["scheme"], results[0].scheme)

    def test_theory_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = os.path.join(tmp, "theory.csv")
            simulation.write_theory_csv(path, ebno_db=np.array([0.0, 5.0, 10.0]))
            rows = plotting.read_csv(path)
            self.assertEqual(len(rows), 3)
            # при низком Eb/N0 кодирование проигрывает, при высоком — выигрывает
            self.assertGreater(rows[0]["ber_hamming_theory"], rows[0]["ber_bpsk_theory"])
            self.assertLess(rows[-1]["ber_hamming_theory"], rows[-1]["ber_bpsk_theory"])


class TestNoBitLoss(unittest.TestCase):
    def test_information_bits_conserved(self):
        """Декодированные последовательности по длине совпадают с исходными."""
        rng = np.random.default_rng(7)
        info = rng.integers(0, 2, size=4 * 300, dtype=np.uint8)
        self.assertEqual(hamming.decode(hamming.encode(info)).size, info.size)

        packet = rng.integers(0, 2, size=500, dtype=np.uint8)
        self.assertEqual(viterbi.decode(conv.encode(packet)).size, packet.size)

        self.assertEqual(conv.encode(packet).size, 2 * (packet.size + conv.TAIL))


class TestFigures(unittest.TestCase):
    def test_ber_plot_from_csv(self):
        plan = small_plan()
        results = simulation.run_experiment(plan, verbose=False)
        with tempfile.TemporaryDirectory() as tmp:
            csv_path = os.path.join(tmp, "res.csv")
            th_path = os.path.join(tmp, "theory.csv")
            simulation.write_csv(results, csv_path)
            simulation.write_theory_csv(th_path, ebno_db=np.linspace(0, 6, 25))
            paths = plotting.ber_curve_plot(csv_path, th_path, tmp)
            self.assertEqual(len(paths), 2)
            for p in paths:
                self.assertTrue(os.path.getsize(p) > 1000)

    def test_auxiliary_figures(self):
        with tempfile.TemporaryDirectory() as tmp:
            for func in (plotting.channel_scheme_plot, plotting.state_diagram_plot,
                         plotting.trellis_example_plot, plotting.hamming_correction_plot):
                paths = func(tmp)
                self.assertEqual(len(paths), 2)




class TestZeroHistograms(unittest.TestCase):
    """Гистограммы Z_j (добавлены по замечанию Агента №2, §5)."""

    def test_hist_consistent_with_counters(self):
        plan = small_plan()
        for scheme in ("uncoded", "hamming", "convolutional"):
            res = simulation.run_point(scheme, 4.0, plan)
            hist = metrics.hist_from_str(res.z_hist)
            self.assertEqual(sum(hist.values()), res.n_units, msg=scheme)
            self.assertEqual(sum(z * c for z, c in hist.items()), res.n_errors, msg=scheme)
            self.assertAlmostEqual(sum(z * z * c for z, c in hist.items()),
                                   res.sum_z2, places=9, msg=scheme)

    def test_hist_to_str_roundtrip(self):
        hist = {0: 10, 3: 2, 7: 1}
        self.assertEqual(metrics.hist_from_str(metrics.hist_to_str(hist)), hist)

    def test_cluster_stats(self):
        hist = {0: 900, 1: 80, 2: 20}
        st = metrics.cluster_stats_from_hist(hist, ell=4)
        self.assertEqual(st["M"], 1000)
        self.assertEqual(st["E"], 120)
        self.assertEqual(st["K"], 100)
        self.assertAlmostEqual(st["mean_W"], 120 / (1000 * 4))
        self.assertGreater(st["se"], 0)

    def test_cluster_interval_requires_enough_events(self):
        few = metrics.cluster_normal_from_hist({0: 90, 5: 10}, 1)
        self.assertFalse(few["applicable"])
        many = metrics.cluster_normal_from_hist({0: 900, 1: 100}, 1)
        self.assertTrue(many["applicable"])
        self.assertLess(many["ci_low"], many["mean_W"])
        self.assertGreater(many["ci_high"], many["mean_W"])

    def test_bootstrap_gain_shape(self):
        """Bootstrap возвращает точечную оценку и интервал, содержащий её."""
        curves = {
            "uncoded": {4.0: {0: 9900, 1: 100}, 5.0: {0: 9990, 1: 10}},
            "hamming": {4.0: {0: 960, 1: 35, 2: 5}, 5.0: {0: 992, 1: 7, 2: 1}},
        }
        ells = {"uncoded": 1, "hamming": 4}
        out = metrics.bootstrap_gain(curves, ells, target=1e-2, n_boot=200, seed=7)
        h = out["hamming"]
        self.assertIsNotNone(h["gain_db"])
        self.assertLessEqual(h["ci_low"], h["ci_high"])
        self.assertEqual(h["n_boot"], 200)
        self.assertGreater(h["n_valid"], 0)
        self.assertLessEqual(h["invalid_fraction"], 1.0)
        self.assertIn("percentile", h["method"])


if __name__ == "__main__":
    unittest.main(verbosity=2)
