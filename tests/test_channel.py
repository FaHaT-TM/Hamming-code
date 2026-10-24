"""Тесты канала BPSK + AWGN и метрик (интервалы, теория)."""

import math
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import channel, metrics  # noqa: E402


class TestBPSK(unittest.TestCase):
    def test_mapping(self):
        np.testing.assert_array_equal(channel.bpsk_modulate([0, 1, 1, 0]),
                                      np.array([1.0, -1.0, -1.0, 1.0]))

    def test_energy_is_one(self):
        bits = np.array([0, 1] * 500, dtype=np.uint8)
        x = channel.bpsk_modulate(bits)
        self.assertAlmostEqual(float(np.mean(x ** 2)), 1.0, places=12)

    def test_hard_demod_clean(self):
        bits = np.array([0, 1, 0, 0, 1, 1], dtype=np.uint8)
        np.testing.assert_array_equal(channel.bpsk_demodulate_hard(channel.bpsk_modulate(bits)),
                                      bits)

    def test_hard_demod_threshold(self):
        np.testing.assert_array_equal(channel.bpsk_demodulate_hard([0.1, -0.1, 0.0]),
                                      np.array([0, 1, 0], dtype=np.uint8))

    def test_invalid_bits(self):
        with self.assertRaises(ValueError):
            channel.bpsk_modulate([0, 2])


class TestSigma(unittest.TestCase):
    def test_formula(self):
        """σ² = 1/(2·R_eff·γ_b); σ = sqrt(σ²)."""
        for ebno in (0.0, 3.0, 10.0):
            for rate in (1.0, 4 / 7, 10_000 / 20_004):
                expected = 1.0 / (2.0 * rate * 10 ** (ebno / 10))
                self.assertAlmostEqual(channel.variance_from_ebno(ebno, rate), expected, places=15)
                self.assertAlmostEqual(channel.sigma_from_ebno(ebno, rate), expected ** 0.5,
                                       places=15)

    def test_rate_matters(self):
        """Одинаковый Eb/N0 при разных скоростях требует разной σ²."""
        a = channel.variance_from_ebno(5.0, 1.0)
        b = channel.variance_from_ebno(5.0, 4 / 7)
        self.assertGreater(b, a)
        self.assertAlmostEqual(b / a, 7 / 4, places=12)

    def test_invalid_rate(self):
        with self.assertRaises(ValueError):
            channel.variance_from_ebno(0.0, 0.0)
        with self.assertRaises(ValueError):
            channel.variance_from_ebno(0.0, 1.5)


class TestNoiseStatistics(unittest.TestCase):
    def test_noise_mean_and_variance(self):
        rng = np.random.default_rng(2024)
        sigma = 0.5
        n = rng.normal(0.0, sigma, size=400_000)
        self.assertLess(abs(float(np.mean(n))), 4 * sigma / math.sqrt(n.size))
        self.assertLess(abs(float(np.var(n)) - sigma ** 2), 0.02 * sigma ** 2)

    def test_awgn_preserves_symbol_plus_noise(self):
        rng = np.random.default_rng(5)
        x = np.ones(1000)
        sigma = 0.1
        y = channel.add_awgn(x, sigma, rng)
        self.assertLess(abs(float(np.std(y - x)) - sigma), 0.01)
        self.assertLess(abs(float(np.mean(y - x))), 0.01)

    def test_requires_rng(self):
        with self.assertRaises(ValueError):
            channel.transmit_hard([0, 1], 5.0, 1.0)


class TestEmpiricalErrorRate(unittest.TestCase):
    def test_uncoded_matches_theory(self):
        """p = Q(√(2γ_b)) подтверждается на 2·10^5 битах при 4 дБ."""
        ebno, n = 4.0, 200_000
        rng = np.random.default_rng(111)
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        rx = channel.transmit_hard(bits, ebno, 1.0, rng)
        p_emp = metrics.count_bit_errors(bits, rx) / n
        p_th = channel.bit_error_probability(ebno, 1.0)
        se = math.sqrt(p_th * (1 - p_th) / n)
        self.assertLess(abs(p_emp - p_th), 4.5 * se,
                        msg=f"эмп. {p_emp:.5f} vs теор. {p_th:.5f}")

    def test_hamming_bit_error_rate_matches_theory(self):
        """Вероятность ошибки кодового бита при R=4/7 соответствует Q(√(2Rγ_b))."""
        ebno, n = 5.0, 200_000
        rate = 4 / 7
        rng = np.random.default_rng(222)
        bits = rng.integers(0, 2, size=n, dtype=np.uint8)
        rx = channel.transmit_hard(bits, ebno, rate, rng)
        p_emp = metrics.count_bit_errors(bits, rx) / n
        p_th = channel.bit_error_probability(ebno, rate)
        se = math.sqrt(p_th * (1 - p_th) / n)
        self.assertLess(abs(p_emp - p_th), 4.5 * se)


class TestQfunc(unittest.TestCase):
    def test_values(self):
        self.assertAlmostEqual(channel.qfunc(0.0), 0.5, places=15)
        self.assertAlmostEqual(channel.qfunc(1.0), 0.15865525393145707, places=12)
        self.assertAlmostEqual(channel.qfunc(3.0), 0.0013498980316300946, places=14)

    def test_array(self):
        out = channel.qfunc(np.array([0.0, 1.0]))
        self.assertEqual(out.shape, (2,))
        self.assertAlmostEqual(float(out[0]), 0.5, places=15)

    def test_bpsk_theory_equals_half_erfc(self):
        for x in (0.0, 1.0, 4.0):
            self.assertAlmostEqual(metrics.bpsk_ber_theory(x),
                                   0.5 * math.erfc(math.sqrt(10 ** (x / 10))), places=15)


class TestHammingTheory(unittest.TestCase):
    def test_weight_profile(self):
        """Точный профиль Σz(e) по весам должен совпасть со спецификацией."""
        np.testing.assert_array_equal(metrics.hamming_weight_profile(),
                                      np.array([0, 0, 36, 76, 64, 48, 28, 4]))

    def test_series_equals_polynomial(self):
        """P_b = 9p² − 26p³ + 30p⁴ − 12p⁵ совпадает с суммой по шаблонам."""
        for p in (0.0, 1e-3, 0.01, 0.05, 0.1, 0.25, 0.5, 0.9, 1.0):
            poly = 9 * p ** 2 - 26 * p ** 3 + 30 * p ** 4 - 12 * p ** 5
            self.assertAlmostEqual(metrics.hamming_ber_exact_series(p), poly, places=14)

    def test_boundary_values(self):
        self.assertAlmostEqual(metrics.hamming_ber_exact_series(0.0), 0.0, places=15)
        self.assertAlmostEqual(metrics.hamming_ber_exact_series(0.5), 0.5, places=12)
        self.assertAlmostEqual(metrics.hamming_ber_exact_series(1.0), 1.0, places=12)

    def test_asymptotic_coefficient(self):
        """При p → 0: P_b ~ 9p² (а не 6p², как давала отклонённая граница)."""
        p = 1e-4
        self.assertAlmostEqual(metrics.hamming_ber_exact_series(p) / (9 * p ** 2), 1.0, places=3)

    def test_array_input(self):
        out = metrics.hamming_ber_theory(np.array([0.0, 5.0]))
        self.assertEqual(out.shape, (2,))
        # при 0 дБ p велико, кривая не равна нулю; сверяем со скалярным вызовом
        self.assertAlmostEqual(float(out[0]), metrics.hamming_ber_theory(0.0), places=15)
        self.assertGreater(float(out[0]), float(out[1]))


class TestIntervals(unittest.TestCase):
    def test_clopper_pearson_known(self):
        """Известное значение: k=5, m=50, 95 % → примерно (0.033, 0.218)."""
        low, high = metrics.clopper_pearson(5, 50)
        self.assertAlmostEqual(low, 0.0333, places=3)
        self.assertAlmostEqual(high, 0.2183, places=3)

    def test_clopper_pearson_edges(self):
        self.assertEqual(metrics.clopper_pearson(0, 10)[0], 0.0)
        self.assertEqual(metrics.clopper_pearson(10, 10)[1], 1.0)
        with self.assertRaises(ValueError):
            metrics.clopper_pearson(11, 10)

    def test_zero_error_bound(self):
        for m in (10, 1000, 2_000_000):
            expected = 1.0 - 0.05 ** (1.0 / m)
            self.assertAlmostEqual(metrics.zero_error_upper_bound(m), expected, places=15)

    def test_zero_error_bound_vs_three_over_n(self):
        """При больших M граница ≈ 2.996/M, а не 3/M_info для декодированных битов."""
        self.assertAlmostEqual(metrics.zero_error_upper_bound(2_000_000) * 2_000_000,
                               2.9957, places=3)

    def test_conservative_interval_contains_ber(self):
        res = metrics.ber_confidence(n_errors=40, n_info_bits=2_000_000,
                                     n_units=500_000, info_per_unit=4,
                                     n_units_with_errors=30)
        self.assertLess(res["ci_low"], res["ber"])
        self.assertGreater(res["ci_high"], res["ber"])
        self.assertEqual(res["interval_type"], "conservative_95_by_independent_units")
        # нижняя граница делится на ell, верхняя — нет
        self.assertAlmostEqual(res["ci_low"] * 4, metrics.clopper_pearson(30, 500_000)[0],
                               places=12)

    def test_zero_errors_interval(self):
        res = metrics.ber_confidence(n_errors=0, n_info_bits=1_000_000,
                                     n_units=100, info_per_unit=10_000)
        self.assertEqual(res["ber"], 0.0)
        self.assertEqual(res["interval_type"], "conservative_one_sided_95_no_errors")
        self.assertAlmostEqual(res["upper95"], 1 - 0.05 ** (1 / 100), places=15)

    def test_cluster_normal(self):
        low, high, se = metrics.cluster_normal_interval(z_sum=100.0, z2_sum=400.0,
                                                        n_units=100, length=10_000)
        mean = (100.0 / 10_000) / 100
        self.assertAlmostEqual(mean, 1e-4, places=15)
        self.assertLess(low, mean)
        self.assertGreater(high, mean)
        self.assertGreaterEqual(low, 0.0)


class TestEnergyGain(unittest.TestCase):
    def test_crossing_simple(self):
        x = metrics.crossing_ebn0(4.0, 1e-2, 6.0, 1e-4, 1e-3)
        self.assertAlmostEqual(x, 5.0, places=12)

    def test_no_extrapolation(self):
        self.assertIsNone(metrics.crossing_ebn0(4.0, 1e-2, 6.0, 1e-4, 1e-6))
        self.assertIsNone(metrics.crossing_ebn0(4.0, 1e-2, 6.0, 1e-4, 1e-1))

    def test_zero_ber_not_used(self):
        self.assertIsNone(metrics.crossing_ebn0(4.0, 0.0, 6.0, 1e-4, 1e-6))

    def test_gain(self):
        uncoded = {4.0: 1e-2, 5.0: 1e-3, 6.0: 1e-5}
        coded = {4.0: 1e-3, 5.0: 1e-4, 6.0: 1e-6}
        res = metrics.energy_gain_db(uncoded, coded, 1e-4)
        # coded достигает 1e-4 ровно в 5 дБ, uncoded — в 5.5 дБ (по lg-интерполяции)
        self.assertAlmostEqual(res["gain_db"], 0.5, places=10)

    def test_gain_undefined(self):
        res = metrics.energy_gain_db({0.0: 0.1}, {0.0: 0.05}, 1e-5)
        self.assertIsNone(res["gain_db"])


class TestBerCounters(unittest.TestCase):
    def test_count_and_ber(self):
        ref = np.array([0, 1, 1, 0, 1], dtype=np.uint8)
        dec = np.array([0, 1, 0, 0, 1], dtype=np.uint8)
        self.assertEqual(metrics.count_bit_errors(ref, dec), 1)
        self.assertAlmostEqual(metrics.ber(1, 5), 0.2)

    def test_length_mismatch_raises(self):
        with self.assertRaises(ValueError):
            metrics.count_bit_errors(np.zeros(4), np.zeros(5))

    def test_invalid_ber(self):
        with self.assertRaises(ValueError):
            metrics.ber(5, 4)
        with self.assertRaises(ValueError):
            metrics.ber(0, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
