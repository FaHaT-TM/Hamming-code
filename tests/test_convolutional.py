"""Тесты свёрточного кодера R=1/2, K=3, (7,5)₈ (таблица переходов, контрольные векторы, d_free).

Критерии корректности из спецификации (модуль
«свёрточный кодер»): все 8 переходов, длина выхода, конечное состояние 00,
нулевой вход → нулевой выход, контрольные последовательности, независимость
пакетов друг от друга, d_free = 5.
"""

import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import convolutional as conv  # noqa: E402


def arr(s):
    return np.array([int(c) for c in s], dtype=np.uint8)


def as_str(a):
    return "".join(str(int(b)) for b in np.ravel(a))


class TestParameters(unittest.TestCase):
    def test_parameters(self):
        self.assertEqual(conv.K, 3)
        self.assertEqual(conv.N_OUT, 2)
        self.assertEqual(conv.TAIL, 2)
        self.assertEqual(conv.N_STATES, 4)
        self.assertEqual(conv.L_DEFAULT, 10_000)
        self.assertEqual(conv.D_FREE, 5)

    def test_rate_effective(self):
        self.assertAlmostEqual(conv.rate_effective(10_000), 10_000 / 20_004, places=15)
        self.assertAlmostEqual(conv.energy_penalty_db(10_000), 0.0008685, places=6)
        self.assertAlmostEqual(conv.energy_penalty_db(10_000),
                               10 * np.log10(10_002 / 10_000), places=12)

    def test_rate_shorter_packet_is_lower(self):
        self.assertLess(conv.rate_effective(10), conv.rate_effective(10_000))


class TestTransitionTable(unittest.TestCase):
    """Все восемь переходов должны соответствовать таблице спецификации."""

    SPEC = {
        (0, 0): ("00", 0), (0, 1): ("11", 2),
        (1, 0): ("11", 0), (1, 1): ("00", 2),
        (2, 0): ("10", 1), (2, 1): ("01", 3),
        (3, 0): ("01", 1), (3, 1): ("10", 3),
    }

    def test_table(self):
        self.assertEqual(len(conv.TRANSITIONS), 8)
        for s in range(4):
            for u in (0, 1):
                with self.subTest(state=s, u=u):
                    out, nxt = self.SPEC[(s, u)]
                    self.assertEqual(int(conv.NEXT_TABLE[s, u]), nxt)
                    got = conv.OUT_TABLE[s, u]
                    self.assertEqual(f"{got >> 1}{got & 1}", out)

    def test_equations(self):
        """Проверка по уравнениям: v1 = u⊕a⊕b, v2 = u⊕b, S' = (u,a)."""
        for s in range(4):
            a, b = s >> 1, s & 1
            for u in (0, 1):
                v1 = u ^ a ^ b
                v2 = u ^ b
                got = conv.OUT_TABLE[s, u]
                self.assertEqual((got >> 1, got & 1), (v1, v2))
                self.assertEqual(int(conv.NEXT_TABLE[s, u]), 2 * u + a)


class TestEncodeKnownVectors(unittest.TestCase):
    """Контрольные последовательности из спецификации (включая хвост)."""

    KNOWN = [
        ("0", "000000"),
        ("1", "111011"),
        ("10", "11101100"),
        ("11", "11010111"),
        ("1011", "111000010111"),
    ]

    def test_known_vectors(self):
        for info, expected in self.KNOWN:
            with self.subTest(info=info):
                self.assertEqual(as_str(conv.encode(arr(info))), expected)

    def test_output_length_includes_tail(self):
        for length in (1, 2, 5, 1000):
            self.assertEqual(conv.encode(np.zeros(length, dtype=np.uint8)).size,
                             2 * (length + conv.TAIL))

    def test_zero_input_zero_output(self):
        self.assertTrue(np.all(conv.encode(np.zeros(37, dtype=np.uint8)) == 0))

    def test_final_state_is_zero(self):
        rng = np.random.default_rng(11)
        for length in (1, 2, 3, 17, 500):
            info = rng.integers(0, 2, size=length, dtype=np.uint8)
            self.assertEqual(conv.finish_state(info), 0)
            self.assertEqual(int(conv.state_sequence(info)[-1]), 0)

    def test_state_sequence_length(self):
        seq = conv.state_sequence(arr("1011"))
        self.assertEqual(len(seq), len("1011") + conv.TAIL + 1)


class TestBatching(unittest.TestCase):
    def test_batch_matches_single(self):
        rng = np.random.default_rng(2024)
        packets = rng.integers(0, 2, size=(8, 50), dtype=np.uint8)
        batch = conv.encode_batch(packets)
        for i in range(8):
            np.testing.assert_array_equal(batch[i], conv.encode(packets[i]))

    def test_packets_do_not_inherit_state(self):
        """Пакет, начинающийся после «грязного» состояния, кодируется как из 00."""
        p1 = arr("111111")
        p2 = arr("000000")
        batch = conv.encode_batch(np.stack([p1, p2]))
        np.testing.assert_array_equal(batch[1], conv.encode(p2))
        self.assertTrue(np.all(batch[1] == 0))

    def test_validation(self):
        with self.assertRaises(ValueError):
            conv.encode([])
        with self.assertRaises(ValueError):
            conv.encode([0, 1, 2])
        with self.assertRaises(ValueError):
            conv.encode([[0, 1]])
        with self.assertRaises(ValueError):
            conv.encode_batch([0, 1, 0])


class TestFreeDistance(unittest.TestCase):
    def test_d_free_equals_5(self):
        """Минимальный вес ненулевого пути 00 → … → 00 равен 5."""
        best = 99
        for value in range(1, 4096):          # информационные слова до 12 бит
            info = arr(format(value, "b").zfill(12))
            weight = int(conv.encode(info).sum())
            best = min(best, weight)
        self.assertEqual(best, conv.D_FREE)

    def test_known_free_distance_path(self):
        """Путь по входу 100: 00 → 10 → 01 → 00, выход 11|10|11, вес 5."""
        info = arr("100")
        code = conv.encode(info)
        # 5 шагов решётки: 3 информационных + 2 хвостовых (4 бита хвоста)
        self.assertEqual(code.size, 10)
        self.assertEqual(as_str(code[:-4]), "111011")
        self.assertEqual(as_str(code[-4:]), "0000")
        self.assertEqual(int(code.sum()), 5)


if __name__ == "__main__":
    unittest.main(verbosity=2)
