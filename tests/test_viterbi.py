"""Тесты жёсткого декодера Витерби (сверка с полным перебором, tie-break, пакетный режим).

Критерии из спецификации (модуль «жёсткий Витерби»):
* декодирование без шума для всех сообщений малой длины;
* L=1: все 6 одиночных инверсий для сообщений 0 и 1 восстанавливаются;
* одиночные инверсии на коротких пакетах (d_min завершённого кода ≥ 5);
* сравнение итоговой метрики с полным перебором 2^L сообщений;
* воспроизводимость tie-break;
* начальный и конечный нули, удаление хвоста, длины, валидация входа;
* батч == одиночная обработка на L=10000, отсутствие переполнения.
"""

import itertools
import os
import sys
import unittest

import numpy as np

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import convolutional as conv  # noqa: E402
from src import viterbi  # noqa: E402


def arr(s):
    return np.array([int(c) for c in s], dtype=np.uint8)


def as_str(a):
    return "".join(str(int(b)) for b in np.ravel(a))


def brute_force(rx, length):
    """Полный перебор всех 2^L сообщений: (лучшая метрика, лучшие сообщения)."""
    best_metric, best_msgs = None, []
    for value in range(1 << length):
        msg = np.array([(value >> (length - 1 - i)) & 1 for i in range(length)], dtype=np.uint8)
        code = conv.encode(msg)
        metric = int(np.count_nonzero(code != rx))
        if best_metric is None or metric < best_metric:
            best_metric, best_msgs = metric, [tuple(int(b) for b in msg)]
        elif metric == best_metric:
            best_msgs.append(tuple(int(b) for b in msg))
    return best_metric, best_msgs


class TestNoiseFree(unittest.TestCase):
    def test_all_short_messages(self):
        for length in range(1, 9):
            for msg in itertools.product((0, 1), repeat=length):
                info = np.array(msg, dtype=np.uint8)
                rx = conv.encode(info)
                decoded, details = viterbi.decode(rx, length=length, return_details=True)
                self.assertEqual(as_str(decoded), "".join(str(b) for b in msg),
                                 msg=f"L={length}, msg={msg}")
                # хвостовые входы на выходе декодера — нули (конечное состояние 00)
                self.assertTrue(np.all(details["tail_input"] == 0))
                self.assertEqual(int(details["end_state"]), 0)

    def test_zero_packet(self):
        decoded = viterbi.decode(np.zeros(2 * 6, dtype=np.uint8), length=4)
        np.testing.assert_array_equal(decoded, np.zeros(4, dtype=np.uint8))

    def test_length_matches_input(self):
        for length in (1, 2, 7, 100, 1000):
            info = np.random.default_rng(length).integers(0, 2, size=length, dtype=np.uint8)
            self.assertEqual(viterbi.decode(conv.encode(info)).size, length)


class TestSingleErrors(unittest.TestCase):
    def test_l1_all_six_positions(self):
        """L=1: длина кода 6; каждая из 6 одиночных инверсий исправляется."""
        for msg in ("0", "1"):
            info = arr(msg)
            code = conv.encode(info)
            self.assertEqual(code.size, 6)
            for pos in range(6):
                rx = code.copy()
                rx[pos] ^= 1
                self.assertEqual(as_str(viterbi.decode(rx, length=1)), msg,
                                 msg=f"msg={msg}, pos={pos}")

    def test_single_errors_short_packets(self):
        """Одиночная инверсия в любом месте восстанавливает пакет (d_min ≥ 5)."""
        rng = np.random.default_rng(5)
        checked = 0
        for length in range(1, 6):
            for _ in range(6):
                info = rng.integers(0, 2, size=length, dtype=np.uint8)
                code = conv.encode(info)
                for pos in range(code.size):
                    rx = code.copy()
                    rx[pos] ^= 1
                    self.assertEqual(as_str(viterbi.decode(rx, length=length)), as_str(info),
                                     msg=f"L={length}, pos={pos}")
                    checked += 1
        self.assertGreater(checked, 100)

    def test_errors_beyond_guarantee_are_not_always_corrected(self):
        """Код не гарантирует исправление трёх ошибок — тест фиксирует факт, не гарантию."""
        rng = np.random.default_rng(3)
        failures = 0
        for _ in range(50):
            info = rng.integers(0, 2, size=8, dtype=np.uint8)
            code = conv.encode(info)
            rx = code.copy()
            pos = rng.choice(code.size, size=3, replace=False)
            rx[pos] ^= 1
            if as_str(viterbi.decode(rx, length=8)) != as_str(info):
                failures += 1
        self.assertGreater(failures, 0)


class TestAgainstBruteForce(unittest.TestCase):
    def test_metric_and_message(self):
        """Итоговая метрика декодера равна минимуму по полному перебору."""
        rng = np.random.default_rng(77)
        for length in (1, 2, 3, 4, 5):
            for _ in range(5):
                info = rng.integers(0, 2, size=length, dtype=np.uint8)
                code = conv.encode(info)
                rx = code.copy()
                n_err = rng.integers(0, 3)
                if n_err:
                    rx[rng.choice(code.size, size=n_err, replace=False)] ^= 1

                decoded, details = viterbi.decode(rx, length=length, return_details=True)
                best_metric, best_msgs = brute_force(rx, length)
                self.assertEqual(int(details["final_metrics"][0]), best_metric,
                                 msg=f"L={length}")
                # сообщение совпадает, если оно единственное оптимальное
                if len(best_msgs) == 1:
                    self.assertEqual(tuple(int(b) for b in decoded), best_msgs[0])


class TestTieBreak(unittest.TestCase):
    def test_deterministic(self):
        rng = np.random.default_rng(123)
        code = conv.encode(rng.integers(0, 2, size=20, dtype=np.uint8))
        first = viterbi.decode(code)
        for _ in range(5):
            np.testing.assert_array_equal(viterbi.decode(code), first)

    def test_edge_order_is_smaller_predecessor(self):
        """Tie-break: при равных метриках берётся меньший номер предыдущего состояния."""
        for q in range(4):
            self.assertLess(int(viterbi.EDGE_PREV[q, 0]), int(viterbi.EDGE_PREV[q, 1]))
            self.assertEqual(int(viterbi.EDGE_PREV[q, 0]), 2 * (q % 2))

    def test_all_zero_received(self):
        """Нулевой принятый поток: выживший путь — нулевое сообщение."""
        decoded = viterbi.decode(np.zeros(2 * 12, dtype=np.uint8), length=10)
        np.testing.assert_array_equal(decoded, np.zeros(10, dtype=np.uint8))


class TestStates(unittest.TestCase):
    def test_start_and_end_zero(self):
        rng = np.random.default_rng(9)
        info = rng.integers(0, 2, size=30, dtype=np.uint8)
        _, details = viterbi.decode(conv.encode(info), length=30, return_details=True)
        path = details["path_state"]
        self.assertEqual(int(path[0]), 0)
        self.assertEqual(int(path[-1]), 0)
        self.assertEqual(len(path), 30 + conv.TAIL + 1)

    def test_metrics_wide_type(self):
        """Метрики — int32; значение заведомо больше 255 при длинном пакете с шумом."""
        rng = np.random.default_rng(1)
        rx = rng.integers(0, 2, size=2 * 5000, dtype=np.uint8)   # чистый шум
        _, details = viterbi.decode(rx, length=4998, return_details=True)
        self.assertEqual(details["final_metrics"].dtype, np.int32)
        self.assertGreater(int(details["final_metrics"][0]), 255)
        self.assertLess(int(details["final_metrics"][0]), int(viterbi.INF))

    def test_unknown_end_state_mode(self):
        """Резервный режим 'best' работает и возвращает тот же размер."""
        rng = np.random.default_rng(4)
        info = rng.integers(0, 2, size=15, dtype=np.uint8)
        code = conv.encode(info)
        code[-1] ^= 1
        out = viterbi.decode(code, length=15, end_state="best")
        self.assertEqual(out.size, 15)
        out_zero = viterbi.decode(code, length=15, end_state="zero")
        self.assertEqual(out_zero.size, 15)


class TestBatch(unittest.TestCase):
    def test_batch_equals_single(self):
        rng = np.random.default_rng(2024)
        packets = conv.encode_batch(rng.integers(0, 2, size=(16, 200), dtype=np.uint8))
        noisy = packets.copy()
        flip = rng.random(noisy.shape) < 0.02
        noisy[flip] ^= 1
        batch = viterbi.decode_batch(noisy, length=200)
        for i in range(16):
            np.testing.assert_array_equal(batch[i], viterbi.decode(noisy[i], length=200))

    def test_long_packet_no_overflow(self):
        """L=10000: пакетная и одиночная обработка совпадают, переполнения нет."""
        rng = np.random.default_rng(31337)
        info = rng.integers(0, 2, size=10_000, dtype=np.uint8)
        code = conv.encode(info)
        noisy = code.copy()
        noisy[rng.random(noisy.size) < 0.05] ^= 1

        single = viterbi.decode(noisy, length=10_000)
        batched = viterbi.decode_batch(noisy[None, :], length=10_000)[0]
        np.testing.assert_array_equal(single, batched)
        self.assertEqual(single.size, 10_000)

        _, details = viterbi.decode(noisy, length=10_000, return_details=True)
        self.assertLess(int(details["final_metrics"][0]), int(viterbi.INF))

    def test_noise_free_long(self):
        rng = np.random.default_rng(888)
        info = rng.integers(0, 2, size=10_000, dtype=np.uint8)
        np.testing.assert_array_equal(viterbi.decode(conv.encode(info)), info)

    def test_batch_different_packets_same_result(self):
        rng = np.random.default_rng(17)
        packets = rng.integers(0, 2, size=(4, 50), dtype=np.uint8)
        codes = conv.encode_batch(packets)
        decoded = viterbi.decode_batch(codes, length=50)
        np.testing.assert_array_equal(decoded, packets)


class TestValidation(unittest.TestCase):
    def test_odd_length_rejected(self):
        with self.assertRaises(ValueError):
            viterbi.decode(np.zeros(7, dtype=np.uint8))
        with self.assertRaises(ValueError):
            viterbi.decode_batch(np.zeros((2, 7), dtype=np.uint8))

    def test_non_binary_rejected(self):
        with self.assertRaises(ValueError):
            viterbi.decode(np.array([0, 1, 2, 0, 1, 0], dtype=np.uint8))

    def test_inconsistent_length(self):
        with self.assertRaises(ValueError):
            viterbi.decode(np.zeros(12, dtype=np.uint8), length=99)

    def test_bad_end_state(self):
        with self.assertRaises(ValueError):
            viterbi.decode(np.zeros(8, dtype=np.uint8), length=2, end_state="middle")

    def test_wrong_ndim(self):
        with self.assertRaises(ValueError):
            viterbi.decode(np.zeros((2, 8), dtype=np.uint8))


if __name__ == "__main__":
    unittest.main(verbosity=2)
