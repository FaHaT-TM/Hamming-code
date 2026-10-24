"""Тесты кода Хэмминга (7,4).

Проверяются все 10 критериев корректности из спецификации,
включая 112 случаев одиночной ошибки и явную фиксацию ограничения
по двойным ошибкам (оно НЕ является дефектом реализации).
"""

import os
import sys
import unittest

import numpy as np

# корень проекта в sys.path: тест запускается и из корня, и из папки tests/
_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

from src import hamming  # noqa: E402


def bits(s: str) -> np.ndarray:
    """Строка '0110011' -> uint8-массив."""
    return np.fromiter((int(ch) for ch in s), dtype=np.uint8)


def as_str(arr) -> str:
    return "".join(str(int(b)) for b in np.ravel(arr))


class TestParameters(unittest.TestCase):
    def test_parameters(self):
        self.assertEqual(hamming.N, 7)
        self.assertEqual(hamming.K, 4)
        self.assertAlmostEqual(hamming.RATE, 4 / 7, places=15)
        self.assertEqual(hamming.D_MIN, 3)
        self.assertEqual(hamming.T_CORRECT, 1)
        self.assertEqual(hamming.PARITY_POSITIONS, (0, 1, 3))
        self.assertEqual(hamming.INFO_POSITIONS, (2, 4, 5, 6))


class TestMatrices(unittest.TestCase):
    def test_G_shape_and_systematic_form(self):
        self.assertEqual(hamming.G.shape, (4, 7))
        # информационные столбцы 2,4,5,6 образуют I4
        info_cols = hamming.G[:, list(hamming.INFO_POSITIONS)]
        np.testing.assert_array_equal(info_cols, np.eye(4, dtype=np.int8))

    def test_H_shape(self):
        self.assertEqual(hamming.H.shape, (3, 7))

    def test_orthogonality_G_Ht(self):
        """Критерий 1: G H^T = 0 над GF(2)."""
        prod = (hamming.G.astype(np.int64) @ hamming.H.T.astype(np.int64)) % 2
        np.testing.assert_array_equal(prod, np.zeros((4, 3), dtype=np.int64))

    def test_H_columns_are_position_numbers(self):
        """Столбец j содержит номер j от младшего разряда к старшему."""
        expected = np.array(
            [[int(b) for b in format(j, "03b")[::-1]] for j in range(1, 8)], dtype=np.int8
        ).T
        np.testing.assert_array_equal(hamming.H, expected)


class TestEncodeKnownVectors(unittest.TestCase):
    """Тестовые примеры из спецификации, раздел «Тестовые примеры»."""

    KNOWN = [
        ("0000", "0000000"),
        ("1000", "1110000"),
        ("0100", "1001100"),
        ("0010", "0101010"),
        ("0001", "1101001"),
        ("1011", "0110011"),
        ("1111", "1111111"),
    ]

    def test_known_vectors(self):
        for info, code in self.KNOWN:
            with self.subTest(info=info):
                self.assertEqual(as_str(hamming.encode(bits(info))), code)

    def test_encode_matches_generator_matrix(self):
        """Критерий 7: кодирование через p1,p2,p4 совпадает с uG mod 2."""
        rng = np.random.default_rng(2024)
        info = rng.integers(0, 2, size=4 * 500, dtype=np.uint8)
        expected = (info.reshape(-1, 4).astype(np.int64)
                    @ hamming.G.astype(np.int64)) % 2
        np.testing.assert_array_equal(hamming.encode(info).reshape(-1, 7), expected)


class TestNoError(unittest.TestCase):
    def test_all_16_word_encoding_and_syndrome(self):
        """Критерии 2, 4, 5: все 16 слов декодируются без ошибок, синдром 0, слова различны."""
        info = np.array(
            [[int(b) for b in format(u, "04b")] for u in range(16)], dtype=np.uint8
        ).reshape(-1)
        code = hamming.encode(info)

        # критерий 4: нулевой синдром
        np.testing.assert_array_equal(hamming.syndrome(code), np.zeros((16, 3), dtype=np.uint8))
        # критерий 2: декодирование без ошибок
        np.testing.assert_array_equal(hamming.decode(code), info)
        # критерий 5: кодовые слова различны
        words = code.reshape(-1, 7)
        self.assertEqual(len({as_str(w) for w in words}), 16)

    def test_double_correction_not_applied(self):
        """Декодер не «исправляет» дважды: одна коррекция на слово, длина сохраняется."""
        code = hamming.encode(bits("1011"))
        rx = code.copy()
        rx[0] ^= 1
        out = hamming.decode_with_diagnostics(rx)
        self.assertEqual(out["n_corrections"], 1)
        self.assertEqual(len(out["corrected_words"]), 7)


class TestSingleErrorCorrection(unittest.TestCase):
    """Критерий 3: 16 слов x 7 позиций = 112 случаев восстановления информации."""

    def test_112_cases(self):
        info_seq = np.array(
            [[int(b) for b in format(u, "04b")] for u in range(16)], dtype=np.uint8
        ).reshape(-1)
        code = hamming.encode(info_seq).reshape(16, 7)
        checked = 0
        for i in range(16):
            for pos in range(7):
                rx = code[i].copy()
                rx[pos] ^= 1
                decoded = hamming.decode(rx)
                self.assertEqual(
                    as_str(decoded), as_str(info_seq[i * 4:(i + 1) * 4]),
                    msg=f"слово {i}, ошибка в позиции {pos}",
                )
                checked += 1
        self.assertEqual(checked, 112)

    def test_specification_table(self):
        """Таблица синдромов из спецификации для слова 1011 / 0110011."""
        code = hamming.encode(bits("1011"))
        self.assertEqual(as_str(code), "0110011")
        table = {
            None: ("0110011", (0, 0, 0), 0),
            1: ("1110011", (1, 0, 0), 1),
            2: ("0010011", (0, 1, 0), 2),
            3: ("0100011", (1, 1, 0), 3),
            4: ("0111011", (0, 0, 1), 4),
            5: ("0110111", (1, 0, 1), 5),
            6: ("0110001", (0, 1, 1), 6),
            7: ("0110010", (1, 1, 1), 7),
        }
        for pos, (rx_str, s_exp, j_exp) in table.items():
            with self.subTest(pos=pos):
                rx = bits(rx_str)
                s = hamming.syndrome(rx)[0]
                self.assertEqual(tuple(int(v) for v in s), s_exp)
                self.assertEqual(int(hamming.syndrome_index(rx)[0]), j_exp)
                self.assertEqual(as_str(hamming.decode(rx)), "1011")


class TestInformationExtraction(unittest.TestCase):
    def test_info_positions_order(self):
        """Информация извлекается именно из позиций 3,5,6,7 в порядке d1,d2,d3,d4."""
        code = hamming.encode(bits("1000"))
        self.assertEqual([int(code[p]) for p in hamming.INFO_POSITIONS], [1, 0, 0, 0])
        code = hamming.encode(bits("0001"))
        self.assertEqual([int(code[p]) for p in hamming.INFO_POSITIONS], [0, 0, 0, 1])


class TestDistances(unittest.TestCase):
    def test_min_distance(self):
        """Критерий 6: минимальное попарное расстояние равно 3."""
        self.assertEqual(hamming.min_distance(), 3)

    def test_min_weight_equals_min_distance(self):
        cw = hamming.codewords_all()
        weights = cw.sum(axis=1)
        self.assertEqual(int(weights[1:].min()), 3)


class TestTwoErrorsLimitation(unittest.TestCase):
    """Критерий 10 / раздел спецификации: двойные ошибки НЕ исправляются."""

    def test_two_errors_produce_wrong_information(self):
        code = hamming.encode(bits("1011"))
        rx = code.copy()
        rx[0] ^= 1  # позиция 1
        rx[1] ^= 1  # позиция 2
        self.assertEqual(as_str(rx), "1010011")

        diag = hamming.decode_with_diagnostics(rx)
        self.assertEqual(int(diag["error_position"][0]), 3)          # ложная позиция
        self.assertEqual(as_str(diag["corrected_words"]), "1000011")
        self.assertEqual(as_str(diag["bits"]), "0011")               # не равно 1011
        # после «коррекции» синдром нулевой — проверка синдрома не выявляет ошибку
        np.testing.assert_array_equal(hamming.syndrome(diag["corrected_words"]),
                                      np.zeros((1, 3), dtype=np.uint8))

    def test_two_errors_not_always_detected_as_two(self):
        """Не утверждаем обнаружение двойных ошибок: часть слов даёт t=1 коррекцию."""
        rng = np.random.default_rng(7)
        n = 2000
        info = rng.integers(0, 2, size=4 * n, dtype=np.uint8)
        code = hamming.encode(info).reshape(n, 7).copy()
        for row in code:
            p = rng.choice(7, size=2, replace=False)
            row[p] ^= 1
        decoded = hamming.decode(code.reshape(-1))
        self.assertGreater(int(np.count_nonzero(decoded != info)), 0)


class TestLongSequences(unittest.TestCase):
    def test_block_order_preserved(self):
        """Критерий 8: на длинной последовательности порядок блоков сохраняется."""
        rng = np.random.default_rng(12345)
        info = rng.integers(0, 2, size=4 * 10_000, dtype=np.uint8)
        code = hamming.encode(info)
        self.assertEqual(code.size, 7 * 10_000)
        np.testing.assert_array_equal(hamming.decode(code), info)

    def test_lengths(self):
        """Выходные длины: 4 -> 7, кратность сохраняется."""
        for n_blocks in (1, 2, 3, 17, 1000):
            info = np.zeros(4 * n_blocks, dtype=np.uint8)
            self.assertEqual(hamming.encode(info).size, 7 * n_blocks)
            self.assertEqual(hamming.decode(np.zeros(7 * n_blocks, dtype=np.uint8)).size,
                             4 * n_blocks)

    def test_multiple_blocks_with_errors(self):
        """Несколько блоков, в каждом по одной ошибке в случайной позиции."""
        rng = np.random.default_rng(99)
        n = 500
        info = rng.integers(0, 2, size=4 * n, dtype=np.uint8)
        code = hamming.encode(info).reshape(n, 7).copy()
        positions = rng.integers(0, 7, size=n)
        for row, pos in zip(code, positions):
            row[pos] ^= 1
        np.testing.assert_array_equal(hamming.decode(code.reshape(-1)), info)


class TestInputValidation(unittest.TestCase):
    """Критерий 9: проверка двоичности и кратности длины."""

    def test_non_multiple_of_4_encode(self):
        with self.assertRaises(ValueError):
            hamming.encode([0, 1, 1])

    def test_non_multiple_of_7_decode(self):
        with self.assertRaises(ValueError):
            hamming.decode([0, 1, 0, 1])

    def test_non_binary(self):
        with self.assertRaises(ValueError):
            hamming.encode([0, 1, 2, 0])
        with self.assertRaises(ValueError):
            hamming.decode([0, 1, 0, 1, 0, 1, 5])

    def test_empty(self):
        with self.assertRaises(ValueError):
            hamming.encode([])

    def test_two_dimensional_rejected(self):
        with self.assertRaises(ValueError):
            hamming.encode([[0, 1, 0, 1]])

    def test_no_silent_truncation(self):
        """Некратная длина не должна усекаться: результат — исключение, а не 7 бит."""
        try:
            out = hamming.encode([0, 0, 0, 0, 1])
        except ValueError:
            return
        self.fail(f"ожидалось ValueError, получено кодирование длины {len(out)}")


if __name__ == "__main__":
    unittest.main(verbosity=2)
