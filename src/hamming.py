"""Код Хэмминга (7,4) — кодирование и синдромное декодирование.

Математическая спецификация: docs/SPECIFICATIONS.md (раздел Q1–Q2).
Декодирование — ограниченное расстоянием (bounded-distance), t = 1.

Параметры
---------
n = 7, k = 4, R = 4/7, d_min = 3, t = 1.

Порядок битов кодового слова (позиции с единицы, как в спецификации)::

    позиция:  1    2    3    4    5    6    7
    бит:      p1   p2   d1   p4   d2   d3   d4

Проверочные позиции: 1, 2, 4 (индексы с нуля: 0, 1, 3).
Информационные позиции: 3, 5, 6, 7 (индексы с нуля: 2, 4, 5, 6).

Проверки на чётность (чётность чётная, все операции над GF(2), т.е. XOR)::

    c1 ⊕ c3 ⊕ c5 ⊕ c7 = 0
    c2 ⊕ c3 ⊕ c6 ⊕ c7 = 0
    c4 ⊕ c5 ⊕ c6 ⊕ c7 = 0

Проверочная матрица H (3x7)::

    H = [1 0 1 0 1 0 1
         0 1 1 0 0 1 1
         0 0 0 1 1 1 1]

Столбец j содержит двоичные разряды номера j от младшего к старшему,
поэтому синдром s = (s1, s2, s4) для одиночной ошибки даёт прямо
номер позиции: j = s1 + 2*s2 + 4*s4.

Порождающая матрица G (4x7), c = u G mod 2::

    G = [1 1 1 0 0 0 0
         1 0 0 1 1 0 0
         0 1 0 1 0 1 0
         1 1 0 1 0 0 1]

Ограничения кода
----------------
Гарантированно исправляется ровно одна ошибка на слово.
При двух ошибках синдром ненулевой, но указывает на «ложную» позицию:
декодер выполнит ошибочную коррекцию, и результат будет неверным
(см. спецификацию, раздел «Ограничение: две ошибки»).
Одновременное «исправить одну и обнаружить две» для этой схемы
не заявляется и не поддерживается.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "N", "K", "RATE", "D_MIN", "T_CORRECT",
    "PARITY_POSITIONS", "INFO_POSITIONS",
    "G", "H",
    "encode", "syndrome", "syndrome_index", "correct", "decode",
    "decode_with_diagnostics", "codewords_all", "min_distance",
    "validate_bits",
]

# --------------------------------------------------------------------------
# Параметры кода
# --------------------------------------------------------------------------
N = 7                 # длина кодового слова
K = 4                 # длина информационного блока
RATE = K / N          # 4/7 — скорость кода
D_MIN = 3             # минимальное кодовое расстояние
T_CORRECT = (D_MIN - 1) // 2   # = 1 гарантированно исправляемая ошибка

# Индексы с нуля: проверочные (p1, p2, p4) и информационные (d1, d2, d3, d4)
PARITY_POSITIONS: tuple[int, ...] = (0, 1, 3)
INFO_POSITIONS: tuple[int, ...] = (2, 4, 5, 6)

# --------------------------------------------------------------------------
# Матрицы (int8, значения 0/1)
# --------------------------------------------------------------------------
G: np.ndarray = np.array(
    [
        [1, 1, 1, 0, 0, 0, 0],
        [1, 0, 0, 1, 1, 0, 0],
        [0, 1, 0, 1, 0, 1, 0],
        [1, 1, 0, 1, 0, 0, 1],
    ],
    dtype=np.int8,
)

H: np.ndarray = np.array(
    [
        [1, 0, 1, 0, 1, 0, 1],
        [0, 1, 1, 0, 0, 1, 1],
        [0, 0, 0, 1, 1, 1, 1],
    ],
    dtype=np.int8,
)


# --------------------------------------------------------------------------
# Валидация входа
# --------------------------------------------------------------------------
def validate_bits(bits, length_multiple: int | None = None, name: str = "bits") -> np.ndarray:
    """Проверить, что данные — двоичная последовательность (0/1).

    Parameters
    ----------
    bits : array_like
        Последовательность битов (список, кортеж, numpy-массив).
    length_multiple : int, optional
        Если задано, длина должна быть кратна этому числу и не равна нулю.
    name : str
        Имя аргумента для текста ошибки.

    Returns
    -------
    numpy.ndarray
        Одномерный массив ``uint8`` со значениями 0/1.

    Raises
    ------
    ValueError
        Если данные не двоичные, пустые либо длина не кратна ``length_multiple``.
        Некратные длины **не усекаются** неявно (требование спецификации).
    """
    arr = np.asarray(bits)
    if arr.ndim != 1:
        raise ValueError(f"{name}: ожидается одномерная последовательность, получено ndim={arr.ndim}")
    if arr.size == 0:
        raise ValueError(f"{name}: пустая последовательность")
    if not np.all((arr == 0) | (arr == 1)):
        raise ValueError(f"{name}: содержит значения, отличные от 0 и 1")
    if length_multiple is not None and arr.size % length_multiple != 0:
        raise ValueError(
            f"{name}: длина {arr.size} не кратна {length_multiple}. "
            "Неявное усечение или дополнение не выполняется."
        )
    return arr.astype(np.uint8, copy=False)


# --------------------------------------------------------------------------
# Кодирование
# --------------------------------------------------------------------------
def encode(bits) -> np.ndarray:
    """Закодировать информационные биты кодом Хэмминга (7,4).

    Parameters
    ----------
    bits : array_like
        Двоичная последовательность, длина кратна 4 (без дополнения).

    Returns
    -------
    numpy.ndarray
        Двоичный массив ``uint8`` длины ``7 * len(bits) / 4``.
        Порядок блоков сохраняется; биты каждого блока — ``(p1, p2, d1, p4, d2, d3, d4)``.
    """
    info = validate_bits(bits, length_multiple=K, name="bits")
    u = info.reshape(-1, K)

    # Проверочные биты по формулам спецификации:
    #   p1 = d1 ^ d2 ^ d4 ;  p2 = d1 ^ d3 ^ d4 ;  p4 = d2 ^ d3 ^ d4
    p1 = u[:, 0] ^ u[:, 1] ^ u[:, 3]
    p2 = u[:, 0] ^ u[:, 2] ^ u[:, 3]
    p4 = u[:, 1] ^ u[:, 2] ^ u[:, 3]

    code = np.empty((u.shape[0], N), dtype=np.uint8)
    code[:, 0] = p1
    code[:, 1] = p2
    code[:, 2] = u[:, 0]   # d1
    code[:, 3] = p4
    code[:, 4] = u[:, 1]   # d2
    code[:, 5] = u[:, 2]   # d3
    code[:, 6] = u[:, 3]   # d4
    return code.reshape(-1)


# --------------------------------------------------------------------------
# Синдром
# --------------------------------------------------------------------------
def syndrome(codewords) -> np.ndarray:
    """Вычислить синдром s = r H^T mod 2 для каждого кодового слова.

    Parameters
    ----------
    codewords : array_like
        Двоичная последовательность, длина кратна 7.

    Returns
    -------
    numpy.ndarray
        Массив формы ``(N_blocks, 3)``: столбцы — (s1, s2, s4).
    """
    rx = validate_bits(codewords, length_multiple=N, name="codewords").reshape(-1, N)
    s1 = rx[:, 0] ^ rx[:, 2] ^ rx[:, 4] ^ rx[:, 6]
    s2 = rx[:, 1] ^ rx[:, 2] ^ rx[:, 5] ^ rx[:, 6]
    s4 = rx[:, 3] ^ rx[:, 4] ^ rx[:, 5] ^ rx[:, 6]
    return np.stack([s1, s2, s4], axis=1).astype(np.uint8)


def syndrome_index(codewords) -> np.ndarray:
    """Номер позиции одиночной ошибки j = s1 + 2*s2 + 4*s4 (0 = ошибки не обнаружено).

    Для одиночной ошибки j совпадает с номером позиции (1..7);
    коррекция выполняется по индексу ``j - 1``.
    """
    s = syndrome(codewords)
    return (s[:, 0].astype(np.int64)
            + 2 * s[:, 1].astype(np.int64)
            + 4 * s[:, 2].astype(np.int64))


# --------------------------------------------------------------------------
# Коррекция и декодирование
# --------------------------------------------------------------------------
def correct(codewords) -> np.ndarray:
    """Исправить одиночную ошибку в каждом кодовом слове (bounded-distance).

    Инвертируется бит в позиции ``j`` (индекс ``j - 1``), если ``j != 0``.

    Returns
    -------
    numpy.ndarray
        Исправленные кодовые слова, длина та же, что у входа.
    """
    rx = validate_bits(codewords, length_multiple=N, name="codewords").reshape(-1, N).copy()
    j = syndrome_index(rx.reshape(-1))
    err = j > 0
    if np.any(err):
        rows = np.nonzero(err)[0]
        rx[rows, j[rows] - 1] ^= 1
    return rx.reshape(-1)


def decode(codewords) -> np.ndarray:
    """Декодировать принятые кодовые слова: коррекция + извлечение d1, d2, d3, d4.

    Parameters
    ----------
    codewords : array_like
        Двоичная последовательность после жёсткого решения, длина кратна 7.

    Returns
    -------
    numpy.ndarray
        Информационные биты, длина ``4 * len(codewords) / 7``.
    """
    fixed = correct(codewords).reshape(-1, N)
    return fixed[:, INFO_POSITIONS].reshape(-1)


def decode_with_diagnostics(codewords) -> dict:
    """Декодирование с диагностикой — для тестов и иллюстраций, не для счётчика BER.

    Returns
    -------
    dict
        ``bits``            : извлечённые информационные биты;
        ``syndrome``        : синдромы до коррекции, форма (N_blocks, 3);
        ``error_position``  : j (1..7) или 0, если синдром нулевой;
        ``corrected_words`` : исправленные кодовые слова;
        ``n_corrections``   : число блоков с ненулевым синдромом.

    Notes
    -----
    ``n_corrections`` — **не** число реальных ошибок канала: при двух ошибках
    в блоке коррекция выполняется, но результат неверен, и счётчик это
    не отражает. Для BER используются только ``bits`` в сравнении с исходными.
    """
    rx = validate_bits(codewords, length_multiple=N, name="codewords").reshape(-1, N)
    s = syndrome(rx.reshape(-1))
    j = syndrome_index(rx.reshape(-1))
    corrected = correct(rx.reshape(-1)).reshape(-1, N)
    return {
        "bits": corrected[:, INFO_POSITIONS].reshape(-1),
        "syndrome": s,
        "error_position": j,
        "corrected_words": corrected.reshape(-1),
        "n_corrections": int(np.count_nonzero(j)),
    }


# --------------------------------------------------------------------------
# Справочные функции (используются тестами и для отчёта)
# --------------------------------------------------------------------------
def codewords_all() -> np.ndarray:
    """Все 16 кодовых слов, форма (16, 7), в порядке информационных слов 0000..1111."""
    u = ((np.arange(16)[:, None] >> np.arange(3, -1, -1)[None, :]) & 1).astype(np.uint8)
    return encode(u.reshape(-1)).reshape(-1, N)


def min_distance() -> int:
    """Минимальное попарное расстояние Хэмминга между кодовыми словами (ожидается 3)."""
    cw = codewords_all()
    diff = cw[:, None, :] ^ cw[None, :, :]
    weights = diff.sum(axis=2)
    weights[np.arange(cw.shape[0]), np.arange(cw.shape[0])] = 99  # исключить диагональ
    return int(weights.min())
