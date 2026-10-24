"""Свёрточный кодер R = 1/2, K = 3, порождающие полиномы (7,5)₈.

Математическая спецификация: Агент №2, «Пакет 2 Агенту №1», модуль
«свёрточный кодер».

Модель (используются именно уравнения, а не словесное описание сдвига)
----------------------------------------------------------------------
Перед шагом ``t`` состояние ``S_t = (a, b) = (u_{t-1}, u_{t-2})``,
индекс состояния ``s = 2a + b``::

    00 -> 0,   01 -> 1,   10 -> 2,   11 -> 3

Для текущего входного бита ``u``::

    v1 = u ⊕ a ⊕ b          (g1(D) = 1 + D + D²,  7₈)
    v2 = u ⊕ b              (g2(D) = 1 + D²,      5₈)
    S_{t+1} = (u, a)

Выходной поток: ``v1`` первого шага, ``v2`` первого шага, ``v1`` второго
шага, ``v2`` второго шага, и так далее (пара на каждый шаг).

Таблица переходов (8 рёбер)
---------------------------
===  ===  ======  =====
S_t  u    v1v2    S_t+1
===  ===  ======  =====
00   0    00      00
00   1    11      10
01   0    11      00
01   1    00      10
10   0    10      01
10   1    01      11
11   0    01      01
11   1    10      11
===  ===  ======  =====

Завершение и скорость (Q4)
--------------------------
Каждый пакет независимо начинается из состояния 00. К ``L`` информационным
битам добавляются ``K - 1 = 2`` нулевых бита, поэтому после двух хвостовых
шагов состояние снова 00 при любом информационном пакете.

    T   = L + 2          число шагов решётки
    N   = 2(L + 2)       число переданных кодовых бит
    R_eff = L / (2(L + 2))

Непрерывный режим (без хвоста) в основной эксперимент не включается.

Свободное расстояние
--------------------
d_free = 5: минимальный выходной вес ненулевого пути из 00 в 00
(путь по входу 100: 00 → 10 → 01 → 00, выход 11|10|11, вес 5).
Величина d_free сама по себе не определяет BER на заданном уровне.
"""

from __future__ import annotations

import numpy as np

__all__ = [
    "K", "N_OUT", "TAIL", "N_STATES", "L_DEFAULT",
    "NEXT_TABLE", "OUT_TABLE", "TRANSITIONS",
    "rate_effective", "energy_penalty_db",
    "encode", "encode_batch", "finish_state", "state_sequence",
]

K = 3                     # длина кодового ограничения
N_OUT = 2                 # число выходных битов на шаг (R = 1/2)
TAIL = K - 1              # 2 хвостовых нуля
N_STATES = 1 << (K - 1)   # 4 состояния
L_DEFAULT = 10_000        # длина информационного пакета в основном эксперименте
D_FREE = 5                # справочно; подтверждается тестом полным перебором

# Таблица переходов из спецификации. Строки — состояние S_t (0..3),
# столбцы — вход u (0, 1).
NEXT_TABLE = np.array(
    [
        [0, 2],   # S=00: u=0 -> 00, u=1 -> 10
        [0, 2],   # S=01: u=0 -> 00, u=1 -> 10
        [1, 3],   # S=10: u=0 -> 01, u=1 -> 11
        [1, 3],   # S=11: u=0 -> 01, u=1 -> 11
    ],
    dtype=np.int8,
)

# Выход v1v2 как целое: v1*2 + v2.
OUT_TABLE = np.array(
    [
        [0b00, 0b11],   # S=00: u=0 -> 00, u=1 -> 11
        [0b11, 0b00],   # S=01: u=0 -> 11, u=1 -> 00
        [0b10, 0b01],   # S=10: u=0 -> 10, u=1 -> 01
        [0b01, 0b10],   # S=11: u=0 -> 01, u=1 -> 10
    ],
    dtype=np.int8,
)

# Человекочитаемое представление таблицы (для отчёта и диаграммы состояний)
TRANSITIONS = tuple(
    (s, u, f"{OUT_TABLE[s, u] >> 1}{OUT_TABLE[s, u] & 1}", int(NEXT_TABLE[s, u]))
    for s in range(N_STATES)
    for u in (0, 1)
)


# --------------------------------------------------------------------------
# Скорость
# --------------------------------------------------------------------------
def rate_effective(length: int = L_DEFAULT) -> float:
    """Фактическая скорость кода с учётом хвостовых битов: L / (2(L+2))."""
    if length < 1:
        raise ValueError("длина пакета должна быть не меньше 1")
    return length / (2.0 * (length + TAIL))


def energy_penalty_db(length: int = L_DEFAULT) -> float:
    """Энергетическая надбавка за хвост: 10·lg((L+2)/L) дБ."""
    return 10.0 * np.log10((length + TAIL) / length)


# --------------------------------------------------------------------------
# Кодирование
# --------------------------------------------------------------------------
def encode(bits) -> np.ndarray:
    """Закодировать один информационный пакет с нулевым завершением решётки.

    Parameters
    ----------
    bits : array_like
        Двоичная последовательность длины L ≥ 1.

    Returns
    -------
    numpy.ndarray
        Кодовые биты ``uint8`` длины ``2(L + 2)``; конечное состояние 00.
    """
    info = np.asarray(bits)
    if info.ndim != 1:
        raise ValueError(f"encode: ожидается одномерная последовательность, ndim={info.ndim}")
    if info.size < 1:
        raise ValueError("encode: пустой пакет")
    if not np.all((info == 0) | (info == 1)):
        raise ValueError("encode: содержит значения, отличные от 0 и 1")
    return encode_batch(info.astype(np.uint8)[None, :])[0]


def encode_batch(bits2d) -> np.ndarray:
    """Закодировать B независимых пакетов одинаковой длины.

    Каждый пакет начинается из состояния 00 и не наследует состояние
    предыдущего пакета.

    Parameters
    ----------
    bits2d : array_like
        Массив формы ``(B, L)``.

    Returns
    -------
    numpy.ndarray
        Массив формы ``(B, 2(L + 2))``.
    """
    info = np.asarray(bits2d)
    if info.ndim != 2:
        raise ValueError(f"encode_batch: ожидается массив (B, L), ndim={info.ndim}")
    b_count, length = info.shape
    if length < 1:
        raise ValueError("encode_batch: пустой пакет")
    if not np.all((info == 0) | (info == 1)):
        raise ValueError("encode_batch: содержит значения, отличные от 0 и 1")

    # нулевое завершение: K-1 нулей в конце каждого пакета
    padded = np.zeros((b_count, length + TAIL), dtype=np.int8)
    padded[:, :length] = info
    steps = padded.shape[1]

    states = np.zeros(b_count, dtype=np.int8)          # начальное состояние 00
    out = np.empty((b_count, steps * N_OUT), dtype=np.uint8)
    rows = np.arange(b_count)

    for t in range(steps):
        u = padded[:, t]
        v = OUT_TABLE[states, u]                        # 2-битный символ 0..3
        out[:, 2 * t] = v >> 1                          # сначала v1
        out[:, 2 * t + 1] = v & 1                       # затем v2
        states = NEXT_TABLE[states, u]                  # S <- (u, a)

    if np.any(states != 0):                             # гарантия zero-tail
        raise AssertionError("внутренняя ошибка: конечное состояние не 00")
    return out


def finish_state(bits) -> int:
    """Конечное состояние кодера после кодирования пакета (с хвостом) — для тестов."""
    info = np.asarray(bits, dtype=np.int8)
    state = 0
    for u in list(info) + [0] * TAIL:
        state = int(NEXT_TABLE[state, int(u)])
    return state


def state_sequence(bits) -> np.ndarray:
    """Последовательность состояний S_0..S_T (для диаграммы решётки в отчёте)."""
    info = np.asarray(bits, dtype=np.int8)
    state = 0
    seq = [0]
    for u in list(info) + [0] * TAIL:
        state = int(NEXT_TABLE[state, int(u)])
        seq.append(state)
    return np.array(seq, dtype=np.int8)
