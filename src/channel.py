"""Канал BPSK + AWGN с жёстким решением.

Математическая спецификация: docs/SPECIFICATIONS_CONVOLUTIONAL.md, модуль
«BPSK и AWGN — Q6».

Модель
------
::

    x   = 1 - 2c          (0 -> +1, 1 -> -1);   Es = 1
    y   = x + n,          n ~ N(0, sigma^2), независимые
    sigma^2 = 1 / (2 * R_eff * gamma_b)
    gamma_b = 10^(Eb/N0_дБ / 10)
    жёсткое решение: 1 при y < 0, иначе 0

``Eb`` — **полная энергия переданных символов, делённая на число исходных
информационных битов**. Это не энергия кодового бита и не энергия символа
до кодера: хвостовые расходы входят в ``R_eff``. Поэтому при одинаковом
Eb/N0 разные скорости требуют разной ``sigma^2`` при ``Es = 1``.

Контроль (используется тестом канала)
-------------------------------------
Вероятность ошибки передаваемого бита после жёсткого решения::

    p = Q( sqrt(2 * R_eff * gamma_b) )

Для передачи без кодирования (R_eff = 1) это совпадает с итоговой BER.

Потоки случайности
------------------
Главный seed фиксируется один раз; независимые потоки создаются через
``numpy.random.SeedSequence(seed).spawn(...)`` по схеме
(вариант, точка Eb/N0, повтор). Генератор никогда не переинициализируется
одним и тем же seed на разных пакетах.
"""

from __future__ import annotations

import math

import numpy as np

__all__ = [
    "bpsk_modulate", "bpsk_demodulate_hard", "sigma_from_ebno", "variance_from_ebno",
    "add_awgn", "transmit_hard", "bit_error_probability",
    "qfunc", "make_rngs", "rng_for",
]

_EPS = 1e-300


# --------------------------------------------------------------------------
# Q-функция (без SciPy)
# --------------------------------------------------------------------------
_erfc_vec = np.frompyfunc(math.erfc, 1, 1)


def qfunc(x):
    """Q(x) = 0.5·erfc(x/√2) — только стандартная библиотека (без SciPy).

    Принимает число или массив чисел.
    """
    arr = np.asarray(x, dtype=float)
    if arr.ndim == 0:
        return 0.5 * math.erfc(float(arr) / math.sqrt(2.0))
    return 0.5 * np.asarray(_erfc_vec(arr / math.sqrt(2.0)), dtype=float)


def bit_error_probability(ebno_db: float, rate: float = 1.0) -> float:
    """Вероятность ошибки передаваемого бита после жёсткого решения: Q(√(2·R·γ_b))."""
    gamma_b = 10.0 ** (ebno_db / 10.0)
    return float(qfunc(math.sqrt(2.0 * rate * gamma_b)))


# --------------------------------------------------------------------------
# Модуляция / демодуляция / канал
# --------------------------------------------------------------------------
def bpsk_modulate(bits) -> np.ndarray:
    """BPSK: 0 → +1, 1 → −1 (Es = 1)."""
    b = np.asarray(bits)
    if not np.all((b == 0) | (b == 1)):
        raise ValueError("bpsk_modulate: содержит значения, отличные от 0 и 1")
    return (1.0 - 2.0 * b.astype(np.float64))


def bpsk_demodulate_hard(y) -> np.ndarray:
    """Жёсткое решение: 1 при y < 0, иначе 0."""
    y = np.asarray(y, dtype=np.float64)
    return (y < 0).astype(np.uint8)


def variance_from_ebno(ebno_db: float, rate: float = 1.0) -> float:
    """Дисперсия шума σ² = 1 / (2 · R_eff · γ_b) при Es = 1."""
    if not (0.0 < rate <= 1.0):
        raise ValueError(f"rate должна быть в (0, 1], получено {rate}")
    gamma_b = 10.0 ** (ebno_db / 10.0)
    return 1.0 / (2.0 * rate * gamma_b)


def sigma_from_ebno(ebno_db: float, rate: float = 1.0) -> float:
    """Среднеквадратичное отклонение шума σ = sqrt(σ²)."""
    return math.sqrt(variance_from_ebno(ebno_db, rate))


def add_awgn(x, sigma: float, rng: np.random.Generator) -> np.ndarray:
    """y = x + n, n ~ N(0, σ²) — независимый шум на каждый символ.

    ``sigma`` — среднеквадратичное отклонение (не дисперсия); дисперсию
    для заданного Eb/N0 даёт :func:`variance_from_ebno`.
    """
    if sigma <= 0:
        raise ValueError("sigma должна быть положительной")
    x = np.asarray(x, dtype=np.float64)
    return x + rng.normal(0.0, sigma, size=x.shape)


def transmit_hard(bits, ebno_db: float, rate: float = 1.0,
                  rng: np.random.Generator | None = None) -> np.ndarray:
    """Полная передача потока битов: модуляция → AWGN → жёсткая демодуляция.

    Один вызов = один «пакет» с независимым шумом на каждый символ.
    Генератор ``rng`` передаётся явно и не переинициализируется внутри.
    """
    if rng is None:
        raise ValueError("transmit_hard: требуется явный rng (воспроизводимость)")
    x = bpsk_modulate(bits)
    y = add_awgn(x, sigma_from_ebno(ebno_db, rate), rng)
    return bpsk_demodulate_hard(y)


# --------------------------------------------------------------------------
# Управление потоками случайности
# --------------------------------------------------------------------------
def make_rngs(seed: int, n_streams: int) -> list[np.random.Generator]:
    """Создать ``n_streams`` независимых генераторов из одного главного seed."""
    children = np.random.SeedSequence(seed).spawn(n_streams)
    return [np.random.default_rng(child) for child in children]


def rng_for(seed: int, *tags: int) -> np.random.Generator:
    """Детерминированный генератор для помеченного потока (вариант, точка, повтор).

    Одинаковый набор тегов всегда даёт одинаковый поток, независимо от того,
    сколько потоков создано в других местах программы.
    """
    entropy = [int(seed)] + [int(t) for t in tags]
    return np.random.default_rng(np.random.SeedSequence(entropy))
