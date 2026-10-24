"""Декодер Витерби для жёстких решений (код R = 1/2, K = 3, (7,5)₈).

Математическая спецификация: Агент №2, «Пакет 2 Агенту №1», модуль
«жёсткий Витерби».

Модель
------
Принято ``T = L + 2`` пар ``z_t = (r_{2t}, r_{2t+1})``.

Метрика ветви для ребра (s, u) с выходом ``v = (v1, v2)``::

    b_t(s, u) = 1[z_{t,1} ≠ v1] + 1[z_{t,2} ≠ v2]      (расстояние Хэмминга)

Инициализация и рекурсия::

    M_0(0) = 0,   M_0(1) = M_0(2) = M_0(3) = +∞
    M_{t+1}(q) = min по всем (s,u) с next(s,u)=q от [ M_t(s) + b_t(s,u) ]

После жёсткого решения канал является BSC с p < 1/2: вероятность принятой
последовательности при данном пути пропорциональна ``p^d (1-p)^(2T-d)``,
поэтому минимизация расстояния эквивалентна максимизации правдоподобия
пути. Это ML по кодовому **пути**, а не побитовый MAP.

Разрешение равенств
-------------------
Из двух кандидатов с одинаковой конечной метрикой выбирается ребро с
**меньшим номером предыдущего состояния**. Правило одинаково во всём
исследовании. Оно не объявляется «не влияющим на BER»: равные метрики
соответствуют равноправным ML-путям, которые могут отличаться числом
ошибочных информационных битов, поэтому выбор меняет отдельные результаты
и потенциально среднюю BER.

Реализация
----------
* Метрики — ``int32`` (максимальная конечная метрика не более ``2T``;
  ``uint8`` не подходит). Переполнение исключено, ``+∞`` моделируется
  большим конечным числом ``INF``.
* Полный traceback для каждого пакета; усечение глубины не применяется.
* Пакетная обработка: ``B`` независимых пакетов одной длины обрабатываются
  одновременно, память ``B × T × 4`` байт (``uint8``).
"""

from __future__ import annotations

import numpy as np

from . import convolutional as conv

__all__ = [
    "INF", "EDGE_PREV", "EDGE_INPUT", "EDGE_OUT", "COST",
    "decode", "decode_batch",
]

INF = np.int32(1 << 24)   # «бесконечная» метрика, заведомо больше 2T

# Рёбра, сгруппированные по конечному состоянию q: ровно два предшественника.
# next(s, u) = 2u + a, где s = 2a + b  =>  u = q // 2, a = q % 2, s ∈ {2a, 2a+1}.
EDGE_PREV = np.empty((conv.N_STATES, 2), dtype=np.int64)
EDGE_INPUT = np.empty((conv.N_STATES, 2), dtype=np.int8)
EDGE_OUT = np.empty((conv.N_STATES, 2), dtype=np.int8)
for _q in range(conv.N_STATES):
    _u = _q // 2
    for _k, _b in enumerate((0, 1)):
        _s = 2 * (_q % 2) + _b          # порядок k: s = 2a (меньший), затем 2a+1
        assert conv.NEXT_TABLE[_s, _u] == _q
        EDGE_PREV[_q, _k] = _s
        EDGE_INPUT[_q, _k] = _u
        EDGE_OUT[_q, _k] = conv.OUT_TABLE[_s, _u]

# COST[z, v] = расстояние Хэмминга между принятой парой z и выходом v (0..3)
COST = np.zeros((4, 4), dtype=np.int32)
for _z in range(4):
    for _v in range(4):
        COST[_z, _v] = bin(_z ^ _v).count("1")


def _pairs(rx2d: np.ndarray) -> np.ndarray:
    """(B, 2T) -> (T, B) значений 0..3 (пара битов как целое)."""
    b_count, n_bits = rx2d.shape
    if n_bits % 2 != 0:
        raise ValueError(f"длина принятого потока должна быть чётной, получено {n_bits}")
    z = (rx2d[:, 0::2].astype(np.int8) << 1) | rx2d[:, 1::2].astype(np.int8)
    return z.T.astype(np.int8)


def decode_batch(rx2d, length: int | None = None, end_state: str = "zero",
                 return_details: bool = False):
    """Декодировать B независимых пакетов жёстким алгоритмом Витерби.

    Parameters
    ----------
    rx2d : array_like
        Принятые кодовые биты после жёсткого решения, форма ``(B, 2T)``.
    length : int, optional
        Длина информационного пакета ``L``. По умолчанию ``L = T - 2``.
    end_state : {'zero', 'best'}
        ``'zero'`` — известное конечное состояние 00 (основной режим);
        ``'best'`` — старт traceback из состояния с минимальной метрикой
        (отдельный тестируемый режим, в сравнении не участвует).
    return_details : bool
        Вернуть также метрики и последовательности состояний (для тестов).

    Returns
    -------
    numpy.ndarray or tuple
        Информационные биты формы ``(B, L)``.
        При ``return_details=True`` — кортеж ``(bits, details)``.
    """
    rx = np.asarray(rx2d)
    if rx.ndim != 2:
        raise ValueError(f"decode_batch: ожидается массив (B, 2T), ndim={rx.ndim}")
    if not np.all((rx == 0) | (rx == 1)):
        raise ValueError("decode_batch: содержит значения, отличные от 0 и 1")
    if end_state not in ("zero", "best"):
        raise ValueError("end_state должно быть 'zero' или 'best'")

    b_count, n_bits = rx.shape
    steps = n_bits // 2
    if length is None:
        length = steps - conv.TAIL
    if length < 0 or length + conv.TAIL != steps:
        raise ValueError(
            f"decode_batch: несовместимые размеры: 2T={n_bits} бит (T={steps}), "
            f"L={length}, хвост={conv.TAIL}. Ожидается T = L + {conv.TAIL}."
        )

    z = _pairs(rx)                                   # (T, B)

    # --- прямой проход -----------------------------------------------------
    metrics = np.full((b_count, conv.N_STATES), INF, dtype=np.int32)
    metrics[:, 0] = 0                                # старт всегда из 00
    pred = np.empty((steps, b_count, conv.N_STATES), dtype=np.uint8)
    rows = np.arange(b_count)

    for t in range(steps):
        # стоимости всех рёбер: (B, 4 состояния-назначения, 2 предшественника)
        cand = metrics[:, EDGE_PREV] + COST[z[t][:, None, None], EDGE_OUT]
        idx = np.argmin(cand, axis=2)                # при равенстве — первое ребро
        pred[t] = EDGE_PREV[np.arange(conv.N_STATES)[None, :], idx]
        metrics = np.take_along_axis(cand, idx[:, :, None], axis=2)[:, :, 0]

    final_metrics = metrics.copy()

    # --- обратный проход ---------------------------------------------------
    state = np.zeros(b_count, dtype=np.int64) if end_state == "zero" \
        else np.argmin(metrics, axis=1).astype(np.int64)
    inputs = np.empty((steps, b_count), dtype=np.uint8)
    states_back = np.empty((steps + 1, b_count), dtype=np.uint8)
    states_back[steps] = state
    for t in range(steps - 1, -1, -1):
        inputs[t] = state >> 1                       # u_t = floor(q_{t+1} / 2)
        state = pred[t, rows, state].astype(np.int64)
        states_back[t] = state

    if end_state == "zero" and np.any(states_back[0] != 0):
        raise AssertionError("внутренняя ошибка: путь не начинается в состоянии 00")

    bits = inputs.T[:, :length].copy()               # обрезаем хвост
    if not return_details:
        return bits

    tail_bits = inputs.T[:, length:]
    details = {
        "final_metrics": final_metrics,
        "path_state": states_back.T,                 # (B, T+1)
        "full_input": inputs.T,                      # (B, T), включая хвост
        "tail_input": tail_bits,
        "end_state": states_back[-1],
    }
    return bits, details


def decode(rx, length: int | None = None, end_state: str = "zero",
           return_details: bool = False):
    """Декодировать один пакет (обёртка над :func:`decode_batch`)."""
    rx1 = np.asarray(rx)
    if rx1.ndim != 1:
        raise ValueError(f"decode: ожидается одномерный поток, ndim={rx1.ndim}")
    out = decode_batch(rx1[None, :], length=length, end_state=end_state,
                       return_details=return_details)
    if return_details:
        bits, details = out
        return bits[0], {k: (v[0] if isinstance(v, np.ndarray) and v.ndim > 0 else v)
                         for k, v in details.items()}
    return out[0]
