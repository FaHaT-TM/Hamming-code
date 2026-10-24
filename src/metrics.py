"""Метрики: BER, точные доверительные интервалы с учётом зависимости, теория.

Математическая спецификация: Агент №2, «Пакет 2 Агенту №1», разделы
«Точные консервативные интервалы с учётом зависимости — Q9»,
«теоретические ориентиры — Q10», «энергетический выигрыш — Q11».

Единица независимости
---------------------
Ошибки зависимы внутри единицы декодирования, поэтому независимой единицей
считается:

===============  ==========================  ==================
схема            единица                     длина ell
===============  ==========================  ==================
без кодирования  информационный бит           1
Хэмминг (7,4)    блок из 4 информационных     4
свёрточный код   завершённый пакет            L
===============  ==========================  ==================

``Z_j`` — число ошибочных информационных битов в j-й единице.

Консервативный интервал для BER
-------------------------------
``I_j = 1[Z_j > 0]``, ``K = Σ I_j ~ Binomial(M, q)``; так как
``I_j ≤ Z_j ≤ ell·I_j``, то ``q/ell ≤ P_b ≤ q``. Если ``[q_low, q_high]`` —
точный двусторонний интервал Клоппера—Пирсона для ``K`` из ``M``, то::

    [q_low / ell, q_high]

— консервативный 95 %-й интервал **для BER**. Интервал ``[q_low, q_high]``
без преобразования относится к BLER/FER, а не к BER.

Ноль ошибок
-----------
При ``E = 0`` также ``K = 0``; односторонняя 95 %-я граница::

    q_upper = 1 - 0.05^(1/M) = -expm1(log(0.05)/M)

именно она используется как ``ber_upper95`` (а не 3/N_info для
декодированных битов). Для некодированной передачи ``M = N_info`` и это
≈ 2.996/N_info; для Хэмминга ``M = N_info/4``; для Витерби ``M = N_info/L``.

Кластерная нормальная оценка (приближённая, дополнительный ориентир)
--------------------------------------------------------------------
``W_j = Z_j/ell``, ``SE = sqrt([Σ W_j² - M·mean(W)²] / (M-1) / M)``,
интервал ``mean(W) ± 1.96·SE``, обрезанный до [0, 1]; помечается
``approximate_cluster_normal`` и применяется при ``K ≥ 30``.

Точная BER кода Хэмминга (Q10)
------------------------------
Для стандартного синдромного декодера при жёстких решениях::

    P_b = (1/4) · Σ_{e ∈ {0,1}^7} z(e) · p^w · (1-p)^(7-w)

где ``w = wt(e)``, ``z(e)`` — число ошибочных информационных битов после
декодирования принятого слова ``e`` (передаётся нулевое кодовое слово,
что корректно в силу линейности декодера), ``p = Q(√(2·(4/7)·γ_b))``.

Суммарные ``Σ z(e)`` по весам: ``[0, 0, 36, 76, 64, 48, 28, 4]``, откуда
``P_b = 9p² - 26p³ + 30p⁴ - 12p⁵`` (в коде используется форма суммы
неотрицательных слагаемых).
"""

from __future__ import annotations

import math

import numpy as np

from . import channel
from . import hamming

__all__ = [
    "INFO_BITS_PER_UNIT", "count_bit_errors", "ber",
    "beta_inc", "beta_ppf", "clopper_pearson",
    "zero_error_upper_bound", "cluster_normal_interval",
    "ber_confidence", "bpsk_ber_theory", "hamming_ber_theory",
    "hamming_weight_profile", "hamming_ber_exact_series",
    "crossing_ebn0", "energy_gain_db",
    "hist_from_str", "hist_to_str", "cluster_stats_from_hist",
    "cluster_normal_from_hist", "bootstrap_gain",
    "theory_crossing_exact", "theory_crossing_grid", "grid_error_report",
]

# Длина единицы независимости (ell) по схеме
INFO_BITS_PER_UNIT = {
    "uncoded": 1,
    "hamming": hamming.K,      # 4 информационных бита в блоке
    "convolutional": None,     # = L, задаётся вызывающим кодом
}


# --------------------------------------------------------------------------
# Подсчёт ошибок
# --------------------------------------------------------------------------
def count_bit_errors(reference, decoded) -> int:
    """Число несовпадающих битов; длины должны совпадать (без потери битов)."""
    a = np.asarray(reference).ravel()
    b = np.asarray(decoded).ravel()
    if a.size != b.size:
        raise ValueError(f"разные длины: передано {a.size}, декодировано {b.size}")
    return int(np.count_nonzero(a.astype(np.uint8) != b.astype(np.uint8)))


def ber(n_errors: int, n_info_bits: int) -> float:
    """BER = N_errors / N_information_bits (всегда по информационным битам)."""
    if n_info_bits <= 0:
        raise ValueError("N_information_bits должно быть положительным")
    if not 0 <= n_errors <= n_info_bits:
        raise ValueError("N_errors вне диапазона [0, N_information_bits]")
    return n_errors / n_info_bits


# --------------------------------------------------------------------------
# Бета-распределение без SciPy (для точного интервала Клоппера—Пирсона)
# --------------------------------------------------------------------------
def beta_inc(a: float, b: float, x: float) -> float:
    """ Регуляризованная неполная бета-функция I_x(a, b).

    Реализована через непрерывную дробь (алгоритм Лентца — NR, betacf).
    Точность ~1e-14, достаточная для доверительных границ.
    """
    if not (0.0 <= x <= 1.0):
        raise ValueError("x должно быть в [0, 1]")
    if x == 0.0:
        return 0.0
    if x == 1.0:
        return 1.0
    log_front = (math.lgamma(a + b) - math.lgamma(a) - math.lgamma(b)
                 + a * math.log(x) + b * math.log1p(-x))
    front = math.exp(log_front)
    if x < (a + 1.0) / (a + b + 2.0):
        return front * _betacf(a, b, x) / a
    return 1.0 - front * _betacf(b, a, 1.0 - x) / b


def _betacf(a: float, b: float, x: float, max_iter: int = 300, eps: float = 3e-16) -> float:
    """Непрерывная дробь для неполной бета-функции (модифицированный Лентц)."""
    tiny = 1e-300
    qab, qap, qam = a + b, a + 1.0, a - 1.0
    c = 1.0
    d = 1.0 - qab * x / qap
    if abs(d) < tiny:
        d = tiny
    d = 1.0 / d
    h = d
    for m in range(1, max_iter + 1):
        m2 = 2 * m
        aa = m * (b - m) * x / ((qam + m2) * (a + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        h *= d * c
        aa = -(a + m) * (qab + m) * x / ((a + m2) * (qap + m2))
        d = 1.0 + aa * d
        if abs(d) < tiny:
            d = tiny
        c = 1.0 + aa / c
        if abs(c) < tiny:
            c = tiny
        d = 1.0 / d
        delta = d * c
        h *= delta
        if abs(delta - 1.0) < eps:
            break
    return h


def beta_ppf(q: float, a: float, b: float, tol: float = 1e-12,
             max_iter: int = 200) -> float:
    """Квантиль бета-распределения: x такой, что I_x(a, b) = q (бисекция)."""
    if not (0.0 <= q <= 1.0):
        raise ValueError("q должно быть в [0, 1]")
    if q == 0.0:
        return 0.0
    if q == 1.0:
        return 1.0
    lo, hi = 0.0, 1.0
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        if beta_inc(a, b, mid) < q:
            lo = mid
        else:
            hi = mid
        if hi - lo < tol:
            break
    return 0.5 * (lo + hi)


def clopper_pearson(k: int, m: int, confidence: float = 0.95) -> tuple[float, float]:
    """Точный двусторонний интервал Клоппера—Пирсона для доли при k из m."""
    if m <= 0:
        raise ValueError("m должно быть положительным")
    if not 0 <= k <= m:
        raise ValueError("k вне диапазона [0, m]")
    alpha = 1.0 - confidence
    low = 0.0 if k == 0 else beta_ppf(alpha / 2.0, k, m - k + 1)
    high = 1.0 if k == m else beta_ppf(1.0 - alpha / 2.0, k + 1, m - k)
    return low, high


def zero_error_upper_bound(m: int, confidence: float = 0.95) -> float:
    """Односторонняя верхняя граница при нуле ошибочных единиц: 1 − (1−conf)^(1/M).

    Численно устойчивая форма: ``-expm1(log(1 - confidence)/M)``.
    """
    if m <= 0:
        raise ValueError("m должно быть положительным")
    return float(-math.expm1(math.log(1.0 - confidence) / m))


def cluster_normal_interval(z_sum: float, z2_sum: float, n_units: int,
                            length: int, z_crit: float = 1.96) -> tuple[float, float, float]:
    """Приближённый кластерный нормальный интервал для BER.

    Returns
    -------
    (low, high, standard_error)
    """
    if n_units < 2:
        return 0.0, 1.0, float("nan")
    mean_w = (z_sum / length) / n_units
    var_w = (z2_sum / length ** 2 - n_units * mean_w ** 2) / (n_units - 1)
    var_w = max(var_w, 0.0)
    se = math.sqrt(var_w / n_units)
    return max(0.0, mean_w - z_crit * se), min(1.0, mean_w + z_crit * se), se


def ber_confidence(n_errors: int, n_info_bits: int, n_units: int,
                   info_per_unit: int, n_units_with_errors: int | None = None,
                   confidence: float = 0.95) -> dict:
    """Свести все интервалы для точки эксперимента.

    Parameters
    ----------
    n_errors : int
        ``E`` — суммарное число ошибочных информационных битов.
    n_info_bits : int
        ``N_info`` — число переданных информационных битов.
    n_units : int
        ``M`` — число независимых единиц (бит / блоков / пакетов).
    info_per_unit : int
        ``ell`` — информационных битов в единице.
    n_units_with_errors : int, optional
        ``K`` — число единиц с ошибками (по умолчанию считается, что K = 0 при E = 0).

    Returns
    -------
    dict
        ``ber``, ``interval_type``, ``ci_low``, ``ci_high``, ``upper95``,
        ``cluster_low``, ``cluster_high``, ``cluster_se``, ``K``, ``M``.
    """
    if n_units <= 0:
        raise ValueError("n_units должно быть положительным")
    observed = ber(n_errors, n_info_bits)
    k = n_units_with_errors
    if k is None:
        k = 0 if n_errors == 0 else min(n_units, n_errors)
    if not 0 <= k <= n_units:
        raise ValueError("n_units_with_errors вне диапазона [0, M]")

    out = {
        "ber": observed,
        "K": int(k),
        "M": int(n_units),
        "ell": int(info_per_unit),
        "ci_low": float("nan"),
        "ci_high": float("nan"),
        "upper95": float("nan"),
        "interval_type": "",
        "cluster_low": float("nan"),
        "cluster_high": float("nan"),
        "cluster_se": float("nan"),
    }

    if n_errors == 0:
        out["upper95"] = zero_error_upper_bound(n_units, confidence)
        out["interval_type"] = "conservative_one_sided_95_no_errors"
        return out

    q_low, q_high = clopper_pearson(k, n_units, confidence)
    # консервативное преобразование для BER: [q_low/ell, q_high]
    out["ci_low"] = q_low / info_per_unit
    out["ci_high"] = q_high
    out["interval_type"] = "conservative_95_by_independent_units"
    return out


# --------------------------------------------------------------------------
# Теория
# --------------------------------------------------------------------------
def bpsk_ber_theory(ebno_db):
    """BER некодированной BPSK: P_b = Q(√(2γ_b)) = 0.5·erfc(√γ_b)."""
    arr = np.asarray(ebno_db, dtype=float)
    out = channel.qfunc(np.sqrt(2.0 * 10.0 ** (arr / 10.0)))
    return float(out) if arr.ndim == 0 else out


def hamming_weight_profile() -> np.ndarray:
    """``Σ z(e)`` по весам ``w = 0…7`` для всех 128 шаблонов ошибок.

    ``z(e)`` — число ошибочных информационных битов после стандартного
    синдромного декодирования принятого слова ``e`` (передаётся нулевое
    кодовое слово). Ожидаемый профиль: ``[0, 0, 36, 76, 64, 48, 28, 4]``.
    """
    profile = np.zeros(8, dtype=np.int64)
    for e_int in range(128):
        e = np.array([(e_int >> (6 - i)) & 1 for i in range(7)], dtype=np.uint8)
        decoded = hamming.decode(e)
        profile[int(e.sum())] += int(decoded.sum())
    return profile


def hamming_ber_exact_series(p):
    """Точная BER Хэмминга (7,4) в форме суммы неотрицательных слагаемых."""
    p = float(p)
    z = hamming_weight_profile() / 4.0        # (1/4)·Σ z(e)
    total = 0.0
    for w in range(8):
        if z[w]:
            total += z[w] * p ** w * (1.0 - p) ** (7 - w)
    return total


def hamming_ber_theory(ebno_db):
    """Точная теоретическая BER кода Хэмминга (7,4) при жёстких решениях.

    ``p = Q(√(2·(4/7)·γ_b))`` — вероятность ошибки кодового бита; далее
    точный перебор 128 шаблонов ошибок.
    """
    arr = np.asarray(ebno_db, dtype=float)
    rate = hamming.RATE
    scalar = arr.ndim == 0
    vals = arr.ravel() if not scalar else arr.reshape(1)
    out = np.array([
        hamming_ber_exact_series(channel.qfunc(math.sqrt(2.0 * rate * 10.0 ** (v / 10.0))))
        for v in vals
    ])
    return float(out[0]) if scalar else out.reshape(arr.shape)


# --------------------------------------------------------------------------
# Энергетический выигрыш (Q11)
# --------------------------------------------------------------------------
def crossing_ebn0(x1: float, p1: float, x2: float, p2: float,
                  target: float) -> float | None:
    """Точка пересечения уровня ``target`` линейной интерполяцией по lg(BER).

    ``x* = x1 + (x2 − x1)·[lg(P*) − lg(p1)] / [lg(p2) − lg(p1)]``.

    Возвращает ``None``, если интерполяция невозможна (нули, равные lg).
    Экстраполяция не выполняется: обе точки должны охватывать уровень.
    """
    if p1 <= 0 or p2 <= 0 or target <= 0:
        return None
    if (p1 - target) * (p2 - target) > 0:
        return None                              # уровень не охвачен парой
    l1, l2, lt = math.log10(p1), math.log10(p2), math.log10(target)
    if l1 == l2:
        return None
    return x1 + (x2 - x1) * (lt - l1) / (l2 - l1)


def energy_gain_db(curve_uncoded: dict, curve_coded: dict, target: float) -> dict:
    """Выигрыш по горизонтали при заданном BER по сетке точек.

    Parameters
    ----------
    curve_uncoded, curve_coded : dict
        ``{ebno_db: ber}`` — только точки с ``ber > 0`` (точки с нулём ошибок
        и верхние границы в интерполяцию не включаются).

    Returns
    -------
    dict
        ``gain_db`` (или ``None``), ``x_uncoded``, ``x_coded``, ``target``.
    """
    def find(curve):
        xs = sorted(curve)
        for a, b in zip(xs, xs[1:]):
            x = crossing_ebn0(a, curve[a], b, curve[b], target)
            if x is not None:
                return x
        return None

    xu, xc = find(curve_uncoded), find(curve_coded)
    gain = None if (xu is None or xc is None) else xu - xc
    return {"target": target, "x_uncoded": xu, "x_coded": xc, "gain_db": gain}


# --------------------------------------------------------------------------
# Кластерные оценки по гистограмме Z_j (замечание Агента №2, §5)
# --------------------------------------------------------------------------
def hist_from_str(text: str) -> dict:
    """Разобрать строку ``"0:123;1:45"`` в словарь ``{z: count}``."""
    out = {}
    for part in str(text).split(";"):
        part = part.strip()
        if not part:
            continue
        z, count = part.split(":")
        out[int(z)] = out.get(int(z), 0) + int(count)
    return out


def hist_to_str(hist: dict) -> str:
    """Обратное преобразование (для CSV)."""
    return ";".join(f"{int(z)}:{int(c)}" for z, c in sorted(hist.items()) if c)


def cluster_stats_from_hist(hist: dict, ell: int) -> dict:
    """Кластерные статистики по гистограмме числа ошибок в единице.

    ``W_j = Z_j/ell``, ``mean_W = E/(M·ell)``,

    ``s_W² = [Σ Z_j²/ell² − M·mean_W²] / (M − 1)``,   ``SE = sqrt(s_W²/M)``.

    Возвращает ``E``, ``M``, ``K`` (единиц с ``Z_j > 0``), ``mean_W``, ``var_W``, ``se``.
    """
    M = int(sum(hist.values()))
    if M <= 0:
        raise ValueError("пустая гистограмма")
    E = float(sum(z * c for z, c in hist.items()))
    S2 = float(sum(z * z * c for z, c in hist.items()))
    K = int(sum(c for z, c in hist.items() if z > 0))
    mean_w = E / (M * ell)
    var_w = (S2 / ell ** 2 - M * mean_w ** 2) / (M - 1) if M > 1 else float("nan")
    se = math.sqrt(max(var_w, 0.0) / M) if M > 1 else float("nan")
    return {"E": E, "M": M, "K": K, "ell": int(ell), "mean_W": mean_w,
            "var_W": var_w, "se": se}


def cluster_normal_from_hist(hist: dict, ell: int, z_crit: float = 1.96,
                             min_units_with_errors: int = 30) -> dict:
    """Приближённый кластерный нормальный интервал ``mean_W ± z·SE``.

    Оценка **не применяется**, когда ``K < min_units_with_errors`` или ``K = 0``:
    нормальная аппроксимация при редких событиях не обоснована. В этом случае
    в результате ``applicable=False`` и ``ci_low/ci_high = nan``.
    """
    st = cluster_stats_from_hist(hist, ell)
    applicable = st["K"] >= min_units_with_errors
    out = dict(st)
    out["applicable"] = bool(applicable)
    out["ci_low"] = max(0.0, st["mean_W"] - z_crit * st["se"]) if applicable else float("nan")
    out["ci_high"] = min(1.0, st["mean_W"] + z_crit * st["se"]) if applicable else float("nan")
    out["z_crit"] = z_crit
    return out


def _draw_multinomial(rng: np.random.Generator, hist: dict, m: int) -> np.ndarray:
    """Ресэмплировать M единиц из гистограммы (мультиномиальное вытягивание)."""
    zs = np.fromiter(hist.keys(), dtype=np.int64, count=len(hist))
    counts = np.fromiter(hist.values(), dtype=np.float64, count=len(hist))
    probs = counts / counts.sum()
    return rng.multinomial(m, probs)


def _find_crossing_pair(curve: dict, target: float):
    """Первое пересечение уровня ``target`` и пара точек, его давшая."""
    xs = sorted(curve)
    for a, b in zip(xs, xs[1:]):
        x = crossing_ebn0(a, curve[a], b, curve[b], target)
        if x is not None:
            return x, (a, b)
    return None


def bootstrap_gain(curves: dict, ells: dict, target: float, n_boot: int = 1000,
                   seed: int = 0, confidence: float = 0.95,
                   reference_pairs: dict | None = None) -> dict:
    """Кластерный bootstrap неопределённости интерполированного выигрыша.

    Процедура (согласована с Агентом №2, §5):

    1. Ресэмплирование **по независимым единицам**: для каждой точки из её
       гистограммы вытягиваются новые частоты ``Multinomial(M, h/M)``, затем
       ``BER* = E*/(M·ℓ)``.
    2. Ресэмплируются **все** схемы, включая некодированную опорную; из-за
       независимых потоков схемы и точки ресэмплируются независимо.
    3. В каждой реплике пересечение ищется **по всей сетке**, а не только по паре
       соседей, давшей пересечение в исходной выборке. Доля реплик, где пара
       изменилась, возвращается как диагностика.
    4. Реплики без пересечения (нули, уровень не охвачен) **не экстраполируются
       и не отбрасываются молча**: они подсчитываются, возвращается их число и доля.
    5. Интервал — percentile по квантилям ``alpha/2`` и ``1-alpha/2`` валидных реплик;
       при заметной доле невалидных реплик интервал условен.

    Parameters
    ----------
    curves : dict
        ``{scheme: {ebno_db: hist}}``; ключ ``"uncoded"`` обязателен.
    ells : dict
        Длина единицы независимости по схеме.
    target : float
        Уровень BER.
    n_boot, seed, confidence
        Параметры процедуры; ``seed`` — отдельный генератор, не расходующий
        поток канала.
    reference_pairs : dict, optional
        Пары точек, давшие пересечение в исходной выборке (для диагностики смены пары).

    Returns
    -------
    dict
        По каждой кодированной схеме: ``gain_db``, ``ci_low``, ``ci_high``,
        ``bootstrap_sd``, ``n_boot``, ``n_valid``, ``invalid_fraction``,
        ``pair_changed_fraction``, ``note``.
    """
    rng = np.random.default_rng(seed)

    def ber_of(hist: dict, ell: int, draws=None) -> float:
        m = int(sum(hist.values()))
        zs = np.fromiter(hist.keys(), dtype=np.int64, count=len(hist))
        counts = np.fromiter(hist.values(), dtype=float, count=len(hist))
        if draws is None:
            draws = counts
        return float(np.dot(draws, zs)) / (m * ell)

    # точечная оценка по исходным данным
    observed = {}
    for name, points in curves.items():
        observed[name] = {x: ber_of(h, ells[name]) for x, h in points.items()}
        observed[name] = {x: v for x, v in observed[name].items() if v > 0}
    xu_obs = _find_crossing_pair(observed["uncoded"], target)
    xu_obs = None if xu_obs is None else xu_obs[0]
    point_gain = {}
    for name in curves:
        if name == "uncoded":
            continue
        xc, _pair = _find_crossing_pair(observed[name], target) or (None, None)
        point_gain[name] = None if (xu_obs is None or xc is None) else xu_obs - xc

    replicates = {name: [] for name in curves if name != "uncoded"}
    pair_changed = {name: 0 for name in curves if name != "uncoded"}
    invalid = 0
    for _ in range(n_boot):
        sim_curves = {}
        for name, points in curves.items():
            sim_points = {}
            for x, h in points.items():
                m = int(sum(h.values()))
                counts = np.fromiter(h.values(), dtype=float, count=len(h))
                draws = rng.multinomial(m, counts / counts.sum())
                sim_points[x] = ber_of(h, ells[name], draws)
            sim_curves[name] = {x: v for x, v in sim_points.items() if v > 0}
        found = _find_crossing_pair(sim_curves["uncoded"], target)
        if found is None:
            invalid += 1
            continue
        xu, pair_u = found
        gains, pairs = {}, {}
        ok = True
        for name in replicates:
            f = _find_crossing_pair(sim_curves[name], target)
            if f is None:
                ok = False
                break
            gains[name], pairs[name] = f
        if not ok:
            invalid += 1
            continue
        for name in replicates:
            replicates[name].append(xu - gains[name])
        # диагностика смены пары соседних точек относительно исходной выборки
        for name in replicates:
            if reference_pairs and name in reference_pairs:
                ref_pair = reference_pairs[name]
                if ref_pair is not None and pairs[name] != ref_pair:
                    pair_changed[name] += 1

    n_valid = n_boot - invalid
    alpha = 1.0 - confidence
    out = {}
    for name in replicates:
        arr = np.asarray(replicates[name], dtype=float)
        out[name] = {
            "gain_db": point_gain.get(name),
            "ci_low": float(np.quantile(arr, alpha / 2)) if arr.size else float("nan"),
            "ci_high": float(np.quantile(arr, 1 - alpha / 2)) if arr.size else float("nan"),
            "bootstrap_sd": float(arr.std(ddof=1)) if arr.size > 1 else float("nan"),
            "n_boot": n_boot,
            "n_valid": n_valid,
            "invalid_fraction": invalid / n_boot if n_boot else float("nan"),
            "pair_changed_fraction": (pair_changed[name] / n_valid
                                      if n_valid else float("nan")),
            "method": "percentile по квантилям 2.5 % и 97.5 % валидных реплик",
            "note": ("приближённый кластерный bootstrap: оценивает статистический разброс "
                     "выбранной процедуры интерполяции на заданной сетке; не включает "
                     "систематическую погрешность сетки, ошибки модели и реализации"),
        }
    return out


def _find_crossing(curve: dict, target: float):
    """Первое пересечение уровня ``target`` на сетке точек (без экстраполяции)."""
    xs = sorted(curve)
    for a, b in zip(xs, xs[1:]):
        x = crossing_ebn0(a, curve[a], b, curve[b], target)
        if x is not None:
            return x
    return None


# --------------------------------------------------------------------------
# Сеточная ошибка самой процедуры интерполяции на точных кривых (замечание §6)
# --------------------------------------------------------------------------
def theory_crossing_exact(func, target: float, lo: float = -2.0, hi: float = 14.0,
                          tol: float = 1e-10) -> float:
    """Точный корень ``func(x) = target`` бисекцией (без моделирования).

    ``func`` — монотонно убывающая по ``x`` функция BER(Eb/N0 в дБ).
    """
    f_lo, f_hi = func(lo) - target, func(hi) - target
    if f_lo * f_hi > 0:
        raise ValueError("корень не отделён на заданном интервале")
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        f_mid = func(mid) - target
        if abs(f_mid) < tol or (hi - lo) < tol:
            return mid
        if f_mid * f_lo <= 0:
            hi = mid
        else:
            lo, f_lo = mid, f_mid
    return 0.5 * (lo + hi)


def theory_crossing_grid(func, target: float, grid) -> float | None:
    """Пересечение уровня по сетке значений функции (lg-интерполяция), без экстраполяции."""
    xs = list(grid)
    vals = {x: func(x) for x in xs}
    return _find_crossing(vals, target)


def grid_error_report(target: float = 1e-4) -> dict:
    """Сеточная ошибка процедуры интерполяции на точных теоретических кривых.

    Сравнивается точный корень с результатом lg-интерполяции тех же теоретических
    значений на сетке 1 дБ (0…10) и на сетке 0.2 дБ (локальные диапазоны).
    Это характеристика **процедуры**, а не экспериментальная оценка.
    """
    out = {}
    specs = {
        "bpsk": (bpsk_ber_theory, 0.2, (7.6, 9.0)),
        "hamming": (hamming_ber_theory, 0.2, (7.4, 8.8)),
    }
    for name, (func, step, local_span) in specs.items():
        exact = theory_crossing_exact(func, target)
        coarse = theory_crossing_grid(func, target, [float(x) for x in range(0, 11)])
        n_steps = int(round((local_span[1] - local_span[0]) / step))
        local = [local_span[0] + i * step for i in range(n_steps + 1)]
        fine = theory_crossing_grid(func, target, local)
        out[name] = {
            "exact_db": exact,
            "grid_1db_db": coarse,
            "grid_error_1db": None if coarse is None else coarse - exact,
            "grid_0p2db_db": fine,
            "grid_error_0p2db": None if fine is None else fine - exact,
        }
    return out
