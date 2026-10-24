"""Движок BER-эксперимента: фиксированный план, счётчики, сохранение в CSV.

Математическая спецификация: docs/SPECIFICATIONS_CONVOLUTIONAL.md, модули
«BPSK и AWGN — Q6» и «план BER-эксперимента — Q7–Q9».

План эксперимента (фиксируется **до** запуска)
----------------------------------------------
===============  ==============  =========================  =============
схема            N_info, бит     единиц M                   ell
===============  ==============  =========================  =============
без кодирования  2 000 000       2 000 000 битов            1
Хэмминг (7,4)    2 000 000       500 000 блоков             4
свёрточный       1 000 000       100 пакетов по L = 10 000  L = 10 000
===============  ==============  =========================  =============

Диапазон 0…10 дБ с шагом 1 дБ для всех схем. Объём фиксирован: число
ошибочных битов — **диагностика**, а не критерий остановки. Для Витерби
дополнительно считается число пакетов с ошибками ``K``: 100 ошибочных битов
в одном пакете не означают 100 независимых событий.

Если точка прервана по времени, она помечается ``incomplete``, счётчики
сохраняются фактическими, и обычные интервалы для неё не выдаются за точные.

Сохраняемые счётчики
--------------------
``E`` — битовые ошибки, ``N_info`` — переданные информационные биты,
``M`` — независимые единицы, ``K`` — единицы с ошибками, ``sum_Z2`` —
Σ Z_j², ``BER = E / N_info``, ``R_nominal``, ``R_eff``, ``sigma2``,
тип и уровень интервала, время, статус.
"""

from __future__ import annotations

import csv
import math
import os
import platform
import time
from dataclasses import dataclass, field, asdict

import numpy as np

from . import channel, convolutional as conv, hamming, metrics, viterbi

__all__ = [
    "SCHEMES", "Plan", "PointResult", "run_point", "run_experiment",
    "write_csv", "write_theory_csv", "performance_probe", "DEFAULT_PLAN",
]

# --------------------------------------------------------------------------
# Схемы
# --------------------------------------------------------------------------
SCHEMES = {
    "uncoded": {
        "name_ru": "Без кодирования (BPSK)",
        "r_nominal": 1.0,
        "ell": 1,                 # независимая единица — бит
        "color": "#1f77b4",
        "marker": "o",
        "label": "без кодирования (BPSK)",
    },
    "hamming": {
        "name_ru": "Хэмминг (7,4)",
        "r_nominal": hamming.RATE,
        "ell": hamming.K,         # независимая единица — блок из 4 бит
        "color": "#d62728",
        "marker": "s",
        "label": "Хэмминг (7,4)",
    },
    "convolutional": {
        "name_ru": "Свёрточный (7,5)₈ + Витерби",
        "r_nominal": 0.5,
        "ell": conv.L_DEFAULT,    # независимая единица — пакет
        "color": "#2ca02c",
        "marker": "^",
        "label": "свёрточный (7,5)₈, Витерби",
    },
}


@dataclass
class Plan:
    """Фиксированный план основного эксперимента."""

    seed: int = 20261024
    ebno_db: tuple[float, ...] = tuple(float(x) for x in range(0, 11))
    n_info_uncoded: int = 2_000_000
    n_info_hamming: int = 2_000_000
    n_info_convolutional: int = 1_000_000
    packet_length: int = conv.L_DEFAULT
    chunk_bits: int = 200_000        # размер порции для контроля памяти
    packets_per_batch: int = 10      # пакетов Витерби одновременно в памяти
    time_budget_s: float = 120.0     # ориентир на точку (диагностика, не остановка)
    confidence: float = 0.95

    def n_info(self, scheme: str) -> int:
        return {
            "uncoded": self.n_info_uncoded,
            "hamming": self.n_info_hamming,
            "convolutional": self.n_info_convolutional,
        }[scheme]

    @property
    def r_eff(self) -> dict:
        return {
            "uncoded": 1.0,
            "hamming": hamming.RATE,
            "convolutional": conv.rate_effective(self.packet_length),
        }


DEFAULT_PLAN = Plan()

# Идентификатор схемы в схеме потоков (используется в rng_for(seed, tag, id, ...))
_SCHEME_ID = {"uncoded": 0, "hamming": 1, "convolutional": 2}


@dataclass
class PointResult:
    """Результат одной точки (схема × Eb/N0)."""

    scheme: str
    ebno_db: float
    seed: int
    n_info_bits: int = 0
    n_tx_bits: int = 0
    n_errors: int = 0
    n_units: int = 0
    n_units_with_errors: int = 0
    sum_z2: float = 0.0
    elapsed_s: float = 0.0
    status: str = "ok"
    r_nominal: float = float("nan")
    r_eff: float = float("nan")
    sigma2: float = float("nan")
    ell: int = 1
    length: int = 0
    tail_bits: int = 0
    ber: float = float("nan")
    ci_type: str = ""
    ci_level: float = 0.95
    ci_low: float = float("nan")
    ci_high: float = float("nan")
    upper95: float = float("nan")
    cluster_low: float = float("nan")
    cluster_high: float = float("nan")
    cluster_se: float = float("nan")
    # гистограмма Z_j: "z:count;z:count;…" — нужна для кластерных оценок и bootstrap
    z_hist: str = ""
    dataset: str = "main"          # main | extended | refinement
    streams: str = ""              # описание схемы SeedSequence/идентификаторов потоков
    extra: dict = field(default_factory=dict)

    def to_row(self) -> dict:
        row = asdict(self)
        extra = row.pop("extra")
        row.update({f"x_{k}": v for k, v in extra.items()})
        row["python_version"] = platform.python_version()
        row["numpy_version"] = np.__version__
        row["timestamp"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        return row


# --------------------------------------------------------------------------
# Один режим
# --------------------------------------------------------------------------
def _simulate_uncoded(plan: Plan, ebno_db: float, n_info: int, tag: int):
    """Без кодирования. Возвращает (E, ΣZ², N_info, K), K = E (единица = бит)."""
    r_eff = 1.0
    errors, sum_z2 = 0, 0.0
    hist = np.zeros(2, dtype=np.int64)      # Z_j ∈ {0, 1}
    done = 0
    chunk_index = 0
    while done < n_info:
        size = min(plan.chunk_bits, n_info - done)
        # отдельный поток на каждую порцию: seed не переинициализируется повтором
        rng_bits = channel.rng_for(plan.seed, tag, 0, chunk_index, 1)
        rng_noise = channel.rng_for(plan.seed, tag, 0, chunk_index, 2)
        info = rng_bits.integers(0, 2, size=size, dtype=np.uint8)
        rx = channel.transmit_hard(info, ebno_db, r_eff, rng_noise)
        err = int(np.count_nonzero(info != rx))
        errors += err
        sum_z2 += float(err)          # Z_j ∈ {0,1}, Σ Z_j² = Σ Z_j
        hist[0] += size - err
        hist[1] += err
        done += size
        chunk_index += 1
    return errors, sum_z2, done, errors, hist     # K = E: единица независимости — бит


def _simulate_hamming(plan: Plan, ebno_db: float, n_info: int, tag: int):
    """Хэмминг (7,4). Возвращает (E, ΣZ², N_info, K), K — блоки с ошибками."""
    r_eff = hamming.RATE
    errors, sum_z2, units_with_errors = 0, 0.0, 0
    hist = np.zeros(hamming.K + 1, dtype=np.int64)   # Z_j ∈ 0..4
    done = 0
    chunk_index = 0
    while done < n_info:
        size = min(plan.chunk_bits - plan.chunk_bits % hamming.K, n_info - done)
        rng_bits = channel.rng_for(plan.seed, tag, 1, chunk_index, 1)
        rng_noise = channel.rng_for(plan.seed, tag, 1, chunk_index, 2)
        info = rng_bits.integers(0, 2, size=size, dtype=np.uint8)
        code = hamming.encode(info)
        rx = channel.transmit_hard(code, ebno_db, r_eff, rng_noise)
        decoded = hamming.decode(rx)
        per_block = np.count_nonzero(
            decoded.reshape(-1, hamming.K) != info.reshape(-1, hamming.K), axis=1
        )
        errors += int(per_block.sum())
        sum_z2 += float(np.sum(per_block.astype(np.float64) ** 2))
        units_with_errors += int(np.count_nonzero(per_block))
        hist += np.bincount(per_block, minlength=hamming.K + 1)
        done += size
        chunk_index += 1
    return errors, sum_z2, done, units_with_errors, hist


def _simulate_convolutional(plan: Plan, ebno_db: float, n_info: int, tag: int):
    """Свёрточный код + Витерби. Возвращает (E, ΣZ², N_info, K), K — пакеты с ошибками."""
    length = plan.packet_length
    r_eff = conv.rate_effective(length)
    tail = conv.TAIL
    n_packets = n_info // length
    if n_packets == 0:
        raise ValueError("объём меньше одного пакета")
    errors, sum_z2, units_with_errors = 0, 0.0, 0
    hist = {}                                # Z_j -> число пакетов (разреженная гистограмма)
    packets_done = 0
    batch_index = 0
    while packets_done < n_packets:
        b = min(plan.packets_per_batch, n_packets - packets_done)
        rng_bits = channel.rng_for(plan.seed, tag, 2, batch_index, 1)
        rng_noise = channel.rng_for(plan.seed, tag, 2, batch_index, 2)
        info = rng_bits.integers(0, 2, size=(b, length), dtype=np.uint8)
        code = conv.encode_batch(info)
        # шум независим для каждого символа всех пакетов порции
        x = channel.bpsk_modulate(code)
        y = channel.add_awgn(x, channel.sigma_from_ebno(ebno_db, r_eff), rng_noise)
        rx = channel.bpsk_demodulate_hard(y)
        decoded = viterbi.decode_batch(rx, length=length)
        per_packet = np.count_nonzero(decoded != info, axis=1)
        errors += int(per_packet.sum())
        sum_z2 += float(np.sum(per_packet.astype(np.float64) ** 2))
        units_with_errors += int(np.count_nonzero(per_packet))
        for z_value, count in zip(*np.unique(per_packet, return_counts=True)):
            hist[int(z_value)] = hist.get(int(z_value), 0) + int(count)
        packets_done += b
        batch_index += 1
    dense = np.zeros(max(hist) + 1 if hist else 1, dtype=np.int64)
    for z_value, count in hist.items():
        dense[z_value] = count
    return errors, sum_z2, packets_done * length, units_with_errors, dense


def run_point(scheme: str, ebno_db: float, plan: Plan = DEFAULT_PLAN,
              tag: int = 0, dataset: str = "main") -> PointResult:
    """Выполнить одну точку эксперимента (схема × Eb/N0).

    Parameters
    ----------
    scheme : {'uncoded', 'hamming', 'convolutional'}
    ebno_db : float
        Eb/N0 в дБ (на информационный бит до кодирования).
    plan : Plan
        Фиксированный план (seed, объёмы, длины пакетов).
    tag : int
        Метка потока случайности (различает схемы).

    Returns
    -------
    PointResult
    """
    if scheme not in SCHEMES:
        raise ValueError(f"неизвестная схема: {scheme}")
    meta = SCHEMES[scheme]
    n_info = plan.n_info(scheme)
    r_eff = plan.r_eff[scheme]

    t0 = time.perf_counter()
    if scheme == "uncoded":
        errors, sum_z2, done, units_with_errors, z_hist = _simulate_uncoded(
            plan, ebno_db, n_info, tag)
        n_tx = done
        units = done
        ell = 1
        length, tail = 0, 0
    elif scheme == "hamming":
        errors, sum_z2, done, units_with_errors, z_hist = _simulate_hamming(
            plan, ebno_db, n_info, tag)
        n_tx = done * hamming.N // hamming.K
        units = done // hamming.K
        ell = hamming.K
        length, tail = hamming.K, 0
    else:
        errors, sum_z2, done, units_with_errors, z_hist = _simulate_convolutional(
            plan, ebno_db, n_info, tag)
        n_tx = done * conv.N_OUT * (plan.packet_length + conv.TAIL) // plan.packet_length
        units = done // plan.packet_length
        ell = plan.packet_length
        length, tail = plan.packet_length, conv.TAIL
    elapsed = time.perf_counter() - t0

    res = PointResult(
        scheme=scheme, ebno_db=float(ebno_db), seed=plan.seed,
        n_info_bits=int(done), n_tx_bits=int(n_tx), n_errors=int(errors),
        n_units=int(units), sum_z2=float(sum_z2), elapsed_s=float(elapsed),
        r_nominal=meta["r_nominal"], r_eff=r_eff,
        sigma2=channel.variance_from_ebno(ebno_db, r_eff),
        ell=int(ell), length=int(length), tail_bits=int(tail),
        ci_level=plan.confidence,
        status="ok" if elapsed <= plan.time_budget_s else "ok_over_budget",
        z_hist=";".join(f"{int(z)}:{int(c)}" for z, c in enumerate(z_hist) if c),
        dataset=dataset,
        streams=(f"SeedSequence(entropy=[{plan.seed}, scheme_tag, {tag}, {_SCHEME_ID[scheme]}, "
                 f"chunk_index, purpose]); purpose=1 биты, 2 шум; chunk_index=0.."),
    )

    # K — число независимых единиц с ошибками (битов / блоков / пакетов)
    ci = metrics.ber_confidence(
        n_errors=errors, n_info_bits=done, n_units=units,
        info_per_unit=ell, n_units_with_errors=units_with_errors,
        confidence=plan.confidence,
    )
    res.ber = ci["ber"]
    res.n_units_with_errors = ci["K"]
    res.ci_type = ci["interval_type"]
    res.ci_low, res.ci_high = ci["ci_low"], ci["ci_high"]
    res.upper95 = ci["upper95"]

    if errors > 0 and units >= 2:
        lo, hi, se = metrics.cluster_normal_interval(errors, sum_z2, units, ell)
        res.cluster_low, res.cluster_high, res.cluster_se = lo, hi, se
    res.extra = {"n_units_with_errors": ci["K"], "unit_ell": ell}
    return res


# --------------------------------------------------------------------------
# Весь эксперимент
# --------------------------------------------------------------------------
def run_experiment(plan: Plan = DEFAULT_PLAN, schemes: tuple[str, ...] = tuple(SCHEMES),
                   verbose: bool = True, dataset: str = "main",
                   ebno_by_scheme: dict | None = None) -> list[PointResult]:
    """Прогнать все точки плана и вернуть результаты (без записи на диск)."""
    results: list[PointResult] = []
    for scheme_index, scheme in enumerate(schemes):
        grid = plan.ebno_db if ebno_by_scheme is None else ebno_by_scheme[scheme]
        for ebno in grid:
            if verbose:
                print(f"  {scheme:14s} Eb/N0 = {ebno:4.1f} дБ ...", end=" ", flush=True)
            res = run_point(scheme, ebno, plan, tag=scheme_index, dataset=dataset)
            results.append(res)
            if verbose:
                print(f"E={res.n_errors:7d}  N={res.n_info_bits:9d}  "
                      f"BER={res.ber:.3e}  {res.elapsed_s:5.1f} c")
    return results


# --------------------------------------------------------------------------
# Сохранение
# --------------------------------------------------------------------------
def write_csv(results, path: str) -> str:
    """Записать результаты в CSV без округления исходных данных."""
    if not results:
        raise ValueError("пустой список результатов")
    rows = [r.to_row() for r in results]
    # порядок столбцов: сначала ключевые, затем остальные в стабильном порядке
    first = ["dataset", "scheme", "ebno_db", "n_info_bits", "n_tx_bits", "n_errors", "ber",
             "n_units", "n_units_with_errors", "sum_z2", "z_hist",
             "ci_type", "ci_level", "ci_low", "ci_high", "upper95",
             "cluster_low", "cluster_high", "cluster_se",
             "r_nominal", "r_eff", "sigma2", "ell", "length", "tail_bits",
             "seed", "streams", "status", "elapsed_s"]
    rest = [k for k in rows[0] if k not in first]
    columns = first + sorted(rest)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in columns})
    return path


def write_theory_csv(path: str, ebno_db=None) -> str:
    """Теоретические кривые: BPSK без кодирования и точная BER Хэмминга."""
    if ebno_db is None:
        ebno_db = np.linspace(0.0, 10.0, 201)
    rows = []
    for x in ebno_db:
        rows.append({
            "ebno_db": float(x),
            "ber_bpsk_theory": metrics.bpsk_ber_theory(x),
            "ber_hamming_theory": metrics.hamming_ber_theory(x),
        })
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["ebno_db", "ber_bpsk_theory",
                                               "ber_hamming_theory"])
        writer.writeheader()
        writer.writerows(rows)
    return path


# --------------------------------------------------------------------------
# Замер производительности (диагностика, не входит в итоговую статистику)
# --------------------------------------------------------------------------
def performance_probe(plan: Plan = DEFAULT_PLAN, ebno_db: float = 3.0,
                      repeats: int = 3) -> dict:
    """Предварительный замер стоимости точки, чтобы согласовать объёмы.

    Данные этого замера **не используются** в итоговой статистике.
    """
    probe_plan = Plan(**{**asdict(plan),
                         "n_info_uncoded": 200_000,
                         "n_info_hamming": 200_000,
                         "n_info_convolutional": 100_000,
                         "packets_per_batch": plan.packets_per_batch})
    out = {}
    for scheme in SCHEMES:
        times = []
        for _ in range(repeats):
            res = run_point(scheme, ebno_db, probe_plan)
            times.append(res.elapsed_s)
        out[scheme] = {
            "n_info_bits": probe_plan.n_info(scheme),
            "median_s": float(np.median(times)),
            "per_million_s": float(np.median(times)) * 1e6 / probe_plan.n_info(scheme),
        }
    out["estimated_full_run_s"] = sum(
        v["per_million_s"] * plan.n_info(s) / 1e6 * len(plan.ebno_db)
        for s, v in out.items() if isinstance(v, dict)
    )
    return out
