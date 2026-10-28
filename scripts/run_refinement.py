"""Уточняющий эксперимент по зафиксированному протоколу (шаг 0.2 дБ, seed 20261026).

Протокол зафиксирован до запуска:

    без кодирования : 8.0, 8.2, 8.4, 8.6, 8.8 дБ × 30 000 000 информационных бит
    Хэмминг (7,4)   : 7.8, 8.0, 8.2, 8.4, 8.6 дБ × 30 000 000 (7 500 000 блоков)
    свёрточный      : 6.6, 6.8, 7.0, 7.2, 7.4 дБ × 1000 пакетов по L = 10 000

Объёмы фиксированы до запуска и не меняются после просмотра промежуточных ошибок;
остановки по числу ошибок нет; результаты не объединяются с прежними прогонами
в основную подтверждающую оценку.

Запуск: python scripts/run_refinement.py
"""

from __future__ import annotations

import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import simulation as sim  # noqa: E402

SEED = 20261026
GRIDS = {
    "uncoded": (8.0, 8.2, 8.4, 8.6, 8.8),
    "hamming": (7.8, 8.0, 8.2, 8.4, 8.6),
    "convolutional": (6.6, 6.8, 7.0, 7.2, 7.4),
}
DATA = os.path.join(ROOT, "results", "data")
OUT_CSV = os.path.join(DATA, "ber_refinement.csv")
OUT_JSON = os.path.join(DATA, "refinement_plan.json")


def main() -> None:
    plan = sim.Plan(
        seed=SEED,
        ebno_db=tuple(sorted({x for g in GRIDS.values() for x in g})),
        n_info_uncoded=30_000_000,
        n_info_hamming=30_000_000,
        n_info_convolutional=1000 * 10_000,     # 1000 пакетов по 10 000 бит
        packet_length=10_000,
        chunk_bits=200_000,
        packets_per_batch=10,
        time_budget_s=120.0,
    )
    meta = {
        "seed": SEED,
        "grids": {k: list(v) for k, v in GRIDS.items()},
        "n_info": {"uncoded": plan.n_info_uncoded,
                   "hamming": plan.n_info_hamming,
                   "convolutional": plan.n_info_convolutional},
        "packets": plan.n_info_convolutional // plan.packet_length,
        "packet_length": plan.packet_length,
        "r_eff": plan.r_eff,
        "streams": "numpy.random.SeedSequence(entropy=[seed, scheme_tag, scheme_id, "
                   "chunk_index, purpose]), purpose=1 биты / 2 шум; "
                   "bootstrap использует отдельный генератор с другим seed",
        "fixed_before_run": True,
        "early_stopping": False,
    }
    print("Согласованный уточняющий прогон")
    print(f"  seed = {SEED}")
    for scheme, grid in GRIDS.items():
        n = plan.n_info(scheme)
        extra = ""
        if scheme == "convolutional":
            extra = f" ({n // plan.packet_length} пакетов по {plan.packet_length})"
        elif scheme == "hamming":
            extra = f" ({n // 4:,} блоков)"
        print(f"  {scheme:14s} {grid[0]:.1f}…{grid[-1]:.1f} дБ, шаг 0.2: N = {n:,}{extra}")

    os.makedirs(DATA, exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    t0 = time.perf_counter()
    results = sim.run_experiment(plan, verbose=True, dataset="refinement",
                                 ebno_by_scheme=GRIDS)
    elapsed = time.perf_counter() - t0
    path = sim.write_csv(results, OUT_CSV)
    print(f"\nУточняющий прогон: {elapsed:.1f} c, записано {path}")
    print(f"план зафиксирован: {OUT_JSON}")


if __name__ == "__main__":
    main()
