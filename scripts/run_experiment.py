"""Полный эксперимент: основной (фиксированный) план + расширенный объём.

Запуск из корня проекта::

    python scripts/run_experiment.py            # основной + расширенный
    python scripts/run_experiment.py --main     # только основной (быстро)
    python scripts/run_experiment.py --probe    # только замер производительности

Полный эксперимент запускается явно, а не при открытии ноутбука.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import simulation as sim  # noqa: E402

DATA = os.path.join(ROOT, "results", "data")


def main() -> None:
    parser = argparse.ArgumentParser(description="BER-эксперимент")
    parser.add_argument("--main", action="store_true", help="только основной план")
    parser.add_argument("--probe", action="store_true", help="только замер производительности")
    parser.add_argument("--no-supp", action="store_true", help="без расширенного объёма")
    args = parser.parse_args()

    if args.probe:
        probe = sim.performance_probe()
        print(json.dumps(probe, indent=2, ensure_ascii=False))
        with open(os.path.join(DATA, "performance_probe.json"), "w", encoding="utf-8") as f:
            json.dump(probe, f, indent=2, ensure_ascii=False)
        return

    # ---------------- основной фиксированный план ----------------
    plan = sim.Plan()
    print(f"Основной план: seed={plan.seed}, Eb/N0 = {plan.ebno_db[0]:.0f}…"
          f"{plan.ebno_db[-1]:.0f} дБ, шаг 1 дБ")
    print(f"  без кодирования: N = {plan.n_info_uncoded:,} бит")
    print(f"  Хэмминг (7,4):   N = {plan.n_info_hamming:,} бит "
          f"({plan.n_info_hamming // 4:,} блоков)")
    print(f"  свёрточный:      N = {plan.n_info_convolutional:,} бит "
          f"({plan.n_info_convolutional // plan.packet_length} пакетов по "
          f"{plan.packet_length})")
    t0 = time.perf_counter()
    results = sim.run_experiment(plan)
    elapsed = time.perf_counter() - t0
    path = sim.write_csv(results, os.path.join(DATA, "ber_results.csv"))
    print(f"\nОсновной прогон: {elapsed:.1f} c, записано {path}")

    sim.write_theory_csv(os.path.join(DATA, "theory_curves.csv"))

    if args.main or args.no_supp:
        return

    # ---------------- расширенный объём (отдельный CSV) ----------------
    supp_plan = sim.Plan(
        seed=plan.seed + 1,                     # независимый поток, seed фиксирован заранее
        n_info_uncoded=20_000_000,
        n_info_hamming=20_000_000,
        n_info_convolutional=10_000_000,
        packets_per_batch=10,
    )
    print("\nРасширенный объём (дополнительный прогон, отдельный CSV):")
    print(f"  без кодирования: N = {supp_plan.n_info_uncoded:,} бит")
    print(f"  Хэмминг (7,4):   N = {supp_plan.n_info_hamming:,} бит")
    print(f"  свёрточный:      N = {supp_plan.n_info_convolutional:,} бит")
    t0 = time.perf_counter()
    supp = sim.run_experiment(supp_plan)
    elapsed = time.perf_counter() - t0
    path = sim.write_csv(supp, os.path.join(DATA, "ber_results_extended.csv"))
    print(f"\nРасширенный прогон: {elapsed:.1f} c, записано {path}")


if __name__ == "__main__":
    main()
