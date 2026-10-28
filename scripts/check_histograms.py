"""Проверка тождеств гистограмм Z_j для всех сохранённых строк.

Проверяются четыре тождества для каждой строки CSV:

    Σ_z h_z = M,    Σ_z z·h_z = E,    M − h_0 = K,    Σ_z z²·h_z = sum_z2.

Пишет results/data/histogram_checks.json и документы/отчёты/Проверка_гистограмм.md.
Запуск: python scripts/check_histograms.py (только сохранённые CSV, без моделирования)
"""

from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import metrics  # noqa: E402

DATA = os.path.join(ROOT, "results", "data")
JSON_OUT = os.path.join(DATA, "histogram_checks.json")
MD_OUT = os.path.join(ROOT, "документы", "отчёты", "Проверка_гистограмм.md")


def check_file(path: str, dataset: str) -> list[dict]:
    out = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            hist = metrics.hist_from_str(row["z_hist"])
            m = sum(hist.values())
            e = sum(z * c for z, c in hist.items())
            s2 = sum(z * z * c for z, c in hist.items())
            m_csv = int(row["n_units"])
            e_csv = int(row["n_errors"])
            k_csv = int(row["n_units_with_errors"])
            s2_csv = float(row["sum_z2"])
            h0 = hist.get(0, 0)
            checks = {
                "sum_h_equals_M": m == m_csv,
                "sum_zh_equals_E": e == e_csv,
                "M_minus_h0_equals_K": (m - h0) == k_csv,
                "sum_z2h_equals_sum_z2": abs(s2 - s2_csv) < 1e-9,
            }
            out.append({
                "dataset": dataset, "scheme": row["scheme"],
                "ebno_db": float(row["ebno_db"]),
                "M": m_csv, "E": e_csv, "K": k_csv, "sum_z2": s2_csv,
                "hist_M": m, "hist_E": e, "hist_K": m - h0, "hist_sum_z2": s2,
                "all_ok": all(checks.values()), **checks,
            })
    return out


def main() -> None:
    rows = []
    rows += check_file(os.path.join(DATA, "ber_results.csv"), "main")
    rows += check_file(os.path.join(DATA, "ber_results_extended.csv"), "extended")
    refined = os.path.join(DATA, "ber_refinement.csv")
    if os.path.exists(refined):
        rows += check_file(refined, "refinement")

    payload = {"rows": rows,
               "n_rows": len(rows),
               "n_failed": sum(1 for r in rows if not r["all_ok"])}
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    L = ["# Проверка тождеств гистограмм $Z_j$", "",
         "Проверяю, что сохранённые счётчики согласованы между собой.", "",
         "Файл собирает `scripts/check_histograms.py` по всем строкам сохранённых CSV.", "",
         "Тождества такие: $\\sum_z h_z = M$, $\\sum_z z\\,h_z = E$, "
         "$M - h_0 = K$, $\\sum_z z^2 h_z = \\sum Z_j^2$.", "",
         f"**Строк проверено:** {len(rows)}; **с расхождением:** "
         f"{payload['n_failed']}.", ""]
    bad = [r for r in rows if not r["all_ok"]]
    if bad:
        L += ["| набор | схема | Eb/N0 | какое тождество нарушено |", "|---|---|---:|---|"]
        for r in bad:
            failed = [k for k in ("sum_h_equals_M", "sum_zh_equals_E",
                                  "M_minus_h0_equals_K", "sum_z2h_equals_sum_z2")
                      if not r[k]]
            L.append(f"| {r['dataset']} | {r['scheme']} | {r['ebno_db']:.1f} | "
                     f"{', '.join(failed)} |")
        L.append("")
    else:
        L += ["Все тождества сошлись во всех строках.", ""]

    L += ["## Контрольная строка: Витерби, расширенный прогон, 7 дБ", ""]
    ctrl = next((r for r in rows if r["dataset"] == "extended"
                 and r["scheme"] == "convolutional" and r["ebno_db"] == 7.0), None)
    if ctrl:
        L += ["| величина | значение |", "|---|---:|",
              f"| $M$ | {ctrl['M']} |", f"| $E$ | {ctrl['E']} |",
              f"| $K$ | {ctrl['K']} |", f"| $\\sum Z_j^2$ | {ctrl['sum_z2']:.0f} |",
              f"| $\\sum_z h_z$ | {ctrl['hist_M']} |",
              f"| $\\sum_z z h_z$ | {ctrl['hist_E']} |",
              f"| $M - h_0$ | {ctrl['hist_K']} |",
              f"| $\\sum_z z^2 h_z$ | {ctrl['hist_sum_z2']:.0f} |", ""]
        L += [f"**K = {ctrl['K']}** (значение сверено с сохранёнными данными).", ""]
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"записано: {JSON_OUT}")
    print(f"записано: {MD_OUT}")
    print(f"строк: {len(rows)}, с расхождением: {payload['n_failed']}")
    if ctrl:
        print("контрольная строка (conv 7 дБ, extended):", ctrl["K"], "K,", ctrl["E"], "E")


if __name__ == "__main__":
    main()
