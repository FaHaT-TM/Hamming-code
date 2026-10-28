"""Анализ уточняющего прогона: кластерные интервалы, bootstrap, сеточная ошибка процедуры.

По спецификации (§5–§6): bootstrap 10 000 реплик с отдельным seed,
пересечение ищется по всей локальной сетке, число и доля невалидных реплик
сохраняются; сеточная ошибка процедуры оценивается отдельно на точных
теоретических кривых.

Пишет:
    results/data/refinement_gains.csv
    results/data/refinement_cluster.csv
    документы/отчёты/Уточняющий_прогон.md
"""

from __future__ import annotations

import csv
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import metrics, plotting  # noqa: E402

DATA = os.path.join(ROOT, "results", "data")
N_BOOT = 10_000
BOOT_SEED = 777_001          # отдельный генератор, не расходует поток канала
TARGET = 1e-4
NAMES = {"uncoded": "без кодирования", "hamming": "Хэмминг (7,4)",
         "convolutional": "свёрточный (7,5)₈ + Витерби"}


def load(name: str):
    path = os.path.join(DATA, name)
    if not os.path.exists(path):
        return None
    return plotting.read_csv(path)


def point_curves(rows):
    return {s: {r["ebno_db"]: r["ber"] for r in rows
                if r["scheme"] == s and r["ber"] > 0}
            for s in NAMES}


def main() -> None:
    ref = load("ber_refinement.csv")
    ext = load("ber_results_extended.csv")
    if ref is None:
        raise SystemExit("нет результатов уточняющего прогона")

    ells = {r["scheme"]: int(r["ell"]) for r in ref}
    hists = {s: {r["ebno_db"]: metrics.hist_from_str(r["z_hist"])
                 for r in ref if r["scheme"] == s} for s in NAMES}

    # ---- точечные оценки и кластерные интервалы ----
    cluster_rows = []
    for r in ref:
        st = metrics.cluster_normal_from_hist(metrics.hist_from_str(r["z_hist"]),
                                              int(r["ell"]))
        cluster_rows.append({
            "scheme": r["scheme"], "ebno_db": r["ebno_db"],
            "N_info": int(r["n_info_bits"]), "N_tx": int(r["n_tx_bits"]),
            "E": int(r["n_errors"]), "M": int(r["n_units"]),
            "K": int(r["n_units_with_errors"]), "ell": int(r["ell"]),
            "sum_z2": r["sum_z2"], "ber": r["ber"],
            "mean_W": st["mean_W"], "SE": st["se"],
            "cluster_ci_low": st["ci_low"], "cluster_ci_high": st["ci_high"],
            "cluster_applicable": int(st["applicable"]),
            "conservative_ci_low": r["ci_low"], "conservative_ci_high": r["ci_high"],
            "sinr_sigma2": r["sigma2"], "r_eff": r["r_eff"],
            "seed": int(r["seed"]), "status": r["status"],
        })
    with open(os.path.join(DATA, "refinement_cluster.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(cluster_rows[0].keys()))
        w.writeheader()
        w.writerows(cluster_rows)

    # ---- bootstrap по уточняющей сетке ----
    pcurves = point_curves(ref)
    ref_pairs = {}
    for scheme in ("hamming", "convolutional"):
        found = metrics._find_crossing_pair(pcurves[scheme], TARGET)
        ref_pairs[scheme] = found[1] if found else None
    boot_ref = metrics.bootstrap_gain(hists, ells, TARGET, n_boot=N_BOOT,
                                      seed=BOOT_SEED, reference_pairs=ref_pairs)

    # ---- bootstrap по расширенной сетке 1 дБ (для сравнения) ----
    boot_ext = None
    if ext is not None:
        ells_ext = {r["scheme"]: int(r["ell"]) for r in ext}
        hists_ext = {s: {r["ebno_db"]: metrics.hist_from_str(r["z_hist"])
                         for r in ext if r["scheme"] == s} for s in NAMES}
        pairs_ext = {}
        for scheme in ("hamming", "convolutional"):
            found = metrics._find_crossing_pair(point_curves(ext)[scheme], TARGET)
            pairs_ext[scheme] = found[1] if found else None
        boot_ext = metrics.bootstrap_gain(hists_ext, ells_ext, TARGET, n_boot=N_BOOT,
                                          seed=BOOT_SEED, reference_pairs=pairs_ext)

    # ---- сеточная ошибка процедуры на точных кривых ----
    grid_err = metrics.grid_error_report(TARGET)

    # ---- запись таблицы выигрышей ----
    gain_rows = []
    for dataset, rows, boot, pairs in (
            ("refinement_0p2db", ref, boot_ref, ref_pairs),
            ("extended_1db", ext, boot_ext, None)):
        if rows is None:
            continue
        pc = point_curves(rows)
        for scheme in ("hamming", "convolutional"):
            res = metrics.energy_gain_db(pc["uncoded"], pc[scheme], TARGET)
            b = (boot or {}).get(scheme, {})
            gain_rows.append({
                "dataset": dataset, "scheme": scheme, "target_ber": TARGET,
                "x_uncoded_db": res["x_uncoded"], "x_coded_db": res["x_coded"],
                "gain_point_db": res["gain_db"],
                "gain_ci_low_db": b.get("ci_low"), "gain_ci_high_db": b.get("ci_high"),
                "bootstrap_n": b.get("n_boot"), "bootstrap_valid": b.get("n_valid"),
                "invalid_fraction": b.get("invalid_fraction"),
                "pair_changed_fraction": b.get("pair_changed_fraction"),
                "bootstrap_sd": b.get("bootstrap_sd"), "bootstrap_seed": BOOT_SEED,
            })
    with open(os.path.join(DATA, "refinement_gains.csv"), "w", newline="",
              encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(gain_rows[0].keys()))
        w.writeheader()
        w.writerows(gain_rows)

    # ---- отчёт ----
    L = []

    def w(t=""):
        L.append(t)

    w("# Уточняющий прогон (шаг 0.2 дБ): результаты")
    w()
    w("Это локальное уточнение окрестности BER = 10⁻⁴, набор данных отдельный.")
    w()
    w("Протокол зафиксирован до запуска (§4 спецификации): объёмы заданы заранее, остановки "
      "по числу ошибок нет. С прежними прогонами результаты **не объединяются** — основной "
      "оценкой остаётся то, что посчитано по своим наборам.")
    w()
    plan = json.load(open(os.path.join(DATA, "refinement_plan.json"), encoding="utf-8"))
    w(f"* seed уточнения: **{plan['seed']}** (ранее не использовался);")
    w(f"* объёмы: без кода и Хэмминг по {plan['n_info']['uncoded']:,} информационных "
      f"бит на точку, Витерби — {plan['packets']} пакетов по {plan['packet_length']} "
      f"({plan['n_info']['convolutional']:,} бит);")
    w(f"* bootstrap: {N_BOOT} реплик, отдельный seed **{BOOT_SEED}**;")
    w("* потоки: `SeedSequence(entropy=[seed, scheme_tag, scheme_id, chunk_index, purpose])`, "
      "purpose = 1 (биты) / 2 (шум); перезапуска одним seed на каждом пакете нет.")
    w()
    w("## 1. Точки уточняющей сетки")
    w()
    w("| Схема | $E_b/N_0$ | $N_{info}$ | $N_{tx}$ | E | BER | M | K | SE | кластерный 95 % |")
    w("|---|---:|---:|---:|---:|---|---:|---:|---|---|")
    for r in cluster_rows:
        ci = (f"[{r['cluster_ci_low']:.3e}, {r['cluster_ci_high']:.3e}]"
              if r["cluster_applicable"] else "не применим (K < 30)")
        w(f"| {NAMES[r['scheme']]} | {r['ebno_db']:.1f} | {r['N_info']:,} "
          f"| {r['N_tx']:,} | {r['E']:,} | {r['ber']:.3e} | {r['M']:,} | {r['K']:,} "
          f"| {r['SE']:.2e} | {ci} |")
    w()
    w("## 2. Энергетический выигрыш при BER = 10⁻⁴")
    w()
    w("| Набор | Схема | $E_b/N_0$ без кода | $E_b/N_0$ кодированной | Выигрыш | "
      "bootstrap 95 % | доля невалидных реплик | смена пары точек |")
    w("|---|---|---:|---:|---:|---|---:|---:|")
    for g in gain_rows:
        ci = (f"[{g['gain_ci_low_db']:+.3f}; {g['gain_ci_high_db']:+.3f}]"
              if g["gain_ci_low_db"] is not None else "—")
        inv = ("—" if g["invalid_fraction"] is None
               else f"{g['invalid_fraction'] * 100:.1f} %")
        pair = ("—" if g["pair_changed_fraction"] is None
                else f"{g['pair_changed_fraction'] * 100:.1f} %")
        w(f"| {g['dataset']} | {NAMES[g['scheme']]} | {g['x_uncoded_db']:.2f} "
          f"| {g['x_coded_db']:.2f} | {g['gain_point_db']:+.2f} | {ci} | {inv} | {pair} |")
    w()
    w("Bootstrap-интервал — percentile по квантилям 2.5 % и 97.5 % **валидных** реплик. "
      "Реплики, в которых пересечения не нашлось, не экстраполируются и молча не "
      "выбрасываются: их число и доля приведены в таблице. Если невалидных реплик заметно, "
      "интервал условен.")
    w()
    w("## 3. Насколько виновата сама сетка (проверка на точных теоретических кривых)")
    w()
    w("| Кривая | Точный корень для $10^{-4}$, дБ | Интерполяция на сетке 1 дБ | "
      "Отклонение | Интерполяция на сетке 0.2 дБ | Отклонение |")
    w("|---|---:|---:|---:|---:|---:|")
    for name, label in (("bpsk", "BPSK без кодирования"), ("hamming", "Хэмминг (7,4)")):
        v = grid_err[name]
        w(f"| {label} | {v['exact_db']:.4f} | {v['grid_1db_db']:.4f} "
          f"| {v['grid_error_1db']:+.4f} | {v['grid_0p2db_db']:.4f} "
          f"| {v['grid_error_0p2db']:+.5f} |")
    w()
    w("Это характеристика **процедуры интерполяции**, а не экспериментальная оценка: "
      "на гладких теоретических кривых сетка 1 дБ даёт смещение порядка сотых долей дБ, "
      "а сетка 0.2 дБ — на порядки меньше. Это отдельный источник: с bootstrap-интервалом "
      "он не складывается.")
    w()
    w("## 4. Сопоставление с расширенным прогоном по сетке 1 дБ")
    w()
    if boot_ext:
        w("| Схема | Сетка 1 дБ (расширенный прогон) | Уточнённая сетка 0.2 дБ | Разница |")
        w("|---|---:|---:|---:|")
        by = {g["scheme"]: g for g in gain_rows if g["dataset"] == "extended_1db"}
        rf = {g["scheme"]: g for g in gain_rows if g["dataset"] == "refinement_0p2db"}
        for scheme in ("hamming", "convolutional"):
            if scheme in by and scheme in rf:
                d = rf[scheme]["gain_point_db"] - by[scheme]["gain_point_db"]
                w(f"| {NAMES[scheme]} | {by[scheme]['gain_point_db']:+.2f} дБ "
                  f"| {rf[scheme]['gain_point_db']:+.2f} дБ | {d:+.2f} дБ |")
        w()
        w("Расхождение показывает вклад сетки и независимого шума Монте-Карло. "
          "Оба набора лежат раздельно и не объединяются.")
    w()
    w("## 5. Файлы")
    w()
    w("- `results/data/ber_refinement.csv` — точки уточняющей сетки;")
    w("- `results/data/refinement_cluster.csv` — кластерные и консервативные интервалы;")
    w("- `results/data/refinement_gains.csv` — выигрыши и bootstrap-диагностика;")
    w("- `results/data/refinement_plan.json` — зафиксированный до запуска протокол.")

    out = os.path.join(ROOT, "документы", "отчёты", "Уточняющий_прогон.md")
    with open(out, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print("записано:", out)
    print("записано:", os.path.join(DATA, "refinement_gains.csv"))
    for g in gain_rows:
        print(f"{g['dataset']:18s} {g['scheme']:14s} gain={g['gain_point_db']:+.3f} "
              f"CI=[{g['gain_ci_low_db']:+.3f}; {g['gain_ci_high_db']:+.3f}] "
              f"invalid={g['invalid_fraction']:.3f} pair_changed={g['pair_changed_fraction']:.3f}")


if __name__ == "__main__":
    main()
