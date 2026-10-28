"""Основной отчёт по результатам (кластерная неопределённость + bootstrap, только сохранённые данные).

Читает только сохранённые CSV и JSON и генерирует:

    документы/отчёты/Отчёт_по_результатам.md
    results/data/energy_gain.csv
    results/data/cluster_intervals.csv

Никакого нового моделирования здесь не выполняется: пересчёт идёт по сохранённым
гистограммам ``Z_j`` (столбец ``z_hist`` в CSV).
"""

from __future__ import annotations

import json
import os
import platform
import sys
from datetime import datetime

import matplotlib
import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import channel, convolutional as conv  # noqa: E402
from src import hamming, metrics, plotting, simulation  # noqa: E402

DATA = os.path.join(ROOT, "results", "data")
OUT_MD = os.path.join(ROOT, "документы", "отчёты", "Отчёт_по_результатам.md")

SCHEMES = ("uncoded", "hamming", "convolutional")
NAMES = {
    "uncoded": "без кодирования",
    "hamming": "Хэмминг (7,4)",
    "convolutional": "свёрточный (7,5)₈ + Витерби",
}
SHORT = {"uncoded": "без кода", "hamming": "Хэмминг", "convolutional": "Витерби"}
TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)
BOOTSTRAP_N = 10_000
BOOTSTRAP_SEED = 777_001      # отдельный генератор, не расходует поток канала


def fmt(value, digits=3):
    if value is None or (isinstance(value, float) and not np.isfinite(value)):
        return "—"
    if value == 0:
        return "0"
    return f"{value:.{digits}e}"


def curves_of(rows):
    """{scheme: {ebno: hist}} — гистограммы Z_j по точкам."""
    out = {s: {} for s in SCHEMES}
    for r in rows:
        out[r["scheme"]][r["ebno_db"]] = metrics.hist_from_str(r["z_hist"])
    return out


def ell_of(rows):
    return {r["scheme"]: int(r["ell"]) for r in rows}


def point_curves(rows):
    """{scheme: {ebno: BER}} только для точек с E > 0 (для интерполяции)."""
    out = {s: {} for s in SCHEMES}
    for r in rows:
        if r["ber"] > 0:
            out[r["scheme"]][r["ebno_db"]] = r["ber"]
    return out



def linear_ber_crossing(x1: float, p1: float, x2: float, p2: float, target: float):
    """Пересечение уровня при линейной интерполяции по BER (альтернативный способ)."""
    if p1 <= 0 or p2 <= 0 or (p1 - target) * (p2 - target) > 0:
        return None
    if p1 == p2:
        return None
    return x1 + (x2 - x1) * (target - p1) / (p2 - p1)


def interpolation_sensitivity(rows, target=1e-4):
    """Насколько выигрыш зависит от способа интерполяции (оценка погрешности сетки)."""
    pcurves = point_curves(rows)
    out = {}
    for scheme in ("hamming", "convolutional"):
        res_log = metrics.energy_gain_db(pcurves["uncoded"], pcurves[scheme], target)
        def find(curve, fn):
            xs = sorted(curve)
            for a, b in zip(xs, xs[1:]):
                x = fn(a, curve[a], b, curve[b], target)
                if x is not None:
                    return x
            return None
        xu = find(pcurves["uncoded"], linear_ber_crossing)
        xc = find(pcurves[scheme], linear_ber_crossing)
        gain_lin = None if (xu is None or xc is None) else xu - xc
        out[scheme] = {"gain_log": res_log["gain_db"], "gain_linear": gain_lin,
                       "delta": (None if (gain_lin is None or res_log["gain_db"] is None)
                                 else gain_lin - res_log["gain_db"])}
    return out


def quality_label(gain, boot, e_ok, k_ok):
    """Честная маркировка качества оценки выигрыша."""
    if gain is None:
        return "н/д", "нет двух надёжных положительных соседних оценок для интерполяции"
    parts = ["точечная оценка"]
    if boot is not None and np.isfinite(boot.get("ci_low", np.nan)):
        parts.append(f"bootstrap 95 % [{boot['ci_low']:+.2f}; {boot['ci_high']:+.2f}] дБ")
    else:
        parts.append("неопределённость не оценена")
    note = []
    if not e_ok:
        note.append("E < 100 у одной из соседних точек")
    if not k_ok:
        note.append("K < 30 независимых ошибочных единиц")
    note.append("погрешность сетки 1 дБ не устранена")
    return "; ".join(parts), "; ".join(note)


def pair_quality(rows, scheme, target):
    """Диагностические пороги E ≥ 100 и K ≥ 30 для пары точек, охватывающей уровень."""
    curve = {r["ebno_db"]: r for r in rows if r["scheme"] == scheme}
    xs = sorted(x for x, r in curve.items() if r["ber"] > 0)
    for a, b in zip(xs, xs[1:]):
        if metrics.crossing_ebn0(a, curve[a]["ber"], b, curve[b]["ber"], target) is not None:
            pa, pb = curve[a], curve[b]
            return (min(pa["n_errors"], pb["n_errors"]) >= 100,
                    min(pa["n_units_with_errors"], pb["n_units_with_errors"]) >= 30,
                    (a, b))
    return False, False, (None, None)


def gain_block(rows, label, out_md, out_csv, dataset):
    """Таблица выигрышей: точечная оценка, bootstrap CI, диагностика, качество."""
    pcurves = point_curves(rows)
    hcurves = curves_of(rows)
    ells = ell_of(rows)
    ref_pairs = {}
    for scheme in ("hamming", "convolutional"):
        found = metrics._find_crossing_pair(pcurves[scheme], 1e-4)
        ref_pairs[scheme] = found[1] if found else None
    boot = metrics.bootstrap_gain(hcurves, ells, target=1e-4, n_boot=BOOTSTRAP_N,
                                  seed=BOOTSTRAP_SEED, reference_pairs=ref_pairs)
    out_md.append(f"### {label}")
    out_md.append("")
    out_md.append("| Схема | BER* | $E_b/N_0$ без кода, дБ | $E_b/N_0$ кодированной, дБ "
                  "| Выигрыш, дБ | Качество оценки | Диагностика |")
    out_md.append("|---|---:|---:|---:|---:|---|---|")
    for scheme in ("hamming", "convolutional"):
        for target in TARGETS:
            res = metrics.energy_gain_db(pcurves["uncoded"], pcurves[scheme], target)
            e_ok, k_ok, pair = pair_quality(rows, scheme, target)
            b = boot.get(scheme) if target == 1e-4 else None
            label_q, note = quality_label(res["gain_db"], b, e_ok, k_ok)
            xu = "—" if res["x_uncoded"] is None else f"{res['x_uncoded']:.2f}"
            xc = "—" if res["x_coded"] is None else f"{res['x_coded']:.2f}"
            g = "н/д" if res["gain_db"] is None else f"{res['gain_db']:+.2f}"
            out_md.append(f"| {NAMES[scheme]} | {target:.0e} | {xu} | {xc} | {g} "
                          f"| {label_q} | {note} |")
            out_csv.append({
                "dataset": dataset, "scheme": scheme, "target_ber": target,
                "x_uncoded_db": "" if res["x_uncoded"] is None else res["x_uncoded"],
                "x_coded_db": "" if res["x_coded"] is None else res["x_coded"],
                "gain_point_db": "" if res["gain_db"] is None else res["gain_db"],
                "gain_ci_low_db": "" if b is None else b["ci_low"],
                "gain_ci_high_db": "" if b is None else b["ci_high"],
                "bootstrap_n": "" if b is None else b["n_boot"],
                "bootstrap_valid": "" if b is None else b["n_valid"],
                "invalid_fraction": "" if b is None else b["invalid_fraction"],
                "pair_changed_fraction": "" if b is None else b["pair_changed_fraction"],
                "e_ge_100": int(e_ok), "k_ge_30": int(k_ok),
                "quality": label_q, "note": note,
            })
    out_md.append("")
    out_md.append("Выигрыш — это горизонтальное расстояние между кривыми при одном и том же "
                  "BER. Интерполяция линейная по $\\lg(\\mathrm{BER})$ между соседними "
                  "точками сетки 1 дБ, без экстраполяции. Bootstrap **приближённый** "
                  "кластерный: внутри каждой точки независимые единицы (биты, блоки, "
                  "пакеты) заново выбираются с возвращением и оба пересечения считаются по "
                  "новой выборке. Погрешность сетки он не устраняет и проблему редких "
                  "событий не решает.")
    out_md.append("")
    return boot


def cluster_table(rows, out_md):
    out_md.append("| Схема | $E_b/N_0$ | E | M | K | $\\overline{W}$ | SE | "
                  "кластерный 95 % (приближ.) | консервативный 95 % |")
    out_md.append("|---|---:|---:|---:|---:|---:|---:|---|---|")
    for scheme in SCHEMES:
        for r in sorted((r for r in rows if r["scheme"] == scheme),
                        key=lambda r: r["ebno_db"]):
            hist = metrics.hist_from_str(r["z_hist"])
            st = metrics.cluster_normal_from_hist(hist, int(r["ell"]))
            if st["applicable"]:
                cl = f"[{fmt(st['ci_low'])}, {fmt(st['ci_high'])}]"
            else:
                cl = "не применяется (K < 30)"
            cons = (f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ber"] > 0
                    else f"≤ {fmt(r['upper95'])}")
            out_md.append(
                f"| {SHORT[scheme]} | {r['ebno_db']:.0f} | {int(r['n_errors']):,} "
                f"| {int(r['n_units']):,} | {int(r['n_units_with_errors']):,} "
                f"| {st['mean_W']:.3e} | {st['se']:.2e} | {cl} | {cons} |")
    out_md.append("")


def main() -> None:
    main_rows = plotting.read_csv(os.path.join(DATA, "ber_results.csv"))
    supp_path = os.path.join(DATA, "ber_results_extended.csv")
    supp_rows = plotting.read_csv(supp_path) if os.path.exists(supp_path) else []
    vit_path = os.path.join(DATA, "viterbi_verification.json")
    vit = json.load(open(vit_path, encoding="utf-8")) if os.path.exists(vit_path) else None

    L: list[str] = []

    def w(text: str = "") -> None:
        L.append(text)

    w("# Отчёт по результатам исследования")
    w()
    w(f"**Сформирован:** {datetime.now().strftime('%Y-%m-%d %H:%M')}  ")
    w(f"**Окружение:** Python {platform.python_version()}, NumPy {np.__version__}, "
      f"Matplotlib {matplotlib.__version__}, {platform.system()} {platform.machine()}, "
      f"{os.cpu_count()} vCPU  ")
    w("Здесь собрано всё по трём схемам: точки BER, гистограммы $Z_j$ из CSV, кластерные "
      "SE и интервалы, bootstrap-неопределённость выигрыша при BER = 10⁻⁴ и проверки "
      "декодера Витерби.")
    w()
    w("Отчёт сводный: все три схемы, диапазон Eb/N0 = 0…10 дБ.")
    w()
    w("Числа взяты из сохранённых CSV и JSON, заново ничего не моделировалось. Таблицы "
      "собирает скрипт `scripts/make_report_tables.py`.")
    w()

    # ---------------- 1. План, независимость, честность объединения ----------------
    plan = simulation.Plan()
    w("## 1. План эксперимента: потоки случайности и раздельные прогоны")
    w()
    w("| Схема | $N_{info}$, бит | Единица независимости ℓ | $M$ | $R_{eff}$ |")
    w("|---|---:|---:|---:|---:|")
    r_eff = plan.r_eff
    w(f"| без кодирования | {plan.n_info_uncoded:,} | 1 (бит) | {plan.n_info_uncoded:,} "
      f"| {r_eff['uncoded']:.5f} |")
    w(f"| Хэмминг (7,4) | {plan.n_info_hamming:,} | 4 (блок) "
      f"| {plan.n_info_hamming // 4:,} | {r_eff['hamming']:.5f} |")
    w(f"| свёрточный | {plan.n_info_convolutional:,} | {plan.packet_length} (пакет) "
      f"| {plan.n_info_convolutional // plan.packet_length} | "
      f"{r_eff['convolutional']:.5f} |")
    w()
    w("**Что здесь важно:**")
    w()
    w(f"1. Объём расширенного прогона я зафиксировал **до** его старта "
      f"(20·10⁶ / 20·10⁶ / 10·10⁶ бит, seed = {plan.seed + 1}) и по результатам не менял.")
    w("2. Замер производительности в итоговую статистику **не входит**: он делается "
      "отдельно (`--probe`) и на уменьшенном объёме.")
    w("3. Основной и расширенный прогоны **не объединялись**: это два разных CSV с разными "
      "seed. Объём фиксирован, остановки по числу ошибок нет, то есть ничего адаптивного "
      "в плане не было.")
    w("4. Потоки случайности: `numpy.random.SeedSequence(seed, теги)` даёт свой поток на "
      "каждую комбинацию (схема, точка, порция, назначение: биты или шум). Общих "
      "случайных чисел между схемами **нет**, поэтому bootstrap ресэмплирует схемы и "
      "точки независимо — парность для этого не нужна.")
    w("5. Что биты и шум не пересекаются и что порядок порций не важен, проверяет тест "
      "`test_streams_are_independent_of_order`: результат точки не зависит от того, "
      "какие ещё точки считались.")
    w()

    # ---------------- 1.1. Исправления по итогам проверки ----------------
    w("### 1.1. Уточнения по сохранённым данным")
    w()
    conv7 = next(r for r in supp_rows
                 if r["scheme"] == "convolutional" and r["ebno_db"] == 7.0)
    w(f"1. **Число ошибочных пакетов при 7 дБ (расширенный прогон).** В сохранённых "
      f"данных $M = {int(conv7['n_units'])}$, $E = {int(conv7['n_errors'])}$, "
      f"$K = {int(conv7['n_units_with_errors'])}$, $\\sum Z_j^2 = {conv7['sum_z2']:.0f}$. "
      "Значение $K$ сверено с сохранёнными данными: сами данные не менялись, "
      "общие $E$ и BER совпадают. "
      "Все четыре тождества гистограммы проверены для 81 строки всех трёх наборов, "
      "расхождений нет (`документы/отчёты/Проверка_гистограмм.md`).")
    w("2. **Зачем нужна гистограмма.** Для кластерных SE хватает "
      "$M$, $E$, $\\ell$, $\\sum Z_j^2$; сама гистограмма нужна только для "
      "непараметрического bootstrap.")
    w("3. **Повторный прогон с тем же seed.** Это "
      "**реконструкция отсутствующей статистики и проверка воспроизводимости**, "
      "а не новая независимая выборка: объём доказательств она не увеличивает. "
      "Код генерации и декодирования, параметры потоков и порядок обращений к "
      "генераторам не менялись; все точечные оценки BER совпали с прежними "
      "до последнего знака.")
    w("4. **Как называть 0.10 и 0.19 дБ.** Это «разница между способами интерполяции», "
      "диагностика чувствительности к методу. Доверительной границей она не является и "
      "с bootstrap-интервалом не складывается. "
      "Сеточная ошибка самой процедуры оценена отдельно — на точных теоретических "
      "кривых (§7.3).")
    w()

    # ---------------- 2. Подтверждения ----------------
    w("## 2. Проверка реализации на контрольных примерах")
    w()
    w("### 2.1. Таблица переходов и контрольные последовательности кодера")
    w()
    w("| $S_t$ | $u$ | $v_1v_2$ | $S_{t+1}$ |")
    w("|---|---:|---|---|")
    for state, u, out_v, nxt in conv.TRANSITIONS:
        w(f"| {state:02b} | {u} | {out_v} | {nxt:02b} |")
    w()
    w("| Информация | Полный вход | Выход кодера | $N_{tx}$ | Конечное состояние |")
    w("|---|---|---|---:|---:|")
    for text in ("0", "1", "10", "11", "1011"):
        info = np.array([int(c) for c in text], dtype=np.uint8)
        code = conv.encode(info)
        w(f"| `{text}` | `{text}00` | `{''.join(map(str, code))}` | {code.size} "
          f"| {conv.finish_state(info):02b} |")
    w()
    w(f"Для $L={conv.L_DEFAULT}$: $T = {conv.L_DEFAULT + conv.TAIL}$, "
      f"$N_{{tx}} = {2 * (conv.L_DEFAULT + conv.TAIL)}$ на пакет, "
      f"$R_{{eff}} = {conv.rate_effective():.8f}$, надбавка за хвост "
      f"{conv.energy_penalty_db():.6f} дБ.")
    w()

    if vit:
        bf = vit["brute_force"]["total"]
        si = vit["single_inversions"]["total"]
        tie = vit["tie_break_example"]
        w("### 2.2. Витерби: что проверено дополнительно")
        w()
        w("Источник: `документы/отчёты/Проверка_декодера_Витерби.md`, "
          "`results/data/viterbi_verification.json` (посчитано `scripts/verify_viterbi.py`).")
        w()
        w("| L | случаев (полный перебор) | несовпадений **метрики** | единственный оптимум "
          "| несовпадений сообщения | неоднозначных |")
        w("|---:|---:|---:|---:|---:|---:|")
        for r in vit["brute_force"]["rows"]:
            w(f"| {r['L']} | {r['cases']} | {r['metric_mismatch']} "
              f"| {r['unique_optimum']} | {r['message_mismatch']} | {r['ambiguous']} |")
        w(f"| **итого** | **{bf['cases']}** | **{bf['metric_mismatch']}** "
          f"| **{bf['unique_optimum']}** | **{bf['message_mismatch']}** "
          f"| **{bf['ambiguous']}** |")
        w()
        w(f"* сравнение с полным перебором $2^L$: **{bf['cases']}** случаев "
          f"($L = 1\\ldots6$, ошибки в 0–3 позициях); несовпадений минимальной метрики — "
          f"**{bf['metric_mismatch']}**, несовпадений сообщения при единственном оптимуме — "
          f"**{bf['message_mismatch']}**. Неоднозначных оптимумов {bf['ambiguous']}: "
          "для них сравнивалась только метрика, так и договаривались;")
        w(f"* одиночные инверсии, полный перебор: **{si['cases']}** случаев, восстановлено "
          f"**{si['recovered']}**;")
        w(f"* пример с равными метриками построен специально: на входе `{tie['received']}`, "
          f"есть два равноправных пути `{tie['optimal_messages'][0]}` и "
          f"`{tie['optimal_messages'][1]}` с метриками {tie['metrics']}; декодер выбрал "
          f"`{tie['decoder_choice']}` — по правилу меньшего предыдущего состояния "
          f"(`EDGE_PREV` = {tie['edge_prev_table']});")
        w("* $d_{free}$: доказательство — по четырёхсостоятельной диаграмме, оно в "
          "теоретическом разделе; перебор до $L\\le12$ приведён только как вычислительная "
          "проверка на конечном диапазоне.")
        w()

    w("### 2.3. Канал: $R_{eff}$, $\\sigma^2$")
    w()
    w("| Схема | $\\sigma^2$ при 0 дБ | $\\sigma^2$ при 10 дБ |")
    w("|---|---:|---:|")
    for name, rate in (("без кодирования", 1.0), ("Хэмминг (7,4)", hamming.RATE),
                       ("свёрточный", conv.rate_effective())):
        w(f"| {name} | {channel.variance_from_ebno(0, rate):.6f} "
          f"| {channel.variance_from_ebno(10, rate):.6f} |")
    w()
    w("### 2.4. Профиль ошибок Хэмминга (перебор 128 шаблонов)")
    w()
    profile = metrics.hamming_weight_profile()
    w("| $w$ | " + " | ".join(str(i) for i in range(8)) + " |")
    w("|---|" + "---|" * 8)
    w("| $\\sum_e z(e)$ | " + " | ".join(str(int(v)) for v in profile) + " |")
    w()
    w(f"Совпадает со спецификацией: **{bool(np.array_equal(profile, np.array([0,0,36,76,64,48,28,4])))}**.")
    w()

    # ---------------- 3-4. Таблицы точек ----------------
    for label, rows in (("3. Основной прогон: точки", main_rows),
                        ("4. Расширенный объём: точки", supp_rows)):
        if not rows:
            continue
        w(f"## {label}")
        for scheme in SCHEMES:
            pts = sorted((r for r in rows if r["scheme"] == scheme),
                         key=lambda r: r["ebno_db"])
            first = pts[0]
            w()
            w(f"### {NAMES[scheme]} — $R_{{eff}}$ = {first['r_eff']:.5f}, "
              f"ℓ = {int(first['ell'])}")
            w()
            w("| $E_b/N_0$ | $N_{info}$ | $N_{tx}$ | E | BER | M | K | $\\sum Z_j^2$ "
              "| консервативный 95 % | статус |")
            w("|---:|---:|---:|---:|---|---:|---:|---:|---|---|")
            for r in pts:
                cons = (f"[{fmt(r['ci_low'])}, {fmt(r['ci_high'])}]" if r["ber"] > 0
                        else f"≤ {fmt(r['upper95'])}")
                w(f"| {r['ebno_db']:.0f} | {int(r['n_info_bits']):,} "
                  f"| {int(r['n_tx_bits']):,} | {int(r['n_errors']):,} | {fmt(r['ber'])} "
                  f"| {int(r['n_units']):,} | {int(r['n_units_with_errors']):,} "
                  f"| {r['sum_z2']:.0f} | {cons} | {r['status']} |")
        w()

    # ---------------- 5. Кластерные интервалы ----------------
    w("## 5. Кластерные интервалы")
    w()
    w("Считается из гистограмм $Z_j$ (столбец `z_hist` в CSV), заново ничего не "
      "моделировалось:")
    w()
    w("$$\\overline{W} = \\frac{E}{M\\ell},\\qquad "
      "s_W^2 = \\frac{\\sum_j Z_j^2/\\ell^2 - M\\overline{W}^2}{M-1},\\qquad "
      "SE = \\sqrt{s_W^2/M},\\qquad \\overline{W}\\pm 1.96\\,SE.$$")
    w()
    if supp_rows:
        w("### 5.1. Расширенный объём")
        w()
        cluster_table(supp_rows, L)
    w("### 5.2. Основной прогон")
    w()
    cluster_table(main_rows, L)
    w("Кластерный нормальный интервал **не применяется** при $K < 30$: когда событий "
      "мало, нормальная аппроксимация не обоснована. Консервативные интервалы идут "
      "отдельным видом оценки. Для Витерби они получаются широкими из-за самого метода, "
      "а не из-за кодера: конструкция $q/\\ell \\le P_b \\le q$ намеренно выбрасывает "
      "информацию о том, сколько ошибок было внутри пакета.")
    w()

    # ---------------- 6. Сверка с теорией ----------------
    w("## 6. Эксперимент против теории: сходится или нет")
    w()
    w("| $E_b/N_0$ | BER без кода (эксп.) | теория BPSK | $z$-отклонение | "
      "BER Хэмминга (эксп.) | точная теория | кластерный $z$ |")
    w("|---:|---|---|---:|---|---|---:|")
    for r in sorted((r for r in main_rows if r["scheme"] == "uncoded"),
                    key=lambda r: r["ebno_db"]):
        x = r["ebno_db"]
        th_b = metrics.bpsk_ber_theory(x)
        se_b = np.sqrt(th_b * (1 - th_b) / r["n_info_bits"])
        z_b = (r["ber"] - th_b) / se_b if se_b > 0 else float("nan")
        ham = next(g for g in main_rows if g["scheme"] == "hamming" and g["ebno_db"] == x)
        th_h = metrics.hamming_ber_theory(x)
        st = metrics.cluster_normal_from_hist(metrics.hist_from_str(ham["z_hist"]),
                                              int(ham["ell"]))
        z_h = ((ham["ber"] - th_h) / st["se"]) if st["se"] and st["se"] > 0 else float("nan")
        w(f"| {x:.0f} | {fmt(r['ber'])} | {fmt(th_b)} | {z_b:+.2f} "
          f"| {fmt(ham['ber'])} | {fmt(th_h)} | "
          f"{'—' if not np.isfinite(z_h) else f'{z_h:+.2f}'} |")
    w()
    w("$z$ — стандартизованное отклонение: (наблюдение − теория) / SE. Для BPSK SE "
      "биномиальная, по числу битов; для Хэмминга — кластерная, по блокам, и только при "
      "$K\\ge30$, иначе в таблице «—». Смотреть на отдельные $z$ надо аккуратно: "
      "**11 поточечных 95 %-х интервалов не образуют одну 95 %-ю полосу**. На 11 точек "
      "в среднем приходится примерно 0.55 ложных выхода за 2σ, поэтому одиночное "
      "отклонение около 2σ — ещё не признак систематической ошибки реализации.")
    w()
    w("Теоретические пороги посчитаны отдельно, методом Брента: "
      "BPSK 4.323 / 6.790 / 8.398 дБ и точная теория Хэмминга 4.578 / 6.657 / 8.084 дБ "
      "для уровней $10^{-2}/10^{-3}/10^{-4}$. Теоретическое пересечение кривой Хэмминга "
      "с некодированной передачей — ≈ 5.849 дБ. Экспериментальные значения ниже получены "
      "**интерполяцией по сетке 1 дБ**, поэтому проверить теорию точнее этой сетки они "
      "не могут. Способы между собой не смешиваются: теоретический порог одной схемы "
      "с интерполированным порогом другой здесь не комбинируется.")
    w()

    # ---------------- 7-8. Энергетический выигрыш ----------------
    out_csv_gain: list[dict] = []
    w("## 7. Энергетический выигрыш и его неопределённость")
    w()
    boot_ext = gain_block(supp_rows, "7.1. Расширенный объём", L, out_csv_gain, "extended") \
        if supp_rows else None
    if boot_ext and "hamming" in boot_ext:
        w(f"**Неопределённость при BER = 10⁻⁴ (кластерный bootstrap, "
          f"{boot_ext['hamming']['n_boot']} реплик, отдельный seed {BOOTSTRAP_SEED}):**")
        w()
        w("| Схема | Точечная оценка, дБ | bootstrap 95 % | SD bootstrap | валидных реплик "
          "| невалидных | смена пары точек |")
        w("|---|---:|---|---:|---:|---:|---:|")
        for scheme in ("hamming", "convolutional"):
            b = boot_ext[scheme]
            w(f"| {NAMES[scheme]} | {b['gain_db']:+.2f} | "
              f"[{b['ci_low']:+.2f}; {b['ci_high']:+.2f}] | {b['bootstrap_sd']:.3f} "
              f"| {b['n_valid']} из {b['n_boot']} | {b['invalid_fraction'] * 100:.1f} % "
              f"| {b['pair_changed_fraction'] * 100:.1f} % |")
        w()
        sens = interpolation_sensitivity(supp_rows)
        w("**Разница между способами интерполяции** — диагностика чувствительности "
          "к методу. Это не доверительная граница и не погрешность сетки:")
        w()
        w("| Схема | lg-интерполяция, дБ | линейная по BER, дБ | разница, дБ |")
        w("|---|---:|---:|---:|")
        for scheme in ("hamming", "convolutional"):
            v = sens[scheme]
            w(f"| {NAMES[scheme]} | {v['gain_log']:+.2f} | "
              f"{'—' if v['gain_linear'] is None else f'{v[chr(103)+chr(97)+chr(105)+chr(110)+chr(95)+chr(108)+chr(105)+chr(110)+chr(101)+chr(97)+chr(114)]:+.2f}'} "
              f"| {'—' if v['delta'] is None else f'{v[chr(100)+chr(101)+chr(108)+chr(116)+chr(97)]:+.2f}'} |")
        w()
        w("Механически складывать эти разницы с bootstrap-интервалом **нельзя**, и его "
          "верхней границей они не являются. Сеточная ошибка самой процедуры оценена "
          "отдельно, на точных теоретических кривых (см. §7.3): на сетке 1 дБ это "
          "−0.026 дБ для BPSK и −0.008 дБ для Хэмминга, на сетке 0.2 дБ — "
          "практически нуль.")
        w("В выводах значения округляются до 0.1 дБ. Три знака в расчётной таблице — "
          "это арифметика, а не измеренная точность.")
        w()
    gain_block(main_rows, "7.2. Основной прогон", L, out_csv_gain, "main")

    grid = metrics.grid_error_report(1e-4)
    w("### 7.3. Насколько виновата сама сетка: проверка на точных теоретических кривых")
    w()
    w("Беру точный корень уравнения $P(x) = 10^{-4}$ и сравниваю с тем, что даёт "
      "lg-интерполяция тех же теоретических значений на сетке 1 дБ и на сетке 0.2 дБ. "
      "Это характеристика **процедуры**, а не экспериментальная оценка.")
    w()
    w("| Кривая | Точный корень, дБ | Сетка 1 дБ | Отклонение | Сетка 0.2 дБ | Отклонение |")
    w("|---|---:|---:|---:|---:|---:|")
    for key, label in (("bpsk", "BPSK без кодирования"), ("hamming", "Хэмминг (7,4)")):
        v = grid[key]
        w(f"| {label} | {v['exact_db']:.4f} | {v['grid_1db_db']:.4f} "
          f"| {v['grid_error_1db']:+.4f} | {v['grid_0p2db_db']:.4f} "
          f"| {v['grid_error_0p2db']:+.5f} |")
    w()
    w("На гладких теоретических кривых сетка 1 дБ даёт смещение в сотые доли дБ, "
      "а на сетке 0.2 дБ оно практически исчезает. Это отдельный источник: в bootstrap "
      "он не входит и сам с ним не складывается.")
    w()

    ref_gains = os.path.join(DATA, "refinement_gains.csv")
    ref_cluster = os.path.join(DATA, "refinement_cluster.csv")
    if os.path.exists(ref_gains) and os.path.exists(ref_cluster):
        import csv as _csv
        with open(ref_gains, newline="", encoding="utf-8") as f:
            grows = list(_csv.DictReader(f))
        with open(ref_cluster, newline="", encoding="utf-8") as f:
            crows = list(_csv.DictReader(f))
        w("### 7.4. Уточняющий прогон по сетке 0.2 дБ")
        w()
        plan_ref = json.load(open(os.path.join(DATA, "refinement_plan.json"), encoding="utf-8"))
        w(f"Протокол зафиксирован до запуска: seed "
          f"**{plan_ref['seed']}**, объёмы заданы заранее, остановки по числу "
          f"ошибок нет. С прежними наборами результаты **не объединяются**. "
          f"Подробности — `документы/отчёты/Уточняющий_прогон.md`.")
        w()
        w("| Набор | Схема | $E_b/N_0$ без кода | $E_b/N_0$ кодированной | Выигрыш | "
          "bootstrap 95 % | невалидных реплик | смена пары точек |")
        w("|---|---|---:|---:|---:|---|---:|---:|")
        for g in grows:
            inv = float(g["invalid_fraction"]) * 100 if g["invalid_fraction"] else 0.0
            pair = float(g["pair_changed_fraction"]) * 100 if g["pair_changed_fraction"] else 0.0
            w(f"| {g['dataset']} | {NAMES[g['scheme']]} | {float(g['x_uncoded_db']):.2f} "
              f"| {float(g['x_coded_db']):.2f} | {float(g['gain_point_db']):+.2f} "
              f"| [{float(g['gain_ci_low_db']):+.2f}; {float(g['gain_ci_high_db']):+.2f}] "
              f"| {inv:.1f} % | {pair:.1f} % |")
        w()
        w("| Схема | $E_b/N_0$ | E | M | K | BER | кластерный 95 % |")
        w("|---|---:|---:|---:|---:|---|---|")
        for r in crows:
            ci = (f"[{float(r['cluster_ci_low']):.3e}, {float(r['cluster_ci_high']):.3e}]"
                  if r["cluster_applicable"] == "1" else "не применим")
            w(f"| {NAMES[r['scheme']]} | {float(r['ebno_db']):.1f} | {int(r['E']):,} "
              f"| {int(r['M']):,} | {int(r['K']):,} | {float(r['ber']):.3e} | {ci} |")
        w()
        w("У Витерби пара соседних точек в bootstrap меняется примерно в 17 % реплик. "
          "Поэтому пересечение ищется по всей сетке, а не по одной исходной паре.")
        w()

    w("## 8. Точки без наблюдаемых ошибок")
    w()
    zero = [r for r in main_rows + supp_rows if r["n_errors"] == 0]
    if not zero:
        w("Таких точек нет.")
    else:
        w("| Набор | Схема | $E_b/N_0$ | M | Верхняя граница 95 % | 3/N_info (для сравнения) |")
        w("|---|---|---:|---:|---|---|")
        for r in sorted(zero, key=lambda r: (r["scheme"], r["ebno_db"])):
            dataset = "основной" if r in main_rows else "расширенный"
            w(f"| {dataset} | {SHORT[r['scheme']]} | {r['ebno_db']:.0f} | "
              f"{int(r['n_units']):,} | {fmt(r['upper95'])} | "
              f"{fmt(3.0 / r['n_info_bits'])} |")
        w()
        w("Нулевое наблюдение — **не** численная точка BER-кривой: в интерполяции оно "
          "не участвует и на графике показывается стрелкой вниз от верхней границы. "
          "Про Хэмминг при $10^{-5}$ в основном прогоне правильно писать "
          "«нет двух надёжных положительных соседних оценок для интерполяции» "
          "(при 9 дБ 1.45·10⁻⁵, при 10 дБ нулевое наблюдение), а не «уровень не достигнут».")
    w()

    # ---------------- 10. Производительность ----------------
    probe_path = os.path.join(DATA, "performance_probe.json")
    if os.path.exists(probe_path):
        probe = json.load(open(probe_path, encoding="utf-8"))
        w("## 9. Замер производительности (вне итоговой статистики)")
        w()
        w("| Схема | Объём | Медиана, с | с на 10⁶ бит |")
        w("|---|---:|---:|---:|")
        for scheme in SCHEMES:
            p = probe[scheme]
            w(f"| {NAMES[scheme]} | {p['n_info_bits']:,} | {p['median_s']:.4f} "
              f"| {p['per_million_s']:.3f} |")
        w()
        w(f"Фактическое время основного прогона: {sum(r['elapsed_s'] for r in main_rows):.0f} с; "
          f"расширенного: {sum(r['elapsed_s'] for r in supp_rows):.0f} с. "
          "Эти замеры выполнены отдельно и в статистику не входят.")
        w()

    # ---------------- 11. Ограничения ----------------
    w("## 10. Чего эти результаты не показывают")
    w()
    w("1. Мягкие решения не моделировались, поэтому сравнения с мягким декодированием "
      "здесь нет и про его выигрыш ничего не утверждается.")
    w("2. Теоретическая кривая свёрточного кода не строилась (вариант (a) по Q10); "
      "$d_{free}=5$ приведено в тексте, но выигрыш по нему не рассчитывался.")
    w("3. Bootstrap приближённый: погрешность сетки 1 дБ он не устраняет и проблему "
      "редких и нулевых событий не решает. У Витерби при малом числе ошибочных пакетов "
      "неопределённость остаётся большой.")
    w("4. Окончательная статистическая точность выигрышей не подтверждена: приведены "
      "точечные оценки и приближённые интервалы. Для уточнения сделан отдельный прогон "
      "по фиксированной сетке с шагом 0.2 дБ (раздел 11).")
    w("5. Выводы относятся к указанным кодам, длине пакета $L=10^4$ и жёсткому декодированию.")
    w()

    # ---------------- 12. Предложение по уточнению ----------------
    w("## 11. Уточняющий прогон: что сделано")
    w()
    w("Уточняющий прогон **выполнен** по зафиксированному протоколу (§7.4): "
      "сетка 0.2 дБ, seed 20261026, объёмы заданы до старта, "
      "результаты лежат отдельно от основных наборов.")
    w()
    w("| Схема | Диапазон | Шаг | Объём на точку |")
    w("|---|---|---:|---:|")
    w("| без кодирования | 8.0…8.8 дБ | 0.2 дБ | 30 000 000 информационных бит |")
    w("| Хэмминг (7,4) | 7.8…8.6 дБ | 0.2 дБ | 30 000 000 бит (7 500 000 блоков) |")
    w("| свёрточный | 6.6…7.4 дБ | 0.2 дБ | 1000 пакетов по $L=10^4$ (10 000 000 бит) |")
    w()
    w("Чтобы увеличить статистику дальше, нужен новый опыт с заранее зафиксированным "
      "протоколом. Ограничения исходных наборов при этом никуда не деваются — они "
      "перечислены в разделе 10.")
    w()

    w("## 12. Файлы результатов")
    w()
    for name in ("ber_results.csv", "ber_results_extended.csv", "ber_refinement.csv",
                 "refinement_plan.json", "refinement_gains.csv", "refinement_cluster.csv",
                 "theory_curves.csv", "energy_gain.csv", "cluster_intervals.csv",
                 "histogram_checks.json", "performance_probe.json",
                 "viterbi_verification.json"):
        path = os.path.join(DATA, name)
        size = os.path.getsize(path) if os.path.exists(path) else "—"
        w(f"- `results/data/{name}` ({size} байт)")
    w("- `results/figures/ber_comparison.png` / `.pdf`, `results/figures/ber_zoom.png` / `.pdf`")
    w("- `документы/отчёты/Проверка_декодера_Витерби.md`, `документы/отчёты/Уточняющий_прогон.md`,")
    w("  `документы/отчёты/Проверка_гистограмм.md`")
    w("- `документы/отчёты/Отчёт_по_результатам.md` — этот отчёт (генерируется из сохранённых CSV)")

    # ---------- запись ----------
    os.makedirs(os.path.dirname(OUT_MD), exist_ok=True)
    with open(OUT_MD, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")

    gain_path = os.path.join(DATA, "energy_gain.csv")
    cols = ["dataset", "scheme", "target_ber", "x_uncoded_db", "x_coded_db",
            "gain_point_db", "gain_ci_low_db", "gain_ci_high_db", "bootstrap_n",
            "bootstrap_valid", "invalid_fraction", "pair_changed_fraction",
            "e_ge_100", "k_ge_30", "quality", "note"]
    with open(gain_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=cols)
        writer.writeheader()
        writer.writerows(out_csv_gain)

    # cluster_intervals.csv
    cluster_rows = []
    for dataset, rows in (("main", main_rows), ("extended", supp_rows)):
        for r in rows:
            st = metrics.cluster_normal_from_hist(metrics.hist_from_str(r["z_hist"]),
                                                  int(r["ell"]))
            cluster_rows.append({
                "dataset": dataset, "scheme": r["scheme"], "ebno_db": r["ebno_db"],
                "E": int(r["n_errors"]), "M": int(r["n_units"]),
                "K": int(r["n_units_with_errors"]), "ell": int(r["ell"]),
                "sum_z2": r["sum_z2"], "mean_W": st["mean_W"], "SE": st["se"],
                "cluster_ci_low": st["ci_low"], "cluster_ci_high": st["ci_high"],
                "cluster_applicable": int(st["applicable"]),
                "conservative_ci_low": r["ci_low"], "conservative_ci_high": r["ci_high"],
                "upper95": r["upper95"],
            })
    cluster_path = os.path.join(DATA, "cluster_intervals.csv")
    with open(cluster_path, "w", newline="", encoding="utf-8") as f:
        import csv as _csv
        writer = _csv.DictWriter(f, fieldnames=list(cluster_rows[0].keys()))
        writer.writeheader()
        writer.writerows(cluster_rows)

    print(f"записано: {OUT_MD}")
    print(f"записано: {gain_path}")
    print(f"записано: {cluster_path}")


if __name__ == "__main__":
    main()
