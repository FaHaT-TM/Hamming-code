"""Подтверждения по Витерби (по спецификации, §6): перебор, инверсии, tie-break.

Скрипт выполняет и **считает**:

1. сравнение с полным перебором на коротких пакетах (L = 1…6): число проверенных
   случаев, число несовпадений минимальной метрики с эталоном, число случаев
   с неоднозначным оптимумом и число несовпадений сообщения при единственном оптимуме;
2. полный перебор одиночных инверсий (все сообщения, все позиции) для L = 1…6;
3. специально построенный пример равных метрик (tie-break) с явным разбором;
4. проверку d_free на конечном диапазоне (вычислительное подтверждение, не доказательство).

Пишет ``results/data/viterbi_verification.json`` и ``docs/report/VITERBI_VERIFICATION.md``.
Запуск: python scripts/verify_viterbi.py
"""

from __future__ import annotations

import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import convolutional as conv  # noqa: E402
from src import viterbi  # noqa: E402

JSON_OUT = os.path.join(ROOT, "results", "data", "viterbi_verification.json")
MD_OUT = os.path.join(ROOT, "docs", "report", "VITERBI_VERIFICATION.md")


def brute_force(rx: np.ndarray, length: int):
    """Полный перебор 2^L сообщений: минимальная метрика и все оптимальные сообщения."""
    n_tx = 2 * (length + conv.TAIL)
    if rx.size != n_tx:
        raise ValueError("несовместимая длина")
    best, best_msgs = None, []
    for value in range(1 << length):
        msg = np.array([(value >> (length - 1 - i)) & 1 for i in range(length)],
                       dtype=np.uint8)
        code = conv.encode(msg)
        metric = int(np.count_nonzero(code != rx))
        if best is None or metric < best:
            best, best_msgs = metric, [tuple(int(b) for b in msg)]
        elif metric == best:
            best_msgs.append(tuple(int(b) for b in msg))
    return best, best_msgs


def check_brute_force(rng: np.random.Generator, max_length: int = 6):
    rows = []
    total = mismatches_metric = 0
    unique_cases = message_mismatch = ambiguous_cases = 0
    for length in range(1, max_length + 1):
        misses = 0
        cases = 0
        amb = 0
        msg_miss = 0
        uniq = 0
        for value in range(1 << length):
            info = np.array([(value >> (length - 1 - i)) & 1 for i in range(length)],
                            dtype=np.uint8)
            code = conv.encode(info)
            for n_err in (0, 1, 2, 3):
                if n_err > code.size:
                    continue
                rx = code.copy()
                if n_err:
                    pos = rng.choice(code.size, size=n_err, replace=False)
                    rx[pos] ^= 1
                decoded, det = viterbi.decode(rx, length=length, return_details=True)
                best_metric, best_msgs = brute_force(rx, length)
                cases += 1
                if int(det["final_metrics"][0]) != best_metric:
                    misses += 1
                if len(best_msgs) == 1:
                    uniq += 1
                    if tuple(int(b) for b in decoded) != best_msgs[0]:
                        msg_miss += 1
                else:
                    amb += 1
        rows.append({"L": length, "cases": cases, "metric_mismatch": misses,
                     "unique_optimum": uniq, "message_mismatch": msg_miss,
                     "ambiguous": amb})
        total += cases
        mismatches_metric += misses
        unique_cases += uniq
        message_mismatch += msg_miss
        ambiguous_cases += amb
    return rows, {"cases": total, "metric_mismatch": mismatches_metric,
                  "unique_optimum": unique_cases, "message_mismatch": message_mismatch,
                  "ambiguous": ambiguous_cases}


def check_single_inversions(max_length: int = 6):
    rows = []
    total = ok = 0
    for length in range(1, max_length + 1):
        cases = ok_l = 0
        for value in range(1 << length):
            info = np.array([(value >> (length - 1 - i)) & 1 for i in range(length)],
                            dtype=np.uint8)
            code = conv.encode(info)
            for pos in range(code.size):
                rx = code.copy()
                rx[pos] ^= 1
                cases += 1
                if np.array_equal(viterbi.decode(rx, length=length), info):
                    ok_l += 1
        rows.append({"L": length, "positions": cases, "recovered": ok_l})
        total += cases
        ok += ok_l
    return rows, {"cases": total, "recovered": ok}


def find_tie_example(length: int = 2):
    """Найти принятое слово, где два пути имеют одинаковую метрику, и разобрать выбор."""
    for value in range(1 << length):
        info = np.array([(value >> (length - 1 - i)) & 1 for i in range(length)],
                        dtype=np.uint8)
        code = conv.encode(info)
        for rx_int in range(1 << code.size):
            rx = np.array([(rx_int >> (code.size - 1 - i)) & 1 for i in range(code.size)],
                          dtype=np.uint8)
            best, msgs = brute_force(rx, length)
            if len(msgs) < 2:
                continue
            decoded, det = viterbi.decode(rx, length=length, return_details=True)
            metrics = []
            for msg in msgs:
                c = conv.encode(np.array(msg, dtype=np.uint8))
                metrics.append(int(np.count_nonzero(c != rx)))
            return {
                "length": length,
                "received": "".join(map(str, rx)),
                "optimal_messages": ["".join(map(str, m)) for m in msgs],
                "metrics": metrics,
                "decoder_metric": int(det["final_metrics"][0]),
                "decoder_choice": "".join(map(str, decoded)),
                "rule": "выбирается ребро с меньшим номером предыдущего состояния "
                        "(EDGE_PREV[:, 0] = 2·(q mod 2) < EDGE_PREV[:, 1])",
                "edge_prev_table": {str(q): [int(viterbi.EDGE_PREV[q, 0]),
                                             int(viterbi.EDGE_PREV[q, 1])]
                                    for q in range(conv.N_STATES)},
            }
    return None


def main() -> None:
    rng = np.random.default_rng(20261024)

    bf_rows, bf_tot = check_brute_force(rng)
    si_rows, si_tot = check_single_inversions()
    tie = find_tie_example()

    # вычислительное подтверждение d_free на конечном диапазоне
    best_weight = 99
    for value in range(1, 4096):
        info = np.array([int(b) for b in format(value, "b").zfill(12)], dtype=np.uint8)
        best_weight = min(best_weight, int(conv.encode(info).sum()))

    payload = {
        "brute_force": {"rows": bf_rows, "total": bf_tot},
        "single_inversions": {"rows": si_rows, "total": si_tot},
        "tie_break_example": tie,
        "d_free": {"value": conv.D_FREE, "computational_min_over_L<=12": best_weight},
    }
    os.makedirs(os.path.dirname(JSON_OUT), exist_ok=True)
    with open(JSON_OUT, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)

    L = []
    def w(text=""):
        L.append(text)
    w("# ПОДТВЕРЖДЕНИЯ ПО ДЕКОДЕРУ ВИТЕРБИ")
    w()
    w("Сформировано скриптом `scripts/verify_viterbi.py`; все числа вычислены, "
      "а не переписаны из прошлых отчётов.")
    w()
    w("## 1. Сравнение с полным перебором $2^L$ сообщений")
    w()
    w("* проверялись все сообщения для $L = 1\\ldots6$, ошибки 0–3 случайных позиций;")
    w("* сравнивались **минимальные метрики** (а не только сообщения), что и требуется "
      "для подтверждения ML-корректности;")
    w("* совпадение сообщения требовалось только там, где оптимум единственный.")
    w()
    w("| L | случаев | несовпадений метрики | единственный оптимум | несовпадений сообщения | неоднозначных |")
    w("|---:|---:|---:|---:|---:|---:|")
    for r in bf_rows:
        w(f"| {r['L']} | {r['cases']} | {r['metric_mismatch']} | {r['unique_optimum']} "
          f"| {r['message_mismatch']} | {r['ambiguous']} |")
    w(f"| **итого** | **{bf_tot['cases']}** | **{bf_tot['metric_mismatch']}** "
      f"| **{bf_tot['unique_optimum']}** | **{bf_tot['message_mismatch']}** "
      f"| **{bf_tot['ambiguous']}** |")
    w()
    w("## 2. Полный перебор одиночных инверсий")
    w()
    w("| L | инверсий проверено | восстановлено |")
    w("|---:|---:|---:|")
    for r in si_rows:
        w(f"| {r['L']} | {r['positions']} | {r['recovered']} |")
    w(f"| **итого** | **{si_tot['cases']}** | **{si_tot['recovered']}** |")
    w()
    if tie:
        w("## 3. Пример равных метрик (tie-break)")
        w()
        w(f"Длина пакета $L = {tie['length']}$, принято `{tie['received']}`.")
        w()
        w(f"* два оптимальных сообщения: "
          f"{', '.join('`' + m + '`' for m in tie['optimal_messages'])};")
        w(f"* их метрики: {tie['metrics']} — равны, то есть пути равноправны;")
        w(f"* метрика декодера: {tie['decoder_metric']};")
        w(f"* декодер выбрал `{tie['decoder_choice']}` по правилу: {tie['rule']};")
        w(f"* таблица предшественников `EDGE_PREV` = {tie['edge_prev_table']}.")
        w()
        w("Правило детерминировано и одинаково во всём исследовании; на среднюю BER оно "
          "может влиять, поэтому фиксируется, а не выбирается «по ситуации».")
        w()
    w("## 4. Свободное расстояние")
    w()
    w(f"* $d_{{free}} = {conv.D_FREE}$ — по доказательству на четырёхсостоятельной "
      "диаграмме из спецификации (путь по входу 100: $00\\to10\\to01\\to00$, выход "
      "$11|10|11$, вес 5; меньшее невозможно, так как первое ребро и ребро возврата "
      "имеют вес 2, а среднее — вес 1);")
    w(f"* вычислительная проверка на конечном диапазоне $L\\le12$: минимальный вес "
      f"ненулевого кодового слова = **{best_weight}**. Это подтверждение на конечном "
      "множестве, а не самостоятельное доказательство глобального минимума.")
    w()
    w("## 5. Как воспроизвести")
    w()
    w("```bash")
    w("python scripts/verify_viterbi.py")
    w("```")

    os.makedirs(os.path.dirname(MD_OUT), exist_ok=True)
    with open(MD_OUT, "w", encoding="utf-8") as f:
        f.write("\n".join(L) + "\n")
    print(f"записано: {JSON_OUT}")
    print(f"записано: {MD_OUT}")
    print(json.dumps({k: v for k, v in payload.items() if k != "brute_force"}, ensure_ascii=False)[:600])
    print("brute force total:", bf_tot)


if __name__ == "__main__":
    main()
