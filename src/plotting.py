"""Визуализация: графики строятся только из сохранённых результатов.

Все функции принимают пути к CSV, созданным :mod:`src.simulation`, и сохраняют
рисунки в ``results/figures`` в форматах PNG и PDF.

Точки с нулём ошибок
--------------------
BER = 0 не отображается как точка на логарифмической шкале. Вместо неё
рисуется маркер верхней границы ``ber_upper95`` со стрелкой вниз и подпись
«консервативная односторонняя граница 95 % (по независимым блокам/пакетам)».
Такие точки не участвуют в логарифмической интерполяции.
"""

from __future__ import annotations

import csv
import os

import matplotlib

matplotlib.use("Agg")           # работа без дисплея (ноутбук, CI, сервер)
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np               # noqa: E402

from . import convolutional as conv  # noqa: E402
from . import hamming  # noqa: E402

__all__ = [
    "read_csv", "ber_curve_plot", "ber_zoom_plot", "channel_scheme_plot",
    "state_diagram_plot", "trellis_example_plot", "hamming_correction_plot",
]

STYLE = {
    "uncoded": {"color": "#1f77b4", "marker": "o", "label": "без кодирования (BPSK)"},
    "hamming": {"color": "#d62728", "marker": "s", "label": "Хэмминг (7,4)"},
    "convolutional": {"color": "#2ca02c", "marker": "^",
                      "label": "свёрточный (7,5)₈ + Витерби"},
}
SCHEME_ORDER = ["uncoded", "hamming", "convolutional"]


# --------------------------------------------------------------------------
# Чтение сохранённых результатов
# --------------------------------------------------------------------------
def read_csv(path: str) -> list[dict]:
    """Прочитать CSV результатов, приводя числовые поля к float/int."""
    if not os.path.exists(path):
        raise FileNotFoundError(f"нет файла результатов: {path}")
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for raw in csv.DictReader(f):
            row = {}
            for key, value in raw.items():
                if value == "":
                    row[key] = float("nan")
                    continue
                try:
                    row[key] = float(value)
                except ValueError:
                    row[key] = value
            # целочисленные счётчики
            for key in ("n_info_bits", "n_errors", "n_units", "n_units_with_errors",
                        "ell", "length", "tail_bits", "seed"):
                if key in row and row[key] == row[key]:
                    row[key] = int(row[key])
            rows.append(row)
    return rows


# --------------------------------------------------------------------------
# Основной сравнительный график BER(Eb/N0)
# --------------------------------------------------------------------------
def ber_curve_plot(data_csv: str, theory_csv: str | None = None,
                   out_dir: str = "results/figures",
                   supp_csv: str | None = None,
                   title: str | None = None) -> list[str]:
    """Построить сравнительный график BER(Eb/N0) по сохранённым CSV.

    Parameters
    ----------
    data_csv : str
        Основной прогон (фиксированный план).
    theory_csv : str, optional
        Теоретические кривые (BPSK и точная BER Хэмминга).
    supp_csv : str, optional
        Дополнительный прогон с расширенным объёмом (если выполнялся).
    out_dir : str
        Каталог для PNG/PDF.

    Returns
    -------
    list[str]
        Пути сохранённых файлов.
    """
    rows = read_csv(data_csv)
    supp = read_csv(supp_csv) if supp_csv and os.path.exists(supp_csv) else []
    theory = read_csv(theory_csv) if theory_csv and os.path.exists(theory_csv) else []

    fig, ax = plt.subplots(figsize=(9.0, 6.4))

    # --- теория ---
    if theory:
        x = np.array([r["ebno_db"] for r in theory])
        ax.semilogy(x, [r["ber_bpsk_theory"] for r in theory], "--",
                    color="#1f77b4", lw=1.4, alpha=0.85,
                    label="BPSK без кодирования — теория $Q(\\sqrt{2E_b/N_0})$")
        ax.semilogy(x, [r["ber_hamming_theory"] for r in theory], "--",
                    color="#d62728", lw=1.4, alpha=0.85,
                    label="Хэмминг (7,4) — точная теория (перебор 128 шаблонов)")

    zero_notes = []
    for scheme in SCHEME_ORDER:
        pts = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: r["ebno_db"])
        if not pts:
            continue
        st = STYLE[scheme]
        x_ok, y_ok, lo, hi = [], [], [], []
        for r in pts:
            if r["ber"] > 0:
                x_ok.append(r["ebno_db"])
                y_ok.append(r["ber"])
                lo.append(r.get("ci_low", float("nan")))
                hi.append(r.get("ci_high", float("nan")))
            elif r.get("upper95", float("nan")) == r.get("upper95", float("nan")):
                # ноль ошибок: стрелка вниз от консервативной верхней границы
                ax.annotate("", xy=(r["ebno_db"], r["upper95"] * 0.45),
                            xytext=(r["ebno_db"], r["upper95"]),
                            arrowprops=dict(arrowstyle="-|>", color=st["color"], lw=1.2))
                ax.plot([r["ebno_db"]], [r["upper95"]], marker="v", ms=6, mfc="none",
                        mec=st["color"], ls="none")
                zero_notes.append((scheme, r["ebno_db"], r["upper95"]))

        if x_ok:
            label = st["label"] + " — эксперимент"
            ax.errorbar(x_ok, y_ok, marker=st["marker"], color=st["color"],
                        ls="-", lw=1.6, ms=6, capsize=3, label=label, zorder=3)

        # расширенный объём: полые маркеры того же цвета
        spts = sorted((r for r in supp if r["scheme"] == scheme), key=lambda r: r["ebno_db"])
        sx = [r["ebno_db"] for r in spts if r["ber"] > 0]
        sy = [r["ber"] for r in spts if r["ber"] > 0]
        if sx:
            ax.plot(sx, sy, marker=st["marker"], mfc="none", mec=st["color"],
                    ls="none", ms=6, mew=1.1, alpha=0.9, zorder=4,
                    label=f"{st['label']} — расширенный объём (полые маркеры)")

    ax.set_xlabel("$E_b/N_0$, дБ (на информационный бит до кодирования)")
    ax.set_ylabel("BER — вероятность ошибки на информационный бит")
    if title is None:
        title = ("Помехоустойчивость кодов Хэмминга и свёрточных кодов в канале с шумом\n"
                 "BPSK + AWGN, жёсткие решения; BER — по информационным битам")
    ax.set_title(title, fontsize=11)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.set_xlim(-0.3, 10.3)
    ax.set_ylim(bottom=5e-8)

    handles, labels = ax.get_legend_handles_labels()
    if zero_notes:
        handles.append(plt.Line2D([], [], marker="v", ls="none", mfc="none",
                                  mec="0.35", ms=6))
        labels.append("ошибок не наблюдалось: стрелка вниз от консервативной\n"
                      "односторонней границы 95 % (по независимым блокам/пакетам)")
    ax.legend(handles, labels, fontsize=6.8, loc="lower left", framealpha=0.93,
              ncol=1, handlelength=2.2, borderpad=0.5, labelspacing=0.35)

    # аннотация объёмов
    txt = []
    for scheme in SCHEME_ORDER:
        pts = [r for r in rows if r["scheme"] == scheme]
        if pts:
            n = max(r["n_info_bits"] for r in pts)
            r_eff = pts[0].get("r_eff", float("nan"))
            txt.append(f"{STYLE[scheme]['label']}: N = {n/1e6:.1f}·10⁶ бит, "
                       f"R_eff = {r_eff:.5f}")
    ax.text(0.985, 0.975, "\n".join(txt), transform=ax.transAxes, fontsize=7.2,
            ha="right", va="top",
            bbox=dict(boxstyle="round", fc="white", ec="0.8", alpha=0.92))

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"ber_comparison.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


# --------------------------------------------------------------------------
# Вспомогательные иллюстрации
# --------------------------------------------------------------------------
def channel_scheme_plot(out_dir: str = "results/figures") -> list[str]:
    """Схема системы связи: источник → кодер → BPSK → AWGN → демодулятор → декодер."""
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.set_xlim(0, 11)
    ax.set_ylim(0, 3.4)
    ax.axis("off")

    boxes = [
        (0.15, 1.5, 1.5, 0.9, "Источник\nслучайных\nбитов"),
        (2.0, 1.5, 1.7, 0.9, "Кодер\nХэмминга (7,4)\nили свёрточный\n(7,5)₈"),
        (4.05, 1.5, 1.4, 0.9, "BPSK\n0→+1, 1→−1\n$E_s=1$"),
        (5.85, 1.5, 1.5, 0.9, "AWGN\n$y=x+n$\n$\\sigma^2=\\frac{1}{2R_{eff}\\gamma_b}$"),
        (7.75, 1.5, 1.5, 0.9, "Жёсткий\nдемодулятор\n$\\hat b=1$ при $y<0$"),
        (9.45, 1.5, 1.4, 0.9, "Декодер\n(синдром /\nВитерби)"),
    ]
    for x, y, w, h, text in boxes:
        ax.add_patch(plt.Rectangle((x, y), w, h, fc="#eef4fb", ec="#4c72b0", lw=1.4))
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=8.4)

    for x0, x1 in [(1.65, 2.0), (3.7, 4.05), (5.45, 5.85), (7.35, 7.75), (9.25, 9.45)]:
        ax.annotate("", xy=(x1, 1.95), xytext=(x0, 1.95),
                    arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.3))

    ax.text(0.15, 0.55, "Информационные биты $\\to$ BER считается по информационным "
                        "битам: $BER = N_{errors}/N_{info}$", fontsize=8.6)
    ax.text(0.15, 2.85, "Модель системы связи (канал AWGN, жёсткие решения)",
            fontsize=11.5, weight="bold")

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"channel_scheme.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def state_diagram_plot(out_dir: str = "results/figures") -> list[str]:
    """Диаграмма состояний свёрточного кодера (4 состояния, 8 рёбер).

    Петли (00→00 и 11→11) рисуются отдельными дугами над узлами, чтобы метки
    не перекрывали состояния. Подпись на ребре — ``u/v1v2``.
    """
    positions = {0: (0.0, 1.3), 1: (0.0, -1.3), 2: (2.1, 1.3), 3: (2.1, -1.3)}
    names = {0: "00", 1: "01", 2: "10", 3: "11"}

    fig, ax = plt.subplots(figsize=(8.0, 5.6))
    ax.set_xlim(-1.2, 3.2)
    ax.set_ylim(-2.75, 2.6)
    ax.axis("off")

    def color(u):
        return "#4c72b0" if u == 0 else "#c44e52"

    # --- петли (00→00 сверху, 11→11 снизу; метки не перекрывают узлы) ---
    loop_spec = {(0, 0): (+1, "верх"), (3, 1): (-1, "низ")}
    for s_, u_, out_, nxt_ in conv.TRANSITIONS:
        if s_ != nxt_:
            continue
        x, y = positions[s_]
        direction = loop_spec[(s_, u_)][0]
        dy = 0.24 * direction
        rad = 1.05 * direction
        ax.annotate("", xy=(x - 0.14, y + dy), xytext=(x + 0.14, y + dy),
                    arrowprops=dict(arrowstyle="-|>", color=color(u_), lw=1.7,
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=2, shrinkB=2))
        ax.text(x, y + dy + 0.62 * direction, f"{u_}/{out_}", fontsize=8.5,
                color=color(u_), ha="center",
                va="bottom" if direction > 0 else "top",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # --- переходы между разными состояниями ---
    for s, u, out, nxt in conv.TRANSITIONS:
        if s == nxt:
            continue
        (x0, y0), (x1, y1) = positions[s], positions[nxt]
        dx, dy = x1 - x0, y1 - y0
        norm = max((dx * dx + dy * dy) ** 0.5, 1e-9)
        # два ребра между одной парой состояний должны расходиться
        same_pair = sum(1 for s2, u2, _o, n2 in conv.TRANSITIONS
                        if {s2, n2} == {s, nxt})
        rad = 0.18 if u == 0 else -0.18
        if same_pair == 2:
            rad = 0.34 if u == 0 else -0.34
        ax.annotate("", xy=(x1 - 0.30 * dx / norm, y1 - 0.30 * dy / norm),
                    xytext=(x0 + 0.30 * dx / norm, y0 + 0.30 * dy / norm),
                    arrowprops=dict(arrowstyle="-|>", color=color(u), lw=1.6,
                                    connectionstyle=f"arc3,rad={rad}",
                                    shrinkA=2, shrinkB=2))
        # метка ставится ближе к источнику ребра, чтобы не пересекаться в центре
        t_lab = 0.30
        mx, my = x0 + dx * t_lab, y0 + dy * t_lab
        px, py = -dy / norm, dx / norm
        sign = 1.0 if rad >= 0 else -1.0
        off = (0.30 if same_pair == 2 else 0.20) * sign + 0.10
        ax.text(mx + px * off, my + py * off, f"{u}/{out}", fontsize=8.5,
                color=color(u), ha="center", va="center",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.85))

    # --- узлы ---
    for s, (x, y) in positions.items():
        ax.add_patch(plt.Circle((x, y), 0.30, fc="#eef4fb", ec="#333", lw=1.6, zorder=5))
        ax.text(x, y, names[s], ha="center", va="center", fontsize=11, weight="bold",
                zorder=6)

    ax.annotate("", xy=(-0.32, 1.3), xytext=(-1.05, 1.3),
                arrowprops=dict(arrowstyle="-|>", color="#333", lw=1.5))
    ax.text(-1.10, 1.55, "начальное\nсостояние 00", ha="left", fontsize=8, color="#555")
    ax.text(-1.15, -2.45, "после двух нулевых хвостов решётка возвращается в 00",
            ha="left", fontsize=8, color="#555")

    red = plt.Line2D([], [], color="#c44e52", lw=1.6, label="вход $u=1$")
    blue = plt.Line2D([], [], color="#4c72b0", lw=1.6, label="вход $u=0$")
    ax.legend(handles=[blue, red], loc="upper right", fontsize=8.5, framealpha=0.9)
    ax.set_title("Диаграмма состояний: подпись на ребре $u/v_1v_2$, "
                 "$v_1=u\\oplus a\\oplus b$, $v_2=u\\oplus b$", fontsize=9.5)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"state_diagram.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def trellis_example_plot(out_dir: str = "results/figures", info: str = "1011") -> list[str]:
    """Пример решётки Витерби для конкретной информационной последовательности."""
    bits = np.array([int(c) for c in info], dtype=np.int8)
    states = conv.state_sequence(bits)      # S_0 … S_T (T = L + 2)
    steps = len(states) - 1                 # число переходов решётки

    fig, ax = plt.subplots(figsize=(9.0, 4.4))
    ax.set_xlim(-0.7, steps + 0.35)
    ax.set_ylim(-0.9, 3.7)
    ax.axis("off")

    for t in range(steps + 1):              # слоёв состояний на один больше, чем переходов
        for s in range(4):
            ax.plot([t], [s], marker="o", ms=7, mfc="#eef4fb", mec="#333", zorder=3)
    for t in range(steps):
        u = int(bits[t]) if t < len(bits) else 0
        s0, s1 = int(states[t]), int(states[t + 1])
        out = int(conv.OUT_TABLE[s0, u])
        label = f"{out >> 1}{out & 1}"
        ax.annotate("", xy=(t + 1, s1), xytext=(t, s0),
                    arrowprops=dict(arrowstyle="-|>", color="#c44e52", lw=2.0,
                                    shrinkA=7, shrinkB=7))
        ax.text(t + 0.5, (s0 + s1) / 2 + 0.18, label, ha="center", fontsize=8.5,
                color="#c44e52")

    for s in range(4):
        ax.text(-0.55, s, f"{s:02b}", ha="left", va="center", fontsize=9)
    for t in range(steps):
        u = int(bits[t]) if t < len(bits) else 0
        ax.text(t + 0.5, -0.55, f"$u$={u}", ha="center", fontsize=8.5)
        if t >= len(bits):
            ax.text(t + 0.5, -0.82, "хвост", ha="center", fontsize=7.5, color="#777")
    ax.text((steps - 1) / 2, 3.45, f"Вход $u$ = {info}, выход (пары $v_1v_2$)",
            ha="center", fontsize=10)

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"trellis_example.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def hamming_correction_plot(out_dir: str = "results/figures") -> list[str]:
    """Иллюстрация исправления одиночной ошибки кодом Хэмминга (7,4)."""
    info = np.array([1, 0, 1, 1], dtype=np.uint8)
    code = hamming.encode(info)
    rx = code.copy()
    rx[4] ^= 1                                    # ошибка в позиции 5
    diag = hamming.decode_with_diagnostics(rx)
    fixed = diag["corrected_words"]
    j = int(diag["error_position"][0])

    labels = ["p1", "p2", "d1", "p4", "d2", "d3", "d4"]
    fig, axes = plt.subplots(3, 1, figsize=(8.2, 4.6))
    rows = [(code, "Передано"), (rx, f"Принято (ошибка в позиции {j})"), (fixed, "Исправлено")]
    for ax, (bits, title), idx in zip(axes, rows, range(3)):
        ax.set_xlim(-0.7, 6.7)
        ax.set_ylim(-0.6, 1.6)
        ax.axis("off")
        ax.text(-0.65, 0.55, title, fontsize=9, ha="left", va="center")
        for pos in range(7):
            bad = pos == j - 1 and idx > 0
            fc = "#fde2e1" if bad else ("#e8f4ea" if pos in hamming.INFO_POSITIONS
                                        else "#eef4fb")
            ec = "#c44e52" if bad else "#4c72b0"
            ax.add_patch(plt.Rectangle((pos - 0.42, 0.12), 0.84, 0.8, fc=fc, ec=ec, lw=1.4))
            ax.text(pos, 0.52, str(int(bits[pos])), ha="center", va="center", fontsize=13)
            ax.text(pos, 0.0, labels[pos], ha="center", fontsize=8.5, color="#555")

    fig.suptitle(f"Синдром ({int(diag['syndrome'][0, 0])}, {int(diag['syndrome'][0, 1])}, "
                 f"{int(diag['syndrome'][0, 2])}) → $j$ = {j} → инверсия бита {j}; "
                 f"декодировано: {''.join(str(int(b)) for b in diag['bits'])}", fontsize=10)
    fig.tight_layout()

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"hamming_correction.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths


def ber_zoom_plot(refine_csv: str, theory_csv: str | None = None,
                  out_dir: str = "results/figures") -> list[str]:
    """Увеличенный фрагмент вокруг BER = 10⁻⁴ по уточняющей сетке 0.2 дБ."""
    rows = read_csv(refine_csv)
    theory = read_csv(theory_csv) if theory_csv and os.path.exists(theory_csv) else []

    fig, ax = plt.subplots(figsize=(8.2, 5.6))
    if theory:
        x = np.array([r["ebno_db"] for r in theory])
        ax.semilogy(x, [r["ber_bpsk_theory"] for r in theory], "--", color="#1f77b4",
                    lw=1.3, alpha=0.85, label="BPSK — теория")
        ax.semilogy(x, [r["ber_hamming_theory"] for r in theory], "--", color="#d62728",
                    lw=1.3, alpha=0.85, label="Хэмминг (7,4) — точная теория")

    for scheme in SCHEME_ORDER:
        pts = sorted((r for r in rows if r["scheme"] == scheme), key=lambda r: r["ebno_db"])
        pts = [r for r in pts if r["ber"] > 0]
        if not pts:
            continue
        st = STYLE[scheme]

        def _bounds(r):
            """Кластерный интервал при K = n_units_with_errors >= 30, иначе консервативный."""
            if int(r.get("n_units_with_errors") or 0) >= 30 and r.get("cluster_low"):
                return float(r["cluster_low"]), float(r["cluster_high"])
            return float(r["ci_low"]), float(r["ci_high"])

        bnds = [_bounds(r) for r in pts]
        ax.errorbar([r["ebno_db"] for r in pts], [r["ber"] for r in pts],
                    yerr=[[max(r["ber"] - lo, 1e-12) for r, (lo, hi) in zip(pts, bnds)],
                          [max(hi - r["ber"], 1e-12) for r, (lo, hi) in zip(pts, bnds)]],
                    marker=st["marker"], color=st["color"], ls="-", lw=1.4, ms=5,
                    capsize=3, label=st["label"])

    ax.axhline(1e-4, color="0.4", ls=":", lw=1.2)
    ax.text(ax.get_xlim()[0] + 0.03, 1.05e-4, "BER = $10^{-4}$", fontsize=8, color="0.35")
    ax.set_xlabel("$E_b/N_0$, дБ")
    ax.set_ylabel("BER")
    ax.set_ylim(1e-5, 1e-2)
    ax.set_title("Уточняющая сетка 0.2 дБ: окрестность BER $=10^{-4}$\n"
                 "интервалы — кластерные 95 % по независимым единицам ($K\\geq 30$)",
                 fontsize=10)
    ax.grid(True, which="both", ls=":", alpha=0.5)
    ax.legend(fontsize=8, loc="lower left")

    os.makedirs(out_dir, exist_ok=True)
    paths = []
    for ext in ("png", "pdf"):
        path = os.path.join(out_dir, f"ber_zoom.{ext}")
        fig.savefig(path, dpi=300, bbox_inches="tight")
        paths.append(path)
    plt.close(fig)
    return paths
