"""Построение всех графиков из сохранённых CSV.

Запуск из корня проекта::

    python scripts/make_figures.py
"""

from __future__ import annotations

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src import plotting  # noqa: E402

DATA = os.path.join(ROOT, "results", "data")
FIG = os.path.join(ROOT, "results", "figures")


def main() -> None:
    main_csv = os.path.join(DATA, "ber_results.csv")
    supp_csv = os.path.join(DATA, "ber_results_extended.csv")
    theory_csv = os.path.join(DATA, "theory_curves.csv")

    made = []
    made += plotting.ber_curve_plot(
        main_csv, theory_csv, FIG,
        supp_csv=supp_csv if os.path.exists(supp_csv) else None,
    )
    refine_csv = os.path.join(DATA, "ber_refinement.csv")
    if os.path.exists(refine_csv):
        made += plotting.ber_zoom_plot(refine_csv, theory_csv, FIG)
    made += plotting.channel_scheme_plot(FIG)
    made += plotting.state_diagram_plot(FIG)
    made += plotting.trellis_example_plot(FIG)
    made += plotting.hamming_correction_plot(FIG)

    for path in made:
        print("сохранено:", os.path.relpath(path, ROOT))


if __name__ == "__main__":
    main()
