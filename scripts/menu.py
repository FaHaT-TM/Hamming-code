"""Интерактивное меню проекта (CLI/TUI в обычном терминале).

Запуск::

    python scripts/menu.py              # интерактивное меню
    python scripts/menu.py --list       # список всех действий без меню
    python scripts/menu.py run tests    # выполнить одно действие по имени

Меню ничего не считает само: каждый пункт запускает существующий скрипт
проекта отдельным процессом, показывает его вывод и код завершения.
Отдельный первый пункт устанавливает/обновляет зависимости из
``requirements.txt`` интерпретатором, которым запущено меню
(то есть в активное виртуальное окружение).
"""

from __future__ import annotations

import os
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# ---------------------------------------------------------------------------
# Реестр действий: (имя, заголовок, команда-строитель, подтверждение, долгое?)
# Команда строится функцией от ROOT, чтобы меню работало из любого каталога.
# ---------------------------------------------------------------------------


def _script(name: str, *args: str) -> list[str]:
    return [sys.executable, os.path.join(ROOT, "scripts", name), *args]


ACTIONS: dict[str, dict] = {
    "install": {
        "title": "Установить / обновить зависимости (requirements.txt)",
        "cmd": lambda: [sys.executable, "-m", "pip", "install", "-r",
                        os.path.join(ROOT, "requirements.txt")],
        "confirm": False,
    },
    "tests": {
        "title": "Запустить все тесты (unittest)",
        "cmd": lambda: [sys.executable, "-m", "unittest", "discover",
                        "-s", "tests", "-t", "."],
        "confirm": False,
    },
    "qa": {
        "title": "Контроль качества — 16 пунктов (qa_check.py)",
        "cmd": lambda: _script("qa_check.py"),
        "confirm": False,
    },
    "probe": {
        "title": "Замер производительности (run_experiment.py --probe, ~1 с)",
        "cmd": lambda: _script("run_experiment.py", "--probe"),
        "confirm": False,
    },
    "experiment_main": {
        "title": "Эксперимент: только основной план (run_experiment.py --main, ~30 с)",
        "cmd": lambda: _script("run_experiment.py", "--main"),
        "confirm": True,
    },
    "experiment_full": {
        "title": "Эксперимент: основной + расширенный (run_experiment.py, ~5 мин)",
        "cmd": lambda: _script("run_experiment.py"),
        "confirm": True,
    },
    "refinement": {
        "title": "Уточняющий прогон 0.2 дБ (run_refinement.py, ~2 мин)",
        "cmd": lambda: _script("run_refinement.py"),
        "confirm": True,
    },
    "refinement_analysis": {
        "title": "Анализ уточняющего прогона (refinement_analysis.py)",
        "cmd": lambda: _script("refinement_analysis.py"),
        "confirm": False,
    },
    "verify_viterbi": {
        "title": "Проверка декодера Витерби (verify_viterbi.py)",
        "cmd": lambda: _script("verify_viterbi.py"),
        "confirm": False,
    },
    "verify_hamming": {
        "title": "Проверка кода Хэмминга (verify_hamming_spec.py)",
        "cmd": lambda: _script("verify_hamming_spec.py"),
        "confirm": False,
    },
    "check_histograms": {
        "title": "Проверка тождеств гистограмм (check_histograms.py)",
        "cmd": lambda: _script("check_histograms.py"),
        "confirm": False,
    },
    "figures": {
        "title": "Построить графики из CSV (make_figures.py)",
        "cmd": lambda: _script("make_figures.py"),
        "confirm": False,
    },
    "report": {
        "title": "Собрать отчёт и таблицы из CSV (make_report_tables.py)",
        "cmd": lambda: _script("make_report_tables.py"),
        "confirm": False,
    },
    "notebook_build": {
        "title": "Пересобрать notebook (build_notebook.py)",
        "cmd": lambda: _script("build_notebook.py"),
        "confirm": False,
    },
    "notebook_run": {
        "title": "Пересобрать и выполнить notebook (build + nbconvert --execute)",
        "cmd": lambda: [sys.executable, "-m", "jupyter", "nbconvert",
                        "--to", "notebook", "--execute",
                        os.path.join(ROOT, "notebooks", "01_research.ipynb"),
                        "--output", "01_research.ipynb",
                        "--ExecutePreprocessor.timeout=1800"],
        "confirm": True,
        "before": "notebook_build",
    },
    "presentation": {
        "title": "Собрать презентацию (make_presentation.py, нужен python-pptx)",
        "cmd": lambda: _script("make_presentation.py"),
        "confirm": False,
    },
}

# Порядок и группировка пунктов в меню: (заголовок группы, [имена действий])
GROUPS: list[tuple[str, list[str]]] = [
    ("Подготовка", ["install"]),
    ("Проверки", ["tests", "qa", "verify_viterbi", "verify_hamming",
                  "check_histograms"]),
    ("Эксперимент", ["probe", "experiment_main", "experiment_full",
                     "refinement", "refinement_analysis"]),
    ("Результаты и документы", ["figures", "report", "notebook_build",
                                "notebook_run", "presentation"]),
]


def dependencies_ok() -> bool:
    """Установлены ли обязательные зависимости (numpy, matplotlib)?"""
    try:
        import matplotlib  # noqa: F401
        import numpy  # noqa: F401
        return True
    except ImportError:
        return False


def run_action(name: str, *, ask_confirm: bool = True) -> int:
    """Выполнить действие по имени; вернуть код завершения процесса."""
    action = ACTIONS[name]
    if action.get("before"):
        code = run_action(action["before"], ask_confirm=False)
        if code != 0:
            return code
    if ask_confirm and action.get("confirm"):
        answer = input(f"Действие «{action['title']}» может занять несколько "
                       "минут. Продолжить? [y/N] ").strip().lower()
        if answer not in ("y", "yes", "д", "да"):
            print("Отменено.")
            return 0
    cmd = action["cmd"]()
    print(f"\n>>> {' '.join(cmd)}\n")
    try:
        proc = subprocess.run(cmd, cwd=ROOT)
    except FileNotFoundError as exc:
        print(f"Не удалось запустить: {exc}")
        return 1
    status = "успешно" if proc.returncode == 0 else f"ОШИБКА (код {proc.returncode})"
    print(f"\n<<< Завершено: {status}")
    return proc.returncode


def print_menu() -> dict[str, str]:
    """Напечатать меню; вернуть отображение номера пункта в имя действия."""
    print("=" * 72)
    print(" ИССЛЕДОВАНИЕ ПОМЕХОУСТОЙЧИВОСТИ КОДОВ ХЭММИНГА И СВЁРТОЧНЫХ КОДОВ")
    print(" Интерактивное меню проекта")
    print("=" * 72)
    if not dependencies_ok():
        print("\n[!] Не найдены numpy/matplotlib в текущем окружении.")
        print("    Сначала выберите пункт установки зависимостей "
              "(или активируйте .venv).\n")
    mapping: dict[str, str] = {}
    number = 1
    for group, names in GROUPS:
        print(f"\n{group}:")
        for name in names:
            print(f"  {number:>2}. {ACTIONS[name]['title']}")
            mapping[str(number)] = name
            number += 1
    print("\n   0. Выход")
    return mapping


def interactive() -> int:
    while True:
        mapping = print_menu()
        try:
            choice = input("\nВыберите пункт: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return 0
        if choice in ("0", "q", "exit", "выход"):
            print("Выход.")
            return 0
        name = mapping.get(choice)
        if name is None:
            print("Нет такого пункта, попробуйте ещё раз.")
            continue
        try:
            run_action(name)
        except KeyboardInterrupt:
            print("\nПрервано пользователем.")
        try:
            input("\nНажмите Enter, чтобы вернуться в меню…")
        except (EOFError, KeyboardInterrupt):
            print("\nВыход.")
            return 0


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if argv[:1] == ["--list"]:
        for group, names in GROUPS:
            print(f"{group}:")
            for name in names:
                print(f"  {name:<20} {ACTIONS[name]['title']}")
        return 0
    if argv[:1] == ["run"] and len(argv) == 2:
        if argv[1] not in ACTIONS:
            print(f"Неизвестное действие: {argv[1]}. Список: --list")
            return 2
        return run_action(argv[1], ask_confirm=False)
    if argv:
        print(__doc__)
        return 2
    return interactive()


if __name__ == "__main__":
    raise SystemExit(main())
