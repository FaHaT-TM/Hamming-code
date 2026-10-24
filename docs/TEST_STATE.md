# СОСТОЯНИЕ ТЕСТОВ И СБОРОК

**Обновлено:** 2026-10-24  
**Окружение:** Python 3.13.16, NumPy 2.3.5, Matplotlib 3.10.9, Linux x86_64, 2 vCPU

Артефакт фиксирует **фактически выполненный** прогон. Тесты, которые не запускались, не объявляются пройденными.

## Команды

```bash
python -m unittest discover -s tests -t . -v          # тесты
python scripts/verify_viterbi.py                     # короткие тесты Витерби с подсчётом случаев
python scripts/build_notebook.py                     # пересборка notebook
jupyter nbconvert --to notebook --execute notebooks/01_research.ipynb \
    --output 01_research.ipynb --ExecutePreprocessor.timeout=1800
python scripts/run_experiment.py --probe             # замер производительности
python scripts/run_experiment.py                     # полный эксперимент
python scripts/make_figures.py                       # графики из CSV
python scripts/make_report_tables.py                 # таблицы, кластерные интервалы, bootstrap
python scripts/check_histograms.py                   # 4 тождества гистограмм по всем строкам
python scripts/run_refinement.py                     # уточняющий прогон 0.2 дБ (seed 20261026)
python scripts/refinement_analysis.py                # выигрыш по уточняющему набору
python scripts/qa_check.py                           # 16 пунктов §19 плана проекта
```

## Результат прогона

```text
Ran 119 tests in 6.812s
OK
```

## Состав (119 тестов)

| Файл | Тестов |
|---|---:|
| `test_channel.py` | 36 |
| `test_convolutional.py` | 15 |
| `test_hamming.py` | 25 |
| `test_simulation.py` | 21 |
| `test_viterbi.py` | 22 |

Дополнительно проверено (§5 спецификации): проверки согласованности гистограмм `Z_j` со счётчиками `E`, `M`, `ΣZ²`; round-trip `hist_to_str`/`hist_from_str`; кластерные статистики и условие применимости нормальной аппроксимации (`K ≥ 30`); форма и монотонность bootstrap-интервала выигрыша.

## Проверка исполнения notebook

`jupyter nbconvert --execute`: **48 ячеек (22 исполняемых), ошибок 0**. Notebook не запускает полный эксперимент автоматически. Добавлены подразделы 16.1 (уточняющий прогон 0.2 дБ) и 16.2 (сеточная ошибка и разница интерполяции) — нумерация 19 разделов сохранена.

## Фактически выполненные прогоны (редакция 3)

| Прогон | Команда | Длительность | Результат |
|---|---|---:|---|
| Тесты | `python -m unittest discover -s tests -t .` | 6.8 с | Ran 119, OK |
| Проверка гистограмм | `python scripts/check_histograms.py` | < 1 с | 81 строка, 0 расхождений |
| Тесты Витерби | `python scripts/verify_viterbi.py` | 1.5 с | 504 случая, 0 несовпадений метрики; 1788/1788 инверсий |
| Уточняющий прогон 0.2 дБ | `python scripts/run_refinement.py` | 125.2 с | 15 точек, `ber_refinement.csv` |
| Анализ уточняющего набора | `python scripts/refinement_analysis.py` | 4.6 с | bootstrap 10 000 реплик |
| Фигуры | `python scripts/make_figures.py` | 3.2 с | 6 фигур × (PNG+PDF) |
| Отчёт и таблицы | `python scripts/make_report_tables.py` | 8.2 с | 12 разделов, §7.1–7.4 |
| Notebook | `jupyter nbconvert --execute` | 14.5 с | 48 ячеек, 0 ошибок |
| QA по §19 | `python scripts/qa_check.py` | 7.9 с | 16/16 |

## Что НЕ покрыто

Мягкие решения, другие коды, каналы с замираниями и стираниями не реализованы — тестов нет. Теоретическая граница свёрточного кода не строилась (Q10, вариант (a)).
