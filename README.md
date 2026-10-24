# Исследование помехоустойчивости кодов Хэмминга и свёрточных кодов при передаче данных по каналу с шумом

Сравнение трёх режимов передачи в канале AWGN с модуляцией BPSK и **жёстким** решением:

| Режим | Код | Скорость R | Декодирование |
|-------|-----|-----------|----------------|
| 1 | без кодирования | 1 | — |
| 2 | Хэмминг (7,4) | 4/7 | синдромное, исправление 1 ошибки |
| 3 | свёрточный, K=3, (7,5)₈ | L/(2(L+2)) ≈ 0.4999 | Витерби, жёсткие решения |

![Сравнительный график BER](results/figures/ber_comparison.png)

## Назначение

Проект количественно оценивает, насколько помехоустойчивое кодирование снижает вероятность
ошибки на бит (BER) относительно передачи без кодирования, и даёт полностью воспроизводимый
набор кода, тестов, данных и графиков. Кодеры и декодеры реализованы **вручную**
(сторонние библиотеки помехоустойчивого кодирования не используются).

## Требования

* Python 3.10 или новее (проверено на 3.13.16);
* NumPy 1.24+ (проверено 2.3.5), Matplotlib 3.7+ (проверено 3.10.9);
* Jupyter Notebook / VS Code с расширением Jupyter;
* `scipy` **не** требуется: Q-функция считается через `math.erfc`.

## Установка

```bash
git clone <адрес-репозитория> coding_noise_research
cd coding_noise_research
python -m venv venv
source venv/bin/activate          # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

## Структура

```
coding_noise_research/
├── notebooks/01_research.ipynb     # главный отчёт: 19 разделов, исполняемый
├── src/
│   ├── hamming.py                  # код Хэмминга (7,4)
│   ├── convolutional.py            # свёрточный кодер R=1/2, K=3, (7,5)₈
│   ├── viterbi.py                  # декодер Витерби (жёсткие решения, пакетный)
│   ├── channel.py                  # BPSK, AWGN, жёсткий демодулятор
│   ├── metrics.py                  # BER, интервалы, теория, энергетический выигрыш
│   ├── simulation.py               # план эксперимента, счётчики, CSV
│   └── plotting.py                 # все графики — только из сохранённых CSV
├── tests/                          # 119 тестов, стандартный unittest
├── scripts/
│   ├── run_experiment.py           # полный/быстрый эксперимент, замер производительности
│   ├── make_figures.py             # графики из CSV
│   ├── make_report_tables.py       # таблицы из CSV + energy_gain.csv
│   ├── verify_hamming_spec.py      # пакет результатов по Хэммингу
│   ├── verify_viterbi.py           # короткие тесты Витерби со счётчиками случаев
│   ├── check_histograms.py         # 4 тождества гистограмм Z_j по всем строкам CSV
│   ├── run_refinement.py           # уточняющий прогон по сетке 0.2 дБ (отдельный seed)
│   ├── refinement_analysis.py      # выигрыш и интервалы по уточняющему набору
│   └── build_notebook.py           # пересборка notebook
├── results/
│   ├── data/                       # CSV, JSON (коммитятся: малы и нужны для графиков)
│   └── figures/                    # PNG + PDF
├── docs/
│   ├── packet2_pkg/                # пакет для Агента №2 в поддерживаемых MIME-типах
│   └── ...                         # архитектура, спецификации, отчёты, QA
├── requirements.txt
└── .gitignore
```

## Запуск: быстрый старт

### 1. Тесты

```bash
python -m unittest discover -s tests -t . -v
# Ran 119 tests ... OK
```

### 2. Notebook

Откройте `notebooks/01_research.ipynb` в VS Code или Jupyter и выполните все ячейки
(Run All). Notebook работает с **уже сохранёнными** результатами и короткими
демонстрационными прогонами — полный эксперимент он не запускает.

Проверка исполнения из командной строки:

```bash
jupyter nbconvert --to notebook --execute notebooks/01_research.ipynb \
    --output 01_research.ipynb --ExecutePreprocessor.timeout=1800
```

### 3. Полный эксперимент (запускается явно)

```bash
python scripts/run_experiment.py --probe   # замер производительности, ~1 с, не входит в статистику
python scripts/run_experiment.py           # основной + расширенный объём, ~5 мин на 2 vCPU
python scripts/run_experiment.py --main    # только основной план, ~30 с
```

Фиксированный план (задаётся в `src/simulation.py`, `Plan`, и не подбирается по результатам):

| Схема | N_info на точку | Независимых единиц |
|-------|-----------------|--------------------|
| без кодирования | 2·10⁶ бит | 2·10⁶ бит |
| Хэмминг (7,4) | 2·10⁶ бит | 500 000 блоков |
| свёрточный | 10⁶ бит | 100 пакетов по L = 10 000 |

Диапазон Eb/N0 = 0…10 дБ, шаг 1 дБ, seed = 20261024. Расширенный прогон
(20·10⁶ / 20·10⁶ / 10·10⁶ бит) пишется в отдельный CSV с собственным seed.

### 4. Графики, таблицы и неопределённость из сохранённых данных

```bash
python scripts/make_figures.py         # results/figures/*.png и *.pdf (включая ber_zoom)
python scripts/make_report_tables.py   # отчёт, energy_gain.csv, cluster_intervals.csv
python scripts/verify_viterbi.py       # короткие тесты Витерби со счётчиками случаев
python scripts/check_histograms.py     # 4 тождества гистограмм по всем строкам CSV
python scripts/make_transfer_packet.py  # пакет для Агента №2 в поддерживаемых типах
python scripts/check_transfer_packet.py # MIME ⊂ поддерживаемых, побайтовость, полнота
python scripts/run_refinement.py       # уточняющий прогон 0.2 дБ, ~2 мин (отдельный seed)
python scripts/refinement_analysis.py  # выигрыш по уточняющему набору
```

`make_report_tables.py` не моделирует заново: он пересчитывает кластерные SE,
кластерные интервалы и bootstrap-неопределённость выигрыша из сохранённых
гистограмм `Z_j` (столбец `z_hist` в CSV); bootstrap — 10 000 реплик, seed 777001,
отдельный от seed эксперимента, с диагностикой числа невалидных реплик и доли реплик,
в которых менялась пара соседних точек.

## Результаты

* `results/data/ber_results.csv` — основной прогон;
* `results/data/ber_results_extended.csv` — расширенный объём;
* `results/data/ber_refinement.csv` — уточняющий прогон по сетке 0.2 дБ (отдельный набор);
* `results/data/refinement_gains.csv`, `refinement_cluster.csv` — выигрыш и интервалы по нему;
* `results/data/histogram_checks.json` — проверка 4 тождеств гистограмм;
* `results/data/theory_curves.csv` — теория BPSK и точная теория Хэмминга;
* `results/data/energy_gain.csv` — выигрыш: точечная оценка, bootstrap CI, диагностика;
* `results/data/cluster_intervals.csv` — кластерные и консервативные интервалы по точкам;
* `results/data/performance_probe.json` — замер производительности;
* `results/figures/ber_comparison.{png,pdf}` — сравнительный график BER(Eb/N0);
* `results/figures/ber_zoom.{png,pdf}` — окрестность BER = 10⁻⁴ по сетке 0.2 дБ;
* `results/figures/{channel_scheme,state_diagram,trellis_example,hamming_correction}.*` —
  схема канала, диаграмма состояний, пример решётки, исправление ошибки Хэммингом.

В CSV для каждой точки сохраняются: схема, Eb/N0, seed, `N_info`, `N_tx`, число ошибок `E`,
`BER = E/N_info` (без округления), число независимых единиц `M`, число единиц с ошибками `K`,
`ΣZ²`, `R_nominal`, `R_eff`, `σ²`, тип и уровень доверительного интервала, время, статус.

**Ключевые величины (жёсткие решения, выигрыш округлён до 0.1 дБ):** при BER = 10⁻⁴
свёрточный код (7,5)₈ + Витерби даёт **+1.4 дБ**, код Хэмминга (7,4) — **+0.3 дБ**.
Уточняющая сетка 0.2 дБ (bootstrap 95 %, 10 000 реплик): +1.39 [+1.34; +1.43] и
+0.32 [+0.29; +0.34]; расширенный набор на сетке 1 дБ: +1.39 [+1.34; +1.45] и
+0.27 [+0.24; +0.30]. При Eb/N0 ≲ 4 дБ кодирование проигрывает. Точки без наблюдаемых
ошибок приводятся с консервативной верхней границей 95 %.

**Три источника неопределённости разделены:** статистическая (bootstrap по независимым
блокам/пакетам, 10 000 реплик), сеточная/интерполяционная (разница между способами
интерполяции: 0.10 дБ для Хэмминга и 0.19 дБ для свёрточного кода; сеточная ошибка на
точных кривых: −0.026 дБ для BPSK и −0.008 дБ для Хэмминга на сетке 1 дБ) и редкие
события (при K < 30 нормальная аппроксимация не применяется). Эти составляющие
не складываются.

## Передача результатов Агенту №2

Агент №2 принимает файлы типов: `image/png`, `image/webp`, `image/jpeg`, `image/gif`,
`text/plain`, `text/markdown`, `text/csv`, `text/html`, `text/xml`, `text/css`,
`text/javascript`, `application/json`, `application/xml`, `application/javascript`,
`application/pdf`. Под этот список собирается пакет `docs/packet2_pkg/`
(структура проекта сохраняется, точка входа — `docs/packet2_pkg/00_INDEX.md`):

* `.md`, `.txt`, `.csv`, `.json`, `.png`, `.pdf` — передаются как есть;
* `.py` → `<имя>.py.txt` (`text/plain`), дословно;
* `.ipynb` → `<имя>.ipynb.json` (дословно) и рядом `<имя>.ipynb.md` (читаемый рендер);
* `.zip` и иные контейнеры не передаются.

Все копии побайтовые; это проверяет `python scripts/check_transfer_packet.py`
(машинный отчёт — `results/data/transfer_check.json`). Пояснение для Агента №2 и запрос
одобрения на оформление итогового документа — `docs/report/PACKET_2_TRANSFER_NOTE.md`.

## Воспроизведение графиков

Графики строятся **только** из сохранённых CSV. При том же seed повторный запуск
`scripts/run_experiment.py` даёт те же числа (проверяется тестами
`tests/test_simulation.py::TestReproducibility`).

## Документация

* `docs/ARCHITECTURE.md` — архитектура и принятые решения;
* `docs/SPEC_REQUEST_AGENT2_NEXT.md` — математические спецификации (Q3–Q11);
* `docs/report/PACKET_2_RESULTS.md` — пакет результатов: таблицы, интервалы, выигрыш;
* `docs/report/PACKET_HAMMING_RESULTS.md` — пакет результатов по коду Хэмминга;
* `docs/TEST_STATE.md` — фактическое состояние тестов и прогонов;
* `docs/QA_CHECKLIST.md` — контроль качества по 16 пунктам ТЗ;
* `docs/A2_RESPONSE.md` — ответы на заключения Агента №2 (что пересчитано и как);
* `docs/report/REFINEMENT_RESULTS.md` — уточняющий прогон 0.2 дБ;
* `docs/report/HISTOGRAM_CHECKS.md` — проверка тождеств гистограмм по всем строкам;
* `docs/report/VITERBI_VERIFICATION.md` — короткие тесты декодера со счётчиками;
* `docs/report/MANIFEST.md` — манифест передаваемых файлов с SHA-256;
* `docs/report/PACKET_2_TRANSFER_NOTE.md` — передача пакета в поддерживаемых типах
  и запрос одобрения на оформление итогового документа;
* `docs/packet2_pkg/00_INDEX.md` — состав пакета, MIME-типы, размеры и SHA-256.

## Ограничения

Моделируются только два кода и только жёсткое декодирование; канал — идеальный AWGN
без замираний и стираний. Теоретическая граница для свёрточного кода не строилась.
Интервалы для кодированных схем консервативны по построению и могут быть широкими.
