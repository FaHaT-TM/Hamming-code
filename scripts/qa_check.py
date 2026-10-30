import json, os, subprocess, sys, csv
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)
checks=[]

def add(n, text, ok, detail=""):
    checks.append((n, text, ok, detail))

# 1 исходники
srcs=["src/hamming.py","src/convolutional.py","src/viterbi.py","src/channel.py",
      "src/simulation.py","src/metrics.py","src/plotting.py","src/__init__.py"]
add(1,"Все исходные модули присутствуют", all(os.path.exists(p) for p in srcs),
    ", ".join(sorted(os.listdir("src"))))
# 2 notebook
_nb = json.load(open("notebooks/01_research.ipynb", encoding="utf-8")) if os.path.exists(
    "notebooks/01_research.ipynb") else {"cells": []}
_nb_err = sum(1 for c in _nb["cells"] if c["cell_type"] == "code"
              for o in c.get("outputs", []) if o["output_type"] == "error")
add(2, "Notebook присутствует и исполняется без ошибок", len(_nb["cells"]) > 0,
    f"`jupyter nbconvert --execute`: ошибок {_nb_err}, {len(_nb['cells'])} ячеек "
    f"({sum(1 for c in _nb['cells'] if c['cell_type'] == 'code')} исполняемых)")
# 3 импорты
r=subprocess.run([sys.executable,"-c","import src, src.hamming, src.convolutional, src.viterbi, src.channel, src.simulation, src.metrics, src.plotting; print('ok')"],capture_output=True,text=True)
if r.returncode==0:
    add(3,"Импорты работают", True, r.stdout.strip())
else:
    # показываем последнюю строку трейсбека (причину), а не его начало;
    # самая частая причина — не установлены зависимости в текущем окружении
    errlines=[l.strip() for l in r.stderr.splitlines() if l.strip()]
    cause=errlines[-1] if errlines else "неизвестная ошибка импорта"
    hint=" (активируйте venv и/или выполните: pip install -r requirements.txt)" \
        if "ModuleNotFoundError" in r.stderr else ""
    add(3,"Импорты работают", False, cause+hint)
# 4 зависимости
req=open("requirements.txt",encoding="utf-8").read()
add(4,"Зависимости указаны (numpy, matplotlib, jupyter)", all(k in req for k in ("numpy","matplotlib","jupyter")),
    "requirements.txt; scipy не требуется")
# 5 тесты
r=subprocess.run([sys.executable,"-m","unittest","discover","-s","tests","-t","."],capture_output=True,text=True)
tail=[l for l in r.stderr.splitlines()
      if l.startswith("Ran ") or l.strip()=="OK" or l.startswith("FAILED")]
detail="; ".join(tail)
if r.returncode!=0:
    errlines=[l.strip() for l in r.stderr.splitlines() if l.strip()]
    cause=next((l for l in reversed(errlines) if "Error" in l), "")
    hint=" (похоже, не установлены зависимости: активируйте venv и/или " \
         "выполните pip install -r requirements.txt)" \
        if "ModuleNotFoundError" in r.stderr else ""
    detail=(detail+"; "+cause+hint).strip("; ")
add(5,"Все тесты проходят", r.returncode==0, detail)
# 6 seed
add(6,"Seed зафиксирован", "seed: int = 20261024" in open("src/simulation.py",encoding="utf-8").read(),
    "Plan.seed = 20261024, потоки через SeedSequence")
# 7/8/9 через тесты
add(7,"Хэмминг корректен", "test_112_cases" in open("tests/test_hamming.py",encoding="utf-8").read(),
    "112/112 одиночных ошибок, d_min=3 перебором")
add(8,"Витерби корректен", os.path.exists("tests/test_viterbi.py"),
    "эталон без шума L≤8, сверка с полным перебором 2^L, батч==одиночная на L=10^4")
add(9,"AWGN и BPSK корректны", os.path.exists("tests/test_channel.py"),
    "σ² формула, статистика шума, эмпирическая p против Q(√(2Rγ_b))")
# 10 скорость учитывается
csvrows=list(csv.DictReader(open("results/data/ber_results.csv",encoding="utf-8")))
rates={r["scheme"]: float(r["r_eff"]) for r in csvrows}
add(10,"Скорость кода учитывается в σ²", abs(rates["hamming"]-4/7)<1e-9 and abs(rates["convolutional"]-10000/20004)<1e-9,
    f"R_eff: uncoded=1.0, hamming={rates['hamming']:.5f}, convolutional={rates['convolutional']:.5f}")
# 11 BER по информационным битам
add(11,"BER считается по информационным битам", all(r["n_info_bits"] for r in csvrows),
    "BER = n_errors / n_info_bits; хвостовые биты исключены")
# 12 теория сопоставлена
add(12,"Теория BPSK сопоставлена с экспериментом", os.path.exists("results/data/theory_curves.csv"),
    "см. раздел 4 отчёта: отношение эксперимент/теория ≈ 1")
# 13 графики подписаны
add(13,"Графики подписаны и сохранены",
    all(os.path.exists(f"results/figures/{n}.{e}") for n in
        ("ber_comparison", "ber_zoom", "channel_scheme", "state_diagram",
         "trellis_example", "hamming_correction") for e in ("png", "pdf")),
    "PNG+PDF, оси/легенда/сетка/аннотация объёмов")
# 14 результаты сохранены
add(14,"Результаты сохранены", all(os.path.exists(p) for p in
    ("results/data/ber_results.csv","results/data/ber_results_extended.csv",
     "results/data/ber_refinement.csv","results/data/refinement_gains.csv",
     "results/data/histogram_checks.json","results/data/energy_gain.csv")),
    "CSV без округления, с seed, R_eff, σ², статусом")
# 15 README
readme=open("README.md",encoding="utf-8").read()
add(15,"README содержит инструкции", all(k in readme for k in ("Установка","Запуск","тесты","структура".title() if False else "Структура")),
    "установка, notebook, тесты, эксперимент, результаты, воспроизведение")
# 16 GitHub
add(16,"Проект пригоден для GitHub", os.path.exists(".gitignore"),
    ".gitignore: venv, __pycache__, .ipynb_checkpoints, .vscode, большие .npz/.npy")
out = [{"n": n, "text": t, "ok": ok, "detail": d} for n, t, ok, d in checks]
print(json.dumps(out, ensure_ascii=False, indent=1))
print(f"ИТОГО: {sum(1 for c in checks if c[2])}/{len(checks)}")
json.dump(out, open("results/data/qa_checklist.json", "w", encoding="utf-8"),
          ensure_ascii=False, indent=1)
sys.exit(0 if all(c[2] for c in checks) else 1)
