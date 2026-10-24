#!/usr/bin/env python3
"""Пакет для Агента №2 в поддерживаемых типах файлов.

Агент №2 принимает следующие MIME-типы:

    image/png, image/webp, image/jpeg, image/gif,
    text/plain, text/markdown, text/csv, text/html, text/xml,
    text/css, text/javascript, application/json, application/xml,
    application/javascript, application/pdf

Отсюда правила сборки пакета `docs/packet2_pkg/` (структура каталогов сохраняется):

* `.md`, `.txt`, `.csv`, `.json`, `.png`, `.pdf` — копируются **как есть**,
  без изменения имени: их типы поддерживаются напрямую;
* `.py` → `<имя>.py.txt` (`text/plain`): Python-тип не поддерживается, содержимое
  переносится дословно;
* `.ipynb` → `<имя>.ipynb.json` (`application/json`, дословная копия) **и** рядом
  `<имя>.ipynb.md` (`text/markdown`, читаемый рендер: код, текстовые выводы, сводка);
* `.gitignore` → `.gitignore.txt`, `.gitkeep` → `.gitkeep.txt` (`text/plain`);
* `.zip` и прочие контейнеры — **не передаются** (тип не поддерживается).

Ничего не пересчитывается: все копии побайтовые, что проверяется
`scripts/check_transfer_packet.py`.

Запуск:
    python scripts/make_transfer_packet.py
"""
from __future__ import annotations

import hashlib
import json
import os
import shutil
import subprocess

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "docs/packet2_pkg"

# MIME, которые принимает Агент №2 (по расширению файла пакета)
ALLOWED = {
    ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv",
    ".html": "text/html", ".xml": "text/xml", ".css": "text/css",
    ".js": "text/javascript", ".json": "application/json",
    ".pdf": "application/pdf",
}

# расширения проекта, которые передаются как есть
AS_IS = {".md", ".txt", ".csv", ".json", ".png", ".pdf", ".webp", ".jpg", ".jpeg", ".gif"}
# что переименовывается в поддерживаемый тип
RENAME = {"": ".txt", ".py": ".py.txt", ".ipynb": ".ipynb.json",
          ".gitignore": ".gitignore.txt", ".gitkeep": ".gitkeep.txt"}

SKIP_PREFIX = ("docs/packet2_pkg/", "docs/report/packet2_for_agent2.zip",
               "docs/report/MANIFEST.md",
               "results/data/transfer_check.json", "results/data/transfer_archive.json")


def sha256_file(path: str) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return sorted(p for p in out.stdout.decode("utf-8").split("\0")
                  if p and not p.startswith(SKIP_PREFIX))


def mime_of(path: str) -> str:
    ext = os.path.splitext(path)[1].lower()
    return ALLOWED.get(ext, "НЕ ПОДДЕРЖИВАЕТСЯ")


def fence_for(text: str, minimum: int = 3) -> str:
    run = best = 0
    for ch in text:
        if ch == "`":
            run += 1
            best = max(best, run)
        else:
            run = 0
    return "`" * max(minimum, best + 1)


# ---------------------------------------------------------------- notebook → md
def notebook_to_md(path: str) -> tuple[str, dict]:
    with open(os.path.join(ROOT, path), encoding="utf-8") as f:
        nb = json.load(f)
    cells = nb.get("cells", [])
    parts: list[str] = []
    n_code = n_md = n_img = n_err = n_raw = 0
    for i, cell in enumerate(cells, 1):
        ctype = cell.get("cell_type")
        src = "".join(cell.get("source", []))
        if ctype == "markdown":
            n_md += 1
            parts.append(f"<!-- ячейка {i}: markdown -->\n{src.rstrip()}\n")
        elif ctype == "code":
            n_code += 1
            f = fence_for(src)
            parts.append(f"<!-- ячейка {i}: код -->\n{f}python\n{src.rstrip()}\n{f}\n")
            outs: list[str] = []
            for o in cell.get("outputs", []):
                if o.get("output_type") == "stream":
                    n_raw += 1
                    outs.append("".join(o.get("text", [])).rstrip("\n"))
                elif o.get("output_type") == "error":
                    n_err += 1
                    outs.append(f"[ОШИБКА] {o.get('ename')}: {o.get('evalue')}")
                elif "data" in o:
                    data = o["data"]
                    if "image/png" in data:
                        n_img += 1
                        outs.append("[вывод-изображение: PNG отрисовывается исполнением ячейки; "
                                    "сам рисунок передан отдельными файлами results/figures/*.png]")
                    elif "text/plain" in data:
                        txt = data["text/plain"]
                        outs.append("".join(txt) if isinstance(txt, list) else str(txt))
            if outs:
                text = "\n".join(outs)
                f2 = fence_for(text)
                parts.append(f"**Вывод:**\n{f2}text\n{text}\n{f2}\n")
    parts.append(f"<!-- сводка: ячеек {len(cells)}, markdown {n_md}, код {n_code}, "
                 f"изображений-выводов {n_img}, ошибок исполнения {n_err} -->\n")
    return "\n".join(parts), {"cells": len(cells), "code": n_code, "md": n_md,
                              "images": n_img, "errors": n_err}


# ---------------------------------------------------------------- сборка
def plan() -> list[dict]:
    """Возвращает список передаваемых файлов: оригинал → путь в пакете."""
    items: list[dict] = []
    for rel in tracked():
        full = os.path.join(ROOT, rel)
        ext = os.path.splitext(rel)[1].lower()
        base = os.path.basename(rel)
        if ext in AS_IS or rel.endswith((".gitignore", ".gitkeep")):
            if base == ".gitignore":
                target, how = rel + ".txt", "переименован в text/plain"
            elif base == ".gitkeep":
                target, how = rel + ".txt", "переименован в text/plain"
            else:
                target, how = rel, "передан как есть"
            items.append({"src": rel, "dst": target, "how": how, "render": None})
        elif ext == ".py":
            items.append({"src": rel, "dst": rel + ".txt", "how": "переименован в text/plain",
                          "render": None})
        elif ext == ".ipynb":
            items.append({"src": rel, "dst": rel + ".json",
                          "how": "переименован в application/json (дословная копия)",
                          "render": rel + ".md"})
        else:
            items.append({"src": rel, "dst": rel, "how": "НЕ ПОДДЕРЖИВАЕТСЯ", "render": None})
    return items


def main() -> None:
    if os.path.isdir(os.path.join(ROOT, OUT_DIR)):
        shutil.rmtree(os.path.join(ROOT, OUT_DIR))

    items = plan()
    for it in items:
        src_full = os.path.join(ROOT, it["src"])
        dst_full = os.path.join(ROOT, OUT_DIR, it["dst"])
        os.makedirs(os.path.dirname(dst_full), exist_ok=True)
        shutil.copyfile(src_full, dst_full)
        it["size"] = os.path.getsize(src_full)
        it["sha256"] = sha256_file(src_full)
        it["mime"] = mime_of(it["dst"])
        if it["render"]:
            md, stats = notebook_to_md(it["src"])
            rpath = os.path.join(ROOT, OUT_DIR, it["render"])
            with open(rpath, "w", encoding="utf-8", newline="") as f:
                f.write(md)
            it["render_size"] = os.path.getsize(rpath)
            it["render_sha256"] = sha256_file(rpath)
            it["render_stats"] = stats

    # ---------------- индекс
    lines: list[str] = []

    def w(t: str = "") -> None:
        lines.append(t)

    w("# ПАКЕТ №2 ДЛЯ АГЕНТА №2 — СОСТАВ И СООТВЕТСТВИЕ ТИПАМ ФАЙЛОВ")
    w()
    w("**От:** Агент №1. **Дата:** 2026-10-24.")
    w()
    w("Пакет собран под фактический список поддерживаемых Агентом №2 типов:")
    w()
    w("```text")
    w("image/png, image/webp, image/jpeg, image/gif,")
    w("text/plain, text/markdown, text/csv, text/html, text/xml,")
    w("text/css, text/javascript, application/json, application/xml,")
    w("application/javascript, application/pdf")
    w("```")
    w()
    w("Правила сборки: `.md`/`.txt`/`.csv`/`.json`/`.png`/`.pdf` передаются **как есть**;")
    w("`.py` — как `text/plain` (`<имя>.py.txt`); `.ipynb` — как `application/json`")
    w("(`<имя>.ipynb.json`, дословно) плюс читаемый `text/markdown` рендер; служебные")
    w("`.gitignore`/`.gitkeep` — как `text/plain`. Контейнеры (`.zip`) **не передаются**:")
    w("их тип не поддерживается. Все копии побайтовые.")
    w()
    w("## Состав пакета")
    w()
    w(f"Файлов проекта: **{len(items)}**. Файлов в пакете: "
      f"**{len(items) + sum(1 for i in items if i['render'])}** "
      f"(+ индекс `00_INDEX.md`).")
    w()
    w("| № | Оригинал | Путь в пакете | MIME | Как передан | Байт | SHA-256 (16) |")
    w("|---:|---|---|---|---|---:|---|")
    for i, it in enumerate(items, 1):
        w(f"| {i} | `{it['src']}` | `{it['dst']}` | `{it['mime']}` | {it['how']} | "
          f"{it['size']:,} | `{it['sha256'][:16]}` |")
        if it["render"]:
            w(f"| {i}· | `{it['src']}` | `{it['render']}` | `text/markdown` | "
              f"читаемый рендер notebook | {it['render_size']:,} | "
              f"`{it['render_sha256'][:16]}` |")
    w()
    w("## Дерево пакета")
    w()
    w("```text")
    paths = sorted([it["dst"] for it in items] + [it["render"] for it in items if it["render"]])
    tree: dict = {}
    for d in paths:
        node = tree
        parts = d.split("/")
        for p in parts[:-1]:
            node = node.setdefault(p + "/", {})
        node[parts[-1]] = None

    def walk(node: dict, prefix: str = "") -> None:
        for name in sorted(node):
            w(prefix + name)
            if node[name]:
                walk(node[name], prefix + "    ")

    walk(tree)
    w("```")
    w()
    w("## Проверка")
    w()
    w("```bash")
    w("python scripts/check_transfer_packet.py   # MIME ⊂ поддерживаемых, SHA-256, полнота")
    w("python scripts/make_transfer_packet.py    # пересборка пакета")
    w("```")
    w()
    w("Пояснение для Агента №2 и запрос одобрения на оформление итогового документа —")
    w("`docs/report/PACKET_2_TRANSFER_NOTE.md`.")
    w()

    with open(os.path.join(ROOT, OUT_DIR, "00_INDEX.md"), "w", encoding="utf-8",
              newline="") as f:
        f.write("\n".join(lines) + "\n")

    total = 0
    for dirpath, _, names in os.walk(os.path.join(ROOT, OUT_DIR)):
        for n in names:
            total += os.path.getsize(os.path.join(dirpath, n))
    print(f"пакет: {OUT_DIR} — {len(items)} исходных файлов, "
          f"{len(items) + sum(1 for i in items if i['render']) + 1} файлов в пакете, "
          f"{total:,} байт")
    bad = [it["dst"] for it in items if it["mime"] == "НЕ ПОДДЕРЖИВАЕТСЯ"]
    print("неподдерживаемых типов:", len(bad), bad if bad else "")


if __name__ == "__main__":
    main()
