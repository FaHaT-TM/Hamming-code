#!/usr/bin/env python3
"""Проверка пакета `docs/packet2_pkg/` перед передачей Агенту №2.

Проверяется:

1. **Тип каждого файла пакета** входит в список поддерживаемых Агентом №2
   (по расширению → MIME);
2. **Побайтовость**: SHA-256 файла пакета совпадает с SHA-256 оригинала
   (для `.py.txt`, `.ipynb.json`, `text/plain`-переименований и скопированных как есть);
3. **Полнота**: у каждого файла проекта есть представление в пакете, и в пакете нет
   файлов, которых нет в проекте;
4. **Рендер notebook**: сводка по ячейкам в `.md` совпадает с `.ipynb`.

Запуск:
    python scripts/check_transfer_packet.py
"""
from __future__ import annotations

import hashlib
import json
import os
import re
import subprocess
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT_DIR = "docs/packet2_pkg"

ALLOWED = {
    ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv",
    ".html": "text/html", ".xml": "text/xml", ".css": "text/css",
    ".js": "text/javascript", ".json": "application/json",
    ".pdf": "application/pdf",
}

SKIP_PREFIX = ("docs/packet2_pkg/", "docs/report/packet2_for_agent2.zip",
               "docs/report/MANIFEST.md",
               "results/data/transfer_check.json", "results/data/transfer_archive.json")
NB_RE = re.compile(r"ячеек (\d+), markdown (\d+), код (\d+), изображений-выводов (\d+), "
                   r"ошибок исполнения (\d+)")


def sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def tracked() -> list[str]:
    out = subprocess.run(["git", "ls-files", "-z"], cwd=ROOT, capture_output=True, check=True)
    return sorted(p for p in out.stdout.decode("utf-8").split("\0")
                  if p and not p.startswith(SKIP_PREFIX))


def expected_targets(rel: str) -> list[str]:
    """Какие файлы пакета должны соответствовать оригиналу."""
    base = os.path.basename(rel)
    ext = os.path.splitext(rel)[1].lower()
    if base in (".gitignore", ".gitkeep"):
        return [rel + ".txt"]
    if ext == ".py":
        return [rel + ".txt"]
    if ext == ".ipynb":
        return [rel + ".json", rel + ".md"]
    return [rel]


def main() -> int:
    files = tracked()
    problems: list[str] = []
    by_mime: dict[str, int] = {}
    bytes_total = 0
    verified = 0

    produced: set[str] = set()
    for rel in files:
        origin = open(os.path.join(ROOT, rel), "rb").read()
        origin_sha = sha256(origin)
        for target in expected_targets(rel):
            produced.add(target)
            path = os.path.join(ROOT, OUT_DIR, target)
            if not os.path.exists(path):
                problems.append(f"нет файла в пакете: {target} (оригинал {rel})")
                continue
            ext = os.path.splitext(target)[1].lower()
            mime = ALLOWED.get(ext)
            if mime is None:
                problems.append(f"{target}: MIME для расширения {ext!r} не поддерживается")
                continue
            by_mime[mime] = by_mime.get(mime, 0) + 1
            size = os.path.getsize(path)
            bytes_total += size

            if target == rel + ".md" and rel.endswith(".ipynb"):
                with open(path, encoding="utf-8") as f:
                    text = f.read()
                with open(os.path.join(ROOT, rel), encoding="utf-8") as f:
                    cells = json.load(f).get("cells", [])
                exp = (len(cells),
                       sum(1 for c in cells if c.get("cell_type") == "markdown"),
                       sum(1 for c in cells if c.get("cell_type") == "code"))
                m = NB_RE.search(text)
                if not m or (int(m.group(1)), int(m.group(2)), int(m.group(3))) != exp:
                    problems.append(f"{target}: сводка по ячейкам не совпадает с {rel}")
                else:
                    verified += 1
                continue

            copy_sha = hashlib.sha256(open(path, "rb").read()).hexdigest()
            if copy_sha != origin_sha:
                problems.append(f"{target}: содержимое не совпадает с {rel} побайтово")
            else:
                verified += 1

    # обратная полнота
    in_package: set[str] = set()
    for dirpath, _, names in os.walk(os.path.join(ROOT, OUT_DIR)):
        for n in names:
            rel = os.path.relpath(os.path.join(dirpath, n), os.path.join(ROOT, OUT_DIR))
            in_package.add(rel.replace(os.sep, "/"))
    for extra in sorted(in_package - produced - {"00_INDEX.md"}):
        problems.append(f"лишний файл в пакете: {extra}")
    for miss in sorted(produced - in_package):
        problems.append(f"отсутствует в пакете: {miss}")

    index = os.path.join(ROOT, OUT_DIR, "00_INDEX.md")
    if not os.path.exists(index):
        problems.append("нет 00_INDEX.md")

    print(f"файлов проекта:      {len(files)}")
    print(f"файлов в пакете:     {len(in_package)} (включая 00_INDEX.md)")
    print(f"побайтово сверено:   {verified}")
    print(f"объём пакета:        {bytes_total:,} байт")
    print("распределение по MIME:")
    for mime, cnt in sorted(by_mime.items()):
        print(f"  {mime:<22} {cnt}")
    unsupported = sorted(set(by_mime) - set(ALLOWED.values()))
    if unsupported:
        problems.append(f"типы вне списка поддержки: {unsupported}")
    print(f"расхождений:         {len(problems)}")
    for p in problems:
        print("  ! " + p)
    print("ИТОГ:", "OK — все типы поддерживаются, копии побайтовые, состав полный"
          if not problems else "ЕСТЬ РАСХОЖДЕНИЯ")

    json.dump({"files": len(files), "package_files": len(in_package), "verified": verified,
               "package_bytes": bytes_total, "by_mime": by_mime, "problems": problems,
               "allowed_mime": sorted(ALLOWED.values()), "ok": not problems},
              open(os.path.join(ROOT, "results/data/transfer_check.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)
    return 0 if not problems else 1


if __name__ == "__main__":
    sys.exit(main())
