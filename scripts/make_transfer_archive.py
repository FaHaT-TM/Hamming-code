#!/usr/bin/env python3
"""Архив пакета для Агента №2: `docs/report/packet2_for_agent2.zip`.

Внутрь архива попадает **весь** пакет `docs/packet2_pkg/` — то есть все файлы проекта
в поддерживаемых Агентом №2 MIME-типах (`docs/packet2_pkg/00_INDEX.md` — точка входа) —
и манифест `MANIFEST.md` с размерами, MIME и SHA-256 каждого файла.

Архив детерминированный: фиксированные даты записей и порядок файлов, поэтому
пересборка без изменения содержимого даёт тот же SHA-256.

После записи архив перечитывается и проверяется: число файлов, побайтовое совпадение
с пакетом, отсутствие неподдерживаемых типов.

Запуск:
    python scripts/make_transfer_packet.py     # сначала пакет
    python scripts/make_transfer_archive.py    # затем архив
"""
from __future__ import annotations

import hashlib
import json
import os
import zipfile

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PKG_DIR = "docs/packet2_pkg"
ZIP_PATH = "docs/report/packet2_for_agent2.zip"
MANIFEST_PATH = "docs/report/MANIFEST.md"

ALLOWED = {
    ".png": "image/png", ".webp": "image/webp", ".jpg": "image/jpeg",
    ".jpeg": "image/jpeg", ".gif": "image/gif",
    ".txt": "text/plain", ".md": "text/markdown", ".csv": "text/csv",
    ".html": "text/html", ".xml": "text/xml", ".css": "text/css",
    ".js": "text/javascript", ".json": "application/json",
    ".pdf": "application/pdf",
}


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def mime_of(path: str) -> str:
    return ALLOWED.get(os.path.splitext(path)[1].lower(), "НЕ ПОДДЕРЖИВАЕТСЯ")


# дополнительные поддерживаемые файлы, которые не входят в пакет, но полезны при передаче
EXTRA = ["results/data/transfer_check.json"]


def package_files() -> list[str]:
    out: list[str] = []
    base = os.path.join(ROOT, PKG_DIR)
    for dirpath, _, names in os.walk(base):
        for n in sorted(names):
            rel = os.path.relpath(os.path.join(dirpath, n), base).replace(os.sep, "/")
            out.append(rel)
    return sorted(out)


def build_manifest(files: list[str]) -> str:
    lines: list[str] = []

    def w(text: str = "") -> None:
        lines.append(text)
    w("# МАНИФЕСТ АРХИВА ДЛЯ АГЕНТА №2")
    w()
    w("**От:** Агент №1. **Дата:** 2026-10-24.")
    w()
    w("Архив содержит весь проект в поддерживаемых Агентом №2 MIME-типах:")
    w("`image/png`, `text/plain`, `text/markdown`, `text/csv`, `application/json`,")
    w("`application/pdf`. Файлы проекта в неподдерживаемых типах переименованы без")
    w("изменения содержимого: `.py` → `<имя>.py.txt`, `.ipynb` → `<имя>.ipynb.json`")
    w("(плюс читаемый `<имя>.ipynb.md`), служебные `.gitignore`/`.gitkeep` → `.txt`.")
    w()
    w("**Точка входа:** `00_INDEX.md` — состав, MIME, размеры, SHA-256, дерево пакета.")
    w("**Пояснение и запрос одобрения:** `docs/report/PACKET_2_TRANSFER_NOTE.md`.")
    w()
    w(f"Файлов в архиве (без этого манифеста): **{len(files)}**.")
    w()
    w("Внутри архива нет ни одного файла в неподдерживаемом типе: расширения только"
      " `.png`, `.pdf`, `.csv`, `.json`, `.md`, `.txt`.")
    w()
    w("## Содержимое")
    w()
    w("| № | Файл | MIME | Байт | SHA-256 |")
    w("|---:|---|---|---:|---|")
    for i, rel in enumerate(files, 1):
        full = (os.path.join(ROOT, rel) if rel in EXTRA
                else os.path.join(ROOT, PKG_DIR, rel))
        size = os.path.getsize(full)
        sha = sha256_bytes(open(full, "rb").read())
        w(f"| {i} | `{rel}` | `{mime_of(rel)}` | {size:,} | `{sha}` |")
    w()
    w("## Проверка целостности")
    w()
    w("SHA-256 каждого файла указан в таблице выше и проверяется пересчётом:")
    w()
    w("```bash")
    w("python scripts/check_transfer_packet.py   # пакет: MIME, побайтовость, полнота")
    w("python scripts/make_transfer_archive.py   # архив: самопроверка после записи")
    w("```")
    w()
    w("SHA-256 самого архива печатается сборкой и сохраняется в")
    w("`results/data/transfer_archive.json`. Внутрь архива это значение не включается:")
    w("манифест — часть архива, поэтому его изменение меняло бы SHA-256 архива")
    w("(самоссылочность). Сборка детерминированная: повторный запуск без изменения")
    w("файлов даёт тот же SHA-256.")
    w()
    return "\n".join(lines)


def main() -> None:
    files = package_files() + [e for e in EXTRA if os.path.exists(os.path.join(ROOT, e))]
    if not files:
        raise SystemExit(f"пакет пуст: сначала запустите scripts/make_transfer_packet.py")

    bad = [f for f in files if mime_of(f) == "НЕ ПОДДЕРЖИВАЕТСЯ"]
    if bad:
        raise SystemExit(f"в пакете есть неподдерживаемые типы: {bad}")

    zip_full = os.path.join(ROOT, ZIP_PATH)
    os.makedirs(os.path.dirname(zip_full), exist_ok=True)

    manifest = build_manifest(files)

    def write_zip(manifest_text: str) -> str:
        with zipfile.ZipFile(zip_full, "w", zipfile.ZIP_DEFLATED) as zf:
            for rel in files:
                info = zipfile.ZipInfo.from_file(
                    os.path.join(ROOT, rel) if rel in EXTRA
                    else os.path.join(ROOT, PKG_DIR, rel), arcname=rel)
                info.date_time = (2026, 10, 24, 0, 0, 0)
                info.compress_type = zipfile.ZIP_DEFLATED
                info.external_attr = 0o644 << 16
                with open(os.path.join(ROOT, rel) if rel in EXTRA
                          else os.path.join(ROOT, PKG_DIR, rel), "rb") as f:
                    zf.writestr(info, f.read())
            info = zipfile.ZipInfo("MANIFEST.md", date_time=(2026, 10, 24, 0, 0, 0))
            info.compress_type = zipfile.ZIP_DEFLATED
            info.external_attr = 0o644 << 16
            zf.writestr(info, manifest_text)
        return sha256_bytes(open(zip_full, "rb").read())

    zip_sha = write_zip(manifest)

    # фиксируем манифест в проекте (вне пакета, чтобы не было рекурсии)
    with open(os.path.join(ROOT, MANIFEST_PATH), "w", encoding="utf-8", newline="") as f:
        f.write(manifest)

    # ---------------- самопроверка: перечитываем архив
    problems: list[str] = []
    with zipfile.ZipFile(zip_full) as zf:
        names = zf.namelist()
        if len(names) != len(files) + 1:
            problems.append(f"файлов в архиве {len(names)}, ожидалось {len(files) + 1}")
        if zf.testzip() is not None:
            problems.append(f"повреждённая запись: {zf.testzip()}")
        for rel in files:
            if rel not in names:
                problems.append(f"нет в архиве: {rel}")
                continue
            src_of = (os.path.join(ROOT, rel) if rel in EXTRA
                      else os.path.join(ROOT, PKG_DIR, rel))
            if zf.read(rel) != open(src_of, "rb").read():
                problems.append(f"содержимое в архиве отличается: {rel}")
        if "MANIFEST.md" not in names:
            problems.append("в архиве нет MANIFEST.md")
        unsupported = sorted({mime_of(n) for n in names if n != "MANIFEST.md"}
                             - set(ALLOWED.values()))
        if unsupported:
            problems.append(f"неподдерживаемые типы в архиве: {unsupported}")

    size = os.path.getsize(zip_full)
    by_mime: dict[str, int] = {}
    for rel in files:
        by_mime[mime_of(rel)] = by_mime.get(mime_of(rel), 0) + 1

    print(f"архив:      {ZIP_PATH} ({size:,} байт)")
    print(f"файлов:     {len(files)} + MANIFEST.md")
    print(f"SHA-256:    {zip_sha}")
    print("MIME:")
    for m, c in sorted(by_mime.items()):
        print(f"  {m:<22} {c}")
    print(f"расхождений: {len(problems)}")
    for p in problems:
        print("  ! " + p)
    print("ИТОГ:", "OK — архив полный, копии побайтовые, типы поддерживаемые"
          if not problems else "ЕСТЬ РАСХОЖДЕНИЯ")

    json.dump({"path": ZIP_PATH, "bytes": size, "sha256": zip_sha,
               "files": len(files) + 1, "by_mime": by_mime,
               "problems": problems, "ok": not problems},
              open(os.path.join(ROOT, "results/data/transfer_archive.json"), "w",
                   encoding="utf-8"), ensure_ascii=False, indent=1)


if __name__ == "__main__":
    main()
