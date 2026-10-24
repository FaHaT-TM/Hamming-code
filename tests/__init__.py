"""Тесты пакета. Добавляет корень проекта в sys.path, чтобы работал `import src...`
при любом способе запуска (discover из корня, из tests/, из VS Code)."""

import os
import sys

_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)
