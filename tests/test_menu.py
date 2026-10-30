"""Тесты интерактивного меню scripts/menu.py (без запуска тяжёлых действий)."""

import contextlib
import io
import os
import sys
import unittest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "scripts"))
import menu  # noqa: E402


class TestMenuRegistry(unittest.TestCase):
    def test_all_group_actions_exist(self):
        grouped = [name for _, names in menu.GROUPS for name in names]
        self.assertEqual(sorted(grouped), sorted(menu.ACTIONS))

    def test_no_duplicate_actions_in_groups(self):
        grouped = [name for _, names in menu.GROUPS for name in names]
        self.assertEqual(len(grouped), len(set(grouped)))

    def test_script_files_exist(self):
        for name, action in menu.ACTIONS.items():
            cmd = action["cmd"]()
            for token in cmd:
                if token.endswith(".py"):
                    self.assertTrue(os.path.exists(token),
                                    f"{name}: нет файла {token}")

    def test_install_action_uses_requirements(self):
        cmd = menu.ACTIONS["install"]["cmd"]()
        self.assertIn("-m", cmd)
        self.assertIn("pip", cmd)
        self.assertTrue(any(t.endswith("requirements.txt") for t in cmd))

    def test_run_unknown_action_code(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = menu.main(["run", "no_such_action"])
        self.assertEqual(code, 2)

    def test_list_output_code(self):
        with contextlib.redirect_stdout(io.StringIO()):
            code = menu.main(["--list"])
        self.assertEqual(code, 0)


if __name__ == "__main__":
    unittest.main()
