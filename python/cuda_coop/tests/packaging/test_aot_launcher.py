# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import subprocess
import sys
import textwrap
from pathlib import Path

import pytest

from cuda.coop import _aot_cli

_PACKAGE_ROOT = Path(__file__).parents[2]


def test_launcher_rejects_windows_before_importing_cutlass(
    monkeypatch: pytest.MonkeyPatch,
    capsys: pytest.CaptureFixture[str],
) -> None:
    def unexpected_import(name: str) -> None:
        raise AssertionError(f"unexpected import: {name}")

    monkeypatch.setattr(_aot_cli.sys, "platform", "win32")
    monkeypatch.setattr(_aot_cli.importlib, "import_module", unexpected_import)

    assert _aot_cli.main(()) == 2
    error = capsys.readouterr().err
    assert "currently require Linux" in error
    assert "Traceback" not in error


def test_launcher_reports_missing_cutlass_extra_without_traceback() -> None:
    source = textwrap.dedent(
        """
        import importlib.abc
        import sys

        class BlockCutlass(importlib.abc.MetaPathFinder):
            def find_spec(self, fullname, path=None, target=None):
                if fullname == "cutlass" or fullname.startswith("cutlass."):
                    raise ModuleNotFoundError(
                        f"blocked {fullname}",
                        name=fullname,
                    )
                return None

        sys.meta_path.insert(0, BlockCutlass())
        from cuda.coop import _aot_cli
        raise SystemExit(_aot_cli.main(("inspect", "missing.coop-aot")))
        """
    )
    environment = os.environ.copy()
    environment["CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION"] = "1"
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(
            None,
            (str(_PACKAGE_ROOT), environment.get("PYTHONPATH")),
        )
    )

    completed = subprocess.run(
        [sys.executable, "-c", source],
        check=False,
        capture_output=True,
        text=True,
        env=environment,
    )

    assert completed.returncode == 2
    assert "cuda-coop[cutlass]" in completed.stderr
    assert "Traceback" not in completed.stderr


def test_console_script_uses_dependency_light_launcher() -> None:
    pyproject = (_PACKAGE_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert 'cuda-coop-aot = "cuda.coop._aot_cli:main"' in pyproject
