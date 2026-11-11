# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import importlib
import json
import subprocess
import sys

from ....support.paths import TESTS_ROOT


def test_tile_public_typing_fixture_passes_strict_pyright() -> None:
    importlib.import_module("cuda.tile")
    fixture = TESTS_ROOT / "support" / "fixtures" / "typing_tile_surfaces.py"
    assert "# pyright: strict" in fixture.read_text(encoding="utf-8")
    completed = subprocess.run(
        [
            sys.executable,
            "-m",
            "pyright",
            str(fixture),
            "--pythonversion",
            "3.11",
            "--pythonpath",
            sys.executable,
            "--outputjson",
        ],
        check=False,
        cwd=TESTS_ROOT,
        capture_output=True,
        text=True,
    )
    report = json.loads(completed.stdout)

    assert report["version"] == "1.1.414"
    assert completed.returncode == 0, json.dumps(
        report.get("generalDiagnostics"), indent=2
    )
    assert report["summary"]["errorCount"] == 0
    assert report["summary"]["warningCount"] == 0
