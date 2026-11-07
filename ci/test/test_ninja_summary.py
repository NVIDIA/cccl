# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Check timing reports for Ninja versions used by the CI images."""

import subprocess
import sys
from pathlib import Path

import pytest


@pytest.mark.parametrize("version", [6, 7])
def test_ninja_timing_log_versions(tmp_path, version):
    log = tmp_path / ".ninja_log"
    log.write_text(
        f"# ninja log v{version}\n"
        "0\t1000\t0\tobj/test.obj\t123456789abcdef0\n"
        "1000\t1500\t0\tbin/test.exe\tfedcba9876543210\n"
    )
    script = Path(__file__).resolve().parents[1] / "ninja_summary.py"
    result = subprocess.run(
        [sys.executable, str(script), "--log-file", str(log)],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "2 build steps completed" in result.stdout
    assert "1.5 s weighted time" in result.stdout
