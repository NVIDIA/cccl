# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the debugger walkthrough and confirm that it dumps provider source.

A subprocess runs the script from this checkout with a private dump
directory. The script launches its kernel twice, and both launches must
report a verified copy. Each case uses one algorithm, direct or transpose.
The test does not attach a debugger or check breakpoints.
"""

import os
import subprocess
import sys

import pytest

from tests.support.paths import REPO_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("algorithm", ("direct", "transpose"))
def test_debugger_walkthrough(algorithm, tmp_path):
    env = os.environ.copy()
    env.update(
        CUDA_COOP_SOURCE_DUMP_DIR=str(tmp_path),
    )
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import sys; sys.path[:] = {sys.path!r}; "
                "import runpy; "
                "runpy.run_path(sys.argv.pop(1), run_name='__main__')"
            ),
            str(REPO_ROOT / "docs/python/coop_cutlass/debugger_walkthrough.py"),
            "--algorithm",
            algorithm,
        ],
        env=env,
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert f"Launch 1 ({algorithm}): copy verified" in result.stdout
    assert f"Launch 2 ({algorithm}): copy verified" in result.stdout
    assert list(tmp_path.glob("cuda_coop_cutlass_*.cu"))
