# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Execute both passes of the downloadable CUTLASS debugger example."""

import os
import subprocess
import sys

import pytest

from tests.support.paths import PACKAGE_ROOT, REPO_ROOT

pytest.importorskip("cutlass")

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("algorithm", ("direct", "transpose"))
def test_debugger_walkthrough(algorithm, tmp_path):
    env = os.environ.copy()
    env.update(
        PYTHONPATH=os.pathsep.join(
            filter(None, (str(PACKAGE_ROOT), env.get("PYTHONPATH")))
        ),
        CUDA_COOP_CCCL_ROOT=str(REPO_ROOT),
        CUDA_COOP_DISABLE_AUTO_DSL_REGISTRATION="0",
        CUDA_COOP_SOURCE_DUMP_DIR=str(tmp_path),
    )
    result = subprocess.run(
        [
            sys.executable,
            str(REPO_ROOT / "docs/python/coop/cutlass_debugger_walkthrough.py"),
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
