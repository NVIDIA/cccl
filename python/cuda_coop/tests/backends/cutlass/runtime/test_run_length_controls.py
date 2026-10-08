# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check decoding traps, scratch limits, wrapper inlining, and scratch races.

Trap cases run in separate processes because a device trap leaves the CUDA
context unusable. Successful host checks precede final-code inspection.
Compute-sanitizer coverage is opt-in and limited to two reuse profiles.
"""

import os
import re
import shutil
import subprocess
import sys

import pytest

cutlass = pytest.importorskip("cutlass")

from cutlass.base_dsl.common import DSLRuntimeError
from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from tests.backends.cutlass.runtime.test_run_length import _run
from tests.support.paths import PACKAGE_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize(
    "arguments",
    (
        "invalid='negative', length_dtype=np.int64",
        "invalid='padding'",
        "invalid='overflow'",
        "invalid='wide-overflow'",
        "offset=-1, control_type=cutlass.Int64",
        "bulk=True, offset=-1, control_type=cutlass.Int64",
        "bulk=True, offset=(1 << 64) - 1",
        "bulk=True, capacity=32",
    ),
)
def test_invalid_runtime_controls_trap(arguments):
    """Reject invalid lengths, offsets, and capacity in child processes.

    Cases include negative lengths, a positive run after zero padding, and
    lengths above the Uint32 total limit. Window and destination offsets must
    not be negative. Bulk decoding also rejects offsets beyond the destination
    and destinations too small for the stream. A nonzero exit must include a
    CUDA trap or launch error. No check reads the destination after failure.
    """

    script = (
        "import cutlass, numpy as np\n"
        "from tests.backends.cutlass.runtime.test_run_length import _run\n"
        f"_run({arguments})\n"
        "raise AssertionError('invalid RLD call did not trap')\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [sys.executable, "-B", "-c", script],
        env=environment,
        capture_output=True,
        text=True,
        timeout=180,
        check=False,
    )
    output = completed.stdout + completed.stderr
    assert completed.returncode != 0, output
    assert any(
        error in output
        for error in (
            "ILLEGAL_INSTRUCTION",
            "LAUNCH_FAILED",
            "illegal instruction",
            "launch failed",
            "CUDA Driver call failed: 715",
            "CUDA Driver call failed: 719",
        )
    ), output


def test_undersized_storage():
    with pytest.raises((ValueError, DSLRuntimeError), match="capacity"):
        _run(sharing="shared", storage_bytes=1)


@pytest.mark.parametrize("bulk", (False, True))
def test_final_cubin(tmp_path, bulk):
    """Check that each generated window or bulk wrapper is fully inlined.

    After successful execution, the retained SASS must contain no generated
    Run Length Decode wrapper symbol and no CALL instruction.
    """

    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(bulk=bulk, compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))))
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [tool, "--dump-sass", str(cubin)], text=True
        )
        assert "cuda_coop_cutlass_run_length_" not in sass
        assert re.search(r"\bCALL\b", sass) is None


def test_scratch_reuse_racecheck():
    """Run two scratch-reuse profiles under opt-in race detection.

    Shared window storage uses automatic barriers. Exclusive bulk storage uses
    manual barriers. Each profile repeats three times in a child process, and
    the sanitizer must report zero hazards as well as a successful exit.
    """

    if os.environ.get("CUDA_COOP_RUN_RACECHECK") != "1":
        pytest.skip("set CUDA_COOP_RUN_RACECHECK=1 to run compute-sanitizer")
    tool = shutil.which("compute-sanitizer")
    if tool is None:
        pytest.skip("compute-sanitizer is required")
    script = (
        "from tests.backends.cutlass.runtime.test_run_length import _run\n"
        "_run(bulk=False, repeats=3, sharing='shared')\n"
        "_run(bulk=True, repeats=3, sharing='exclusive', auto_sync=False)\n"
    )
    environment = os.environ.copy()
    environment["PYTHONPATH"] = os.pathsep.join(
        filter(None, (str(PACKAGE_ROOT), environment.get("PYTHONPATH")))
    )
    completed = subprocess.run(
        [
            tool,
            "--tool",
            "racecheck",
            "--error-exitcode",
            "86",
            sys.executable,
            "-B",
            "-c",
            script,
        ],
        env=environment,
        capture_output=True,
        text=True,
        timeout=240,
        check=False,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    assert "RACECHECK SUMMARY: 0 hazards" in completed.stdout + completed.stderr
