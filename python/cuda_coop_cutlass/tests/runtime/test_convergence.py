# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run collectives between divergent branches with a process timeout."""

import subprocess
import sys

import cutlass
import numpy as np
import pytest
from cutlass import cute

import cuda.coop.cutlass as coop
from tests.runtime_support import device_array
from tests.support.paths import TESTS_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


def _exercise_convergence():
    for threads in (32, 64, 96, 256, 1024):
        for explicit_storage in (False, True):

            @cute.kernel
            def kernel(memory: cute.Pointer):
                block = coop.this_block()
                rank = block.rank_as(cutlass.Int32)
                output = cute.make_tensor(memory, cute.make_layout(2))
                if rank == 0:
                    output[1] = 7
                if cutlass.const_expr(explicit_storage):
                    scratch = coop.TempStorage(auto_sync=False)
                    total = coop.sum(block, rank, temp_storage=scratch)
                else:
                    total = coop.sum(block, rank)
                if rank == 0:
                    output[0] = total

            @cute.jit
            def launch(memory: cute.Pointer):
                kernel(memory).launch(grid=1, block=threads)

            observed = np.full(2, -1, dtype=np.int32)
            with device_array(observed) as output:
                launch(output)
            np.testing.assert_array_equal(
                observed, [threads * (threads - 1) // 2, 7]
            )


def test_block_collective_between_reused_divergent_predicates(tmp_path):
    # Isolate a possible barrier regression so it cannot hang the test runner.
    script = tmp_path / "convergence.py"
    script.write_text(
        "import sys, types\n"
        + f"sys.path[:] = {sys.path!r}\n"
        + "tests = types.ModuleType('tests')\n"
        + f"tests.__path__ = {[str(TESTS_ROOT)]!r}\n"
        + "sys.modules['tests'] = tests\n"
        + "from tests.runtime.test_convergence import _exercise_convergence\n"
        + "_exercise_convergence()\n"
    )
    result = subprocess.run(
        [sys.executable, "-I", "-B", str(script)],
        capture_output=True,
        text=True,
        check=False,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
