# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Run the standalone CuTe DSL integration example from this checkout.

A subprocess runs the documentation script with an isolated interpreter.
The script must print its verification line, and it must do so without
importing ``cuda.coop.cutlass``: the example exists to show the CuTe
integration mechanisms with none of the package's abstractions. The test
does not inspect generated code; the example's documentation page records
what the final cubin contains.
"""

import subprocess
import sys

import pytest

from tests.support.paths import REPO_ROOT

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]

_SCRIPT = REPO_ROOT / "docs/python/coop_cutlass/cutlass_integration_example.py"


def test_cutlass_integration_example():
    result = subprocess.run(
        [
            sys.executable,
            "-I",
            "-c",
            (
                f"import sys; sys.path[:] = {sys.path!r}; "
                "import runpy; "
                "runpy.run_path(sys.argv.pop(1), run_name='__main__'); "
                "assert 'cuda.coop.cutlass' not in sys.modules"
            ),
            str(_SCRIPT),
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert "block sum verified: 3008" in result.stdout
