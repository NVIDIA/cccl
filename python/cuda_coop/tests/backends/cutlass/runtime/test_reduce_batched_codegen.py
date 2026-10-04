# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Inspect final batched-reduction kernels after numerical checks pass.

Logical width 8 and the full 32-lane warp use the same runtime harness. The
final SASS must contain no generated wrapper symbol or CALL instruction,
which shows the wrapper was inlined. It must also contain no shared-memory
loads or stores (LDS or STS), or barriers (BAR). These checks cover the whole
kernel, including Load and Store; other resource usage is not measured.
"""

import re
import shutil
import subprocess

import pytest

pytest.importorskip("cutlass")

from cutlass.base_dsl.compiler import DumpDir, KeepCUBIN

from tests.backends.cutlass.runtime.test_reduce_batched import _run

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.runtime, pytest.mark.gpu]


@pytest.mark.parametrize("width", (8, 32))
def test_final_cubin(tmp_path, width):
    """Check that the kernel inlines the wrapper and stays in registers.

    The runtime harness first checks outputs and input preservation. Then the
    retained SASS must not name the generated wrapper or contain any CALL,
    LDS, STS, or BAR instruction.
    """

    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(width=width, compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))))
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output(
            [tool, "--dump-sass", str(cubin)], text=True
        )
        assert "cuda_coop_cutlass_reduce_batched_" not in sass
        assert re.search(r"\b(CALL|LDS|STS|BAR)\b", sass) is None
