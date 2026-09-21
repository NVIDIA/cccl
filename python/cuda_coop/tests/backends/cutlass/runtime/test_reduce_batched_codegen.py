# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Inspect final linked code for the register-only batched provider."""

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
    tool = shutil.which("cuobjdump")
    if tool is None:
        pytest.skip("cuobjdump is required for final linked code inspection")
    _run(width=width, compile_options=(KeepCUBIN(True), DumpDir(str(tmp_path))))
    cubins = list(tmp_path.rglob("*.cubin"))
    assert cubins
    for cubin in cubins:
        sass = subprocess.check_output([tool, "--dump-sass", str(cubin)], text=True)
        assert "cuda_coop_cutlass_reduce_batched_" not in sass
        assert re.search(r"\b(CALL|LDS|STS|BAR)\b", sass) is None
