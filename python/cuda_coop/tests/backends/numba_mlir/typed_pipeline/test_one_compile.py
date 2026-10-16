# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""A kernel's providers compile in one NVRTC call, during storage analysis.

Lowering compiles nothing, and a recompile of the same kernel reuses the
memoized bundle.
"""

from __future__ import annotations

from cuda import coop
from cuda.coop.numba_mlir._pipeline import testing

from ._support import BLOCK, CHIP, f32_in_out, requires_typed_pipeline

pytestmark = requires_typed_pipeline


def three_sites(source, output):
    block = coop.this_block()
    a = coop.ThreadData(4)
    b = coop.ThreadData(4)
    coop.load(block, source, a)
    coop.load(block, source, b)
    coop.store(block, output, a, algorithm="transpose")


def test_one_nvrtc_compile_per_kernel_and_none_on_a_recompile(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _nvrtc
    from cuda.coop.numba_mlir._pipeline import _link, _storage

    compiles = []
    after_storage = []
    compile_with_layouts = _nvrtc.compile_with_layouts
    analyze = _storage.analyze

    def counting_compile(*args, **kwargs):
        compiles.append(1)
        return compile_with_layouts(*args, **kwargs)

    def recording_analyze(*args, **kwargs):
        result = analyze(*args, **kwargs)
        after_storage.append(len(compiles))
        return result

    monkeypatch.setattr(_nvrtc, "compile_with_layouts", counting_compile)
    monkeypatch.setattr(_storage, "analyze", recording_analyze)
    _link.clear()

    first = testing.compile_kernel(
        three_sites, f32_in_out(), block=BLOCK, chip=CHIP
    )
    assert "bundle sites=3 specializations=2" in first.plan
    assert len(compiles) == 1
    assert after_storage == [1]

    second = testing.compile_kernel(
        three_sites, f32_in_out(), block=BLOCK, chip=CHIP
    )
    assert second.plan == first.plan
    assert len(compiles) == 1
    assert after_storage == [1, 1]
