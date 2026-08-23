# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import numpy as np
import pytest
from numba_cuda_mlir import cuda, types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend
from numba_cuda_mlir.numbair_transforms import ir

import cuda.coop as common_coop
import cuda.coop.numba_mlir as coop
from cuda.coop.numba_mlir._compiler._rewrite import (
    CoopSinglePhaseRewrite,
    CoopSinglePhaseRewriteError,
    CoopWholeFunctionPlanner,
)


class _TypingContext:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


def _rewrite(function):
    func_ir = run_frontend(function)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        typingctx=typingctx,
        typemap={},
        calltypes={},
    )
    rewrite = CoopSinglePhaseRewrite(state)
    for label in sorted(func_ir.blocks):
        block = func_ir.blocks[label]
        while rewrite.match(func_ir, block, state.typemap, state.calltypes):
            block = rewrite.apply()
            func_ir.blocks[label] = block
    return func_ir, typingctx


def _call_targets(func_ir):
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._func_ir = func_ir
    targets = []
    for block in func_ir.blocks.values():
        rewrite._block_defs = {
            inst.target.name: inst.value
            for inst in block.body
            if isinstance(inst, ir.Assign)
        }
        for inst in block.body:
            value = getattr(inst, "value", None)
            if isinstance(value, ir.Expr) and value.op == "call":
                targets.append(rewrite._resolve_python_value(value.func))
    return targets


def test_qualified_storage_calls_lower_to_compiler_arrays():
    def kernel():
        data = coop.ThreadData(2, types.int32, alignas=16)
        scratch = coop.TempStorage(32, alignment=16)
        return data[0] + scratch[0]

    func_ir, typingctx = _rewrite(kernel)
    targets = _call_targets(func_ir)

    assert cuda.local.array in targets
    assert cuda.shared.array in targets
    assert typingctx.refresh_count == 1


@pytest.mark.parametrize(
    "alignment",
    [16, np.int64(16)],
    ids=["builtin-int", "index-integer"],
)
def test_thread_data_rewrite_matches_runtime_alignment_aliases(alignment):
    def kernel():
        data = coop.ThreadData(
            2,
            types.int32,
            alignas=alignment,
            alignment=alignment,
        )
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_thread_data_rewrite_rejects_conflicting_alignment_aliases():
    def kernel():
        return coop.ThreadData(2, types.int32, alignas=16, alignment=32)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="alignas and alignment must match when both are set",
    ):
        _rewrite(kernel)


def test_thread_data_rewrite_accepts_explicit_default_alignment():
    def kernel():
        data = coop.ThreadData(2, types.int32, alignment=None)
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_common_thread_data_uses_only_the_portable_signature():
    def kernel():
        data = common_coop.ThreadData(2, types.int32)
        return data[0]

    func_ir, _ = _rewrite(kernel)

    assert cuda.local.array in _call_targets(func_ir)


def test_common_thread_data_rejects_qualified_alignment_control():
    def kernel():
        return common_coop.ThreadData(2, types.int32, alignment=16)

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=r"cuda\.coop\.ThreadData got unexpected keyword.*alignment",
    ):
        _rewrite(kernel)


def test_whole_function_planner_reuses_the_foundation_rewrite():
    def kernel():
        return coop.ThreadData(2, types.int32)

    func_ir = run_frontend(kernel)
    typingctx = _TypingContext()
    state = SimpleNamespace(
        func_ir=func_ir,
        typingctx=typingctx,
        typemap={},
        calltypes={},
        metadata={"targetoptions": {}},
    )

    assert CoopWholeFunctionPlanner(state).run()
    assert cuda.local.array in _call_targets(func_ir)
    assert typingctx.refresh_count == 1


def test_unsized_temp_storage_waits_for_a_primitive_requirement():
    def kernel():
        return coop.TempStorage()

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="size_in_bytes must be specified until a cooperative primitive",
    ):
        _rewrite(kernel)


@pytest.mark.parametrize(
    "kwargs",
    [
        {"auto_sync": False},
        {"sharing": "exclusive"},
    ],
)
def test_nondefault_temp_storage_policy_waits_for_a_primitive(kwargs):
    if "auto_sync" in kwargs:

        def kernel():
            return coop.TempStorage(32, auto_sync=False)

    else:

        def kernel():
            return coop.TempStorage(32, sharing="exclusive")

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="non-default sharing or auto_sync requires a cooperative primitive",
    ):
        _rewrite(kernel)
