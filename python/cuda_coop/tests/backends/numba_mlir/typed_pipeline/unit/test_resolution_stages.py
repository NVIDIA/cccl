# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Specialization planning and the scratch layout, on stub inputs.

Neither needs NVRTC, a GPU or a compiled kernel: planning reads only each
provider's storage ABI, and the layout takes plain sizes.
"""

from __future__ import annotations

from types import SimpleNamespace

import pytest

from cuda.coop.numba_mlir._compiler._operations import StorageABI
from cuda.coop.numba_mlir._pipeline import _link, _specialize, _storage
from cuda.coop.numba_mlir._pipeline._errors import TypedPipelineError
from cuda.coop.numba_mlir._pipeline._plan import Backing, Region
from cuda.coop.numba_mlir._pipeline._specialize import LaunchBounds

BLOCK = (128, 1, 1)
LIMIT = _storage.STATIC_SHARED_LIMIT


def _provider(storage_abi, digest="digest"):
    return SimpleNamespace(
        storage_abi=storage_abi, c_name="stub", _private_symbol_digest=digest
    )


SCRATCH = _provider(StorageABI.LEADING_POINTER, "scratch")
FREE = _provider(StorageABI.NONE, "free")


def test_barriers_follow_scratch_use():
    plan = _specialize.plan(
        [FREE, SCRATCH], [FREE, SCRATCH, FREE, SCRATCH], BLOCK, {}
    )
    assert plan.scratch == (False, True, False, True)
    assert plan.barrier_after == plan.scratch


def test_launch_bounds_default_to_the_block_thread_count():
    assert _specialize.plan_launch_bounds((32, 4, 1), {}) == LaunchBounds(
        128, rebind=True
    )


def test_explicit_launch_bounds_are_kept_and_checked_against_the_block():
    bounds = {"launch_bounds": (256, 2)}
    assert _specialize.plan_launch_bounds(BLOCK, bounds) == LaunchBounds(256)
    with pytest.raises(TypedPipelineError, match="more than launch_bounds"):
        _specialize.plan_launch_bounds((512, 1, 1), bounds)


def test_max_registers_leaves_launch_bounds_unset():
    options = {"max_registers": 32}
    assert _specialize.plan_launch_bounds(BLOCK, options) == LaunchBounds(None)


def test_one_region_as_large_as_the_largest_need_not_their_sum():
    layout = _storage.layout([(1024, 16), None, (512, 8)], 0)
    assert layout.region == Region("implicit.0", 0, 1024, 16)
    assert layout.backing == Backing("static", 1024, 16)
    assert layout.scratch == (True, False, True)


def test_backing_rounds_up_to_its_alignment_of_at_least_16():
    assert _storage.layout([(100, 4)], 0).backing == Backing("static", 112, 16)
    wide = _storage.layout([(100, 32)], 0)
    assert wide.backing == Backing("static", 128, 32)
    assert wide.region == Region("implicit.0", 0, 100, 32)


def test_no_scratch_needs_no_backing():
    layout = _storage.layout([None, None], 4096)
    assert (layout.backing, layout.region) == (None, None)
    assert layout.scratch == (False, False)


def test_static_limit_includes_the_kernels_own_shared_bytes():
    _storage.layout([(LIMIT - 1024, 16)], 1024)
    with pytest.raises(
        TypedPipelineError,
        match=f"needs {LIMIT - 1024} bytes of shared memory, which with the "
        "kernel's own 1040 bytes exceeds",
    ):
        _storage.layout([(LIMIT - 1024, 16)], 1040)


def test_compiled_layouts_must_agree_with_the_planned_scratch(monkeypatch):
    bundle = SimpleNamespace(layouts={"free": (64, 16)})
    monkeypatch.setattr(_link, "compile_bundle", lambda *args: bundle)
    plan = _specialize.plan([FREE], [FREE], BLOCK, {})
    with pytest.raises(AssertionError, match="disagrees"):
        _storage.analyze(plan, 0, identity=None, context=None)
