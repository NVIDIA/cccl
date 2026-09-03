# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from types import SimpleNamespace

import pytest

pytest.importorskip("numba_cuda_mlir")

from numba_cuda_mlir import types
from numba_cuda_mlir.numba_cuda.compiler import run_frontend

from cuda.coop.numba_mlir._compiler._rewrite import (
    CoopSinglePhaseRewrite,
    CoopSinglePhaseRewriteError,
    CoopWholeFunctionPlanner,
)
from cuda.coop.numba_mlir._lowering import load as block_load

pytestmark = [pytest.mark.backend_numba_mlir, pytest.mark.unit]


@pytest.fixture(autouse=True)
def _fixed_provider_compute_capability(monkeypatch):
    from cuda.coop.numba_mlir import _types

    monkeypatch.setattr(
        _types.cuda,
        "get_current_device",
        lambda: SimpleNamespace(compute_capability=(9, 0)),
    )


class _TypingContext:
    def __init__(self):
        self.refresh_count = 0

    def refresh(self):
        self.refresh_count += 1


def _state(function, *, targetoptions):
    array_type = types.Array(types.int32, 1, "C")
    return SimpleNamespace(
        func_ir=run_frontend(function),
        args=(array_type, array_type),
        typingctx=_TypingContext(),
        typemap={},
        calltypes={},
        metadata={"targetoptions": targetoptions},
    )


def _first_block(state):
    return state.func_ir.blocks[sorted(state.func_ir.blocks)[0]]


def _match(state):
    rewrite = CoopSinglePhaseRewrite(state)
    rewrite._prepare_ltoir_bundle_for_matches = lambda _matches: None
    matched = rewrite.match(
        state.func_ir,
        _first_block(state),
        state.typemap,
        state.calltypes,
    )
    return rewrite, matched


@pytest.mark.parametrize(
    ("block", "expected"),
    [
        ((64, 1, 1), 64),
        ((8, 4, 1), (8, 4)),
        ((8, 4, 2), (8, 4, 2)),
    ],
)
def test_launch_block_dimensions_are_canonicalized(block, expected):
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._state = SimpleNamespace(
        metadata={"targetoptions": {"__launch_config__": {"block": block}}}
    )

    assert rewrite._infer_threads_per_block_from_launch_config() == expected


@pytest.mark.parametrize(
    "targetoptions",
    [
        {},
        {"launch_bounds": 128},
        {"launch_bounds": (256, 2)},
        {"__launch_config__": {}},
        {"__launch_config__": {"block": (0, 1, 1)}},
    ],
)
def test_launch_bounds_and_inexact_launches_do_not_infer_dimensions(targetoptions):
    rewrite = object.__new__(CoopSinglePhaseRewrite)
    rewrite._state = SimpleNamespace(metadata={"targetoptions": targetoptions})

    assert rewrite._infer_threads_per_block_from_launch_config() is None


def test_dim_alias_is_rewritten_to_threads_per_block():
    def kernel(source, output):
        return block_load(source, output, dtype=types.int32, dim=64)

    state = _state(
        kernel,
        targetoptions={"__launch_config__": {"block": (64, 1, 1)}},
    )
    rewrite, matched = _match(state)

    assert matched
    match = next(iter(rewrite._matches.values()))
    assert match.factory_kwargs["threads_per_block"] == 64
    assert "dim" not in match.factory_kwargs


def test_explicit_dimension_accepts_an_equivalent_exact_launch_shape():
    def kernel(source, output):
        return block_load(
            source,
            output,
            dtype=types.int32,
            threads_per_block=64,
        )

    state = _state(
        kernel,
        targetoptions={"__launch_config__": {"block": (64, 1, 1)}},
    )
    rewrite, matched = _match(state)

    assert matched
    match = next(iter(rewrite._matches.values()))
    assert match.factory_kwargs["threads_per_block"] == 64


def test_dim_alias_rejects_explicit_threads_per_block():
    def kernel(source, output):
        return block_load(
            source,
            output,
            dtype=types.int32,
            threads_per_block=32,
            dim=64,
        )

    state = _state(kernel, targetoptions={})

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match="received both 'threads_per_block' and its 'dim' alias",
    ):
        _match(state)


def test_explicit_dimension_rejects_an_exact_launch_mismatch():
    def kernel(source, output):
        return block_load(
            source,
            output,
            dtype=types.int32,
            threads_per_block=32,
        )

    state = _state(
        kernel,
        targetoptions={"__launch_config__": {"block": (64, 1, 1)}},
    )

    with pytest.raises(
        CoopSinglePhaseRewriteError,
        match=(
            r"factory 'load' received threads_per_block=32, but the exact "
            r"kernel launch block is \(64, 1, 1\)"
        ),
    ):
        _match(state)


def test_deferred_rewrite_leaves_device_helper_ir_intact():
    def device_function(source, output):
        return block_load(source, output, dtype=types.int32)

    state = _state(device_function, targetoptions={"device": True})
    before = {label: tuple(block.body) for label, block in state.func_ir.blocks.items()}
    rewrite = CoopSinglePhaseRewrite(state)

    assert not rewrite.match(
        state.func_ir,
        _first_block(state),
        state.typemap,
        state.calltypes,
    )
    assert rewrite._deferred_launch_dim_inference
    assert {
        label: tuple(block.body) for label, block in state.func_ir.blocks.items()
    } == before


def test_kernel_planner_retries_with_an_exact_launch(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite as rewrites

    def kernel(source, output):
        return block_load(source, output, dtype=types.int32)

    class _FakeInvocable:
        files = ()
        temp_storage_bytes = 1
        temp_storage_alignment = 1

        def __call__(self, source, output):
            del source, output

    state = _state(kernel, targetoptions={})
    invocable = _FakeInvocable()
    requests = []

    def require_exact_launch(requested_state):
        launch_config = {
            "block": (64, 1, 1),
            "grid": (1, 1, 1),
            "cluster": None,
        }
        requests.append(requested_state)
        requested_state.metadata["targetoptions"]["__launch_config__"] = launch_config
        return launch_config

    monkeypatch.setattr(rewrites, "require_launch_config", require_exact_launch)
    monkeypatch.setattr(
        CoopSinglePhaseRewrite,
        "_prepare_ltoir_bundle_for_matches",
        lambda self, matches: None,
    )
    monkeypatch.setattr(
        CoopSinglePhaseRewrite,
        "_materialize_invocable",
        lambda self, match: (invocable, False),
    )
    monkeypatch.setattr(
        CoopSinglePhaseRewrite,
        "_record_invocable_specialization",
        lambda self, value: None,
    )

    assert CoopWholeFunctionPlanner(state).run()
    assert requests == [state]
    assert state.typingctx.refresh_count == 1


@pytest.mark.parametrize(
    ("targetoptions", "launch_config", "expected_detail"),
    [
        ({}, {"block": (0, 1, 1)}, "invalid block=(0, 1, 1)"),
        ({}, None, "no __launch_config__ metadata"),
        ({}, {}, "contains no block shape"),
        (
            {"launch_bounds": 128},
            None,
            "launch_bounds=128 is only an upper bound",
        ),
    ],
)
def test_kernel_planner_reports_an_unresolved_launch_after_retry(
    monkeypatch,
    targetoptions,
    launch_config,
    expected_detail,
):
    from cuda.coop.numba_mlir._compiler import _rewrite as rewrites

    def kernel(source, output):
        return block_load(source, output, dtype=types.int32)

    state = _state(kernel, targetoptions=targetoptions)
    requests = []

    def require_unresolved_launch(requested_state):
        requests.append(requested_state)
        if launch_config is not None:
            requested_state.metadata["targetoptions"]["__launch_config__"] = (
                launch_config
            )
        return launch_config

    monkeypatch.setattr(rewrites, "require_launch_config", require_unresolved_launch)

    with pytest.raises(CoopSinglePhaseRewriteError) as exc_info:
        CoopWholeFunctionPlanner(state).run()

    message = str(exc_info.value)
    assert "coop operation 'load'" in message
    assert "could not infer an exact positive threads_per_block" in message
    assert expected_detail in message
    assert "compile-time constant launch shape" in message
    assert requests == [state]


def test_device_planner_defers_without_requesting_a_launch(monkeypatch):
    from cuda.coop.numba_mlir._compiler import _rewrite as rewrites

    def device_function(source, output):
        return block_load(source, output, dtype=types.int32)

    state = _state(device_function, targetoptions={"device": True})
    before = {label: tuple(block.body) for label, block in state.func_ir.blocks.items()}

    def unexpected_launch_request(_state):
        pytest.fail("device-function planning requested kernel launch metadata")

    monkeypatch.setattr(rewrites, "require_launch_config", unexpected_launch_request)

    assert not CoopWholeFunctionPlanner(state).run()
    assert {
        label: tuple(block.body) for label, block in state.func_ir.blocks.items()
    } == before
