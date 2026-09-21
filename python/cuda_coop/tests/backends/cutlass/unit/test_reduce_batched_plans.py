# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Batched-reduction topology, provider identity, and failure transactions."""

from dataclasses import replace

import pytest

cutlass = pytest.importorskip("cutlass")

from cuda.coop import cutlass as cutlass_coop
from cuda.coop._core import LaunchFacts, this_block, this_warp
from cuda.coop.cutlass._compiler import _rendering, _state, _types
from cuda.coop.cutlass._lowering import _reduce_batched as lowering

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def _request(**kwargs):
    options = dict(
        group=this_warp(),
        launch=LaunchFacts(exact_block_dim=(8, 4, 2)),
        dtype=cutlass.Int32,
        batches=33,
        op="sum",
        output_layout="striped",
    )
    options.update(kwargs)
    return lowering._CubReduceBatchedRequest(
        lowering._make_reduce_batched_plan(**options), options["op"]
    )


@pytest.mark.parametrize("width", (1, 2, 4, 8, 16, 32))
@pytest.mark.parametrize("layout", ("striped", "blocked"))
def test_storage_free_abi(width, layout):
    request = _request(group=this_warp().group_by(width), output_layout=layout)
    assert request.outputs_per_thread == (33 + width - 1) // width
    assert request.plan.topology.instances == 64 // width
    source = _rendering.render_bundle_source([request])
    assert "storage_address" not in source
    assert "__shared__" not in source
    assert "__syncthreads" not in source
    assert "__syncwarp" not in source
    assert _rendering.bundle_scratch_layout_probes([request]) == {}
    assert request.implementation.template_arguments["SYNC_PHYSICAL_WARP"] == "false"


@pytest.mark.parametrize("dtype", tuple(_types.ALL_PROVIDER_TYPES))
def test_numeric_types(dtype):
    assert _request(dtype=dtype).operation.dtype is dtype


@pytest.mark.parametrize(
    "group", (this_block(), this_warp().group_by(3, exhaustive=False))
)
def test_unsupported_groups(group):
    with pytest.raises((ValueError, NotImplementedError), match="warp|power|width"):
        _request(group=group)


def test_invalid_launch_facts():
    for launch in (LaunchFacts(), LaunchFacts(exact_block_dim=(48, 1, 1))):
        with pytest.raises(
            (ValueError, NotImplementedError), match="block|warp|complete"
        ):
            _request(launch=launch)


def test_bitwise_float_rejected():
    with pytest.raises(TypeError, match="integer dtype"):
        _request(dtype=cutlass.Float32, op="bit_xor")


def test_request_plan_consistency():
    request = _request()
    with pytest.raises(ValueError, match="operator does not match"):
        lowering._CubReduceBatchedRequest(request.plan, "max")
    result = replace(
        request.plan.result,
        values=(replace(request.plan.result.values[0], items_per_member=3),),
    )
    with pytest.raises(ValueError, match="result does not match"):
        lowering._CubReduceBatchedRequest(replace(request.plan, result=result), "sum")


def test_request_identity():
    requests = [
        _request(),
        _request(op="max"),
        _request(output_layout="blocked"),
        _request(group=this_warp().group_by(8)),
    ]
    assert len({request.symbol_name for request in requests}) == len(requests)
    assert _request() == _request()


def test_allocation_rollback(monkeypatch):
    saved, restored = object(), []
    monkeypatch.setattr(_state, "snapshot_active_session_state", lambda: saved)
    monkeypatch.setattr(_state, "restore_active_session_state", restored.append)
    monkeypatch.setattr(_state, "register_request", lambda request: None)

    def fail(*args):
        raise RuntimeError("result allocation failed")

    monkeypatch.setattr(lowering, "_make_rmem_tensor", fail)
    value = cutlass_coop.ThreadData(3, dtype=cutlass.Int32, values=[1, 2, 3])
    with pytest.raises(RuntimeError, match="result allocation failed"):
        lowering.provider_reduce_batched(
            group=this_warp(),
            launch=LaunchFacts(exact_block_dim=(32, 1, 1)),
            value=value,
            op="sum",
            output_layout="striped",
        )
    assert restored == [saved]
    assert tuple(value) == (1, 2, 3)
