# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from __future__ import annotations

import numpy as np
import pytest

from ....support.paths import PACKAGE_ROOT

SOURCE_ROOT = PACKAGE_ROOT

from cutlass.base_dsl.typing import Float32, Int32, Uint8

import cuda.coop.cutlass as coop
from cuda.coop._core import (
    ArgumentBinding,
    GroupLoadStoreKind,
    GroupLoweringTarget,
    LaunchFacts,
    ResultVisibility,
)
from cuda.coop.cutlass._dsl import _cub_load_store_provider as provider
from cuda.coop.cutlass._dsl._load_store import (
    ScopedLoadStoreRoute,
    ScopedLoadStoreRouteDecision,
    classify_scoped_load_store_route,
    contiguous_layout_reason,
)
from cuda.coop.cutlass._dsl.block import _load_store as scoped_block_load_store
from cuda.coop.cutlass._dsl.warp import _load_store as scoped_warp_load_store
from cuda.coop.cutlass._group_load_store import _make_group_load_store_plan
from cuda.coop.cutlass._thread_group import _resolve_collective_group_from_launch
from cuda.coop.cutlass._value_metadata import (
    DefinedThreadDomain,
    attach_thread_data_metadata,
    metadata_for_group,
    thread_data_metadata,
)


def _plan(
    group,
    *,
    kind="load",
    dtype=Int32,
    items_per_thread=2,
    algorithm="direct",
    valid_items=ArgumentBinding.omitted(),
    oob_default=ArgumentBinding.omitted(),
    offset=ArgumentBinding.omitted(),
    source="cutlass_root",
):
    return _make_group_load_store_plan(
        group=group,
        launch=LaunchFacts(exact_block_dim=64),
        kind=GroupLoadStoreKind(kind),
        dtype=dtype,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items,
        oob_default=oob_default,
        offset=offset,
        source=source,
    )


def test_root_exports_explicit_group_load_store():
    assert coop.load.__module__ == "cuda.coop.cutlass._group_load_store"
    assert coop.store.__module__ == "cuda.coop.cutlass._group_load_store"
    assert "load" in coop.__all__
    assert "store" in coop.__all__


def test_root_load_uses_thread_data_count_and_tags_result_domain(
    monkeypatch, set_cutlass_launch_facts
):
    set_cutlass_launch_facts(64)
    calls = []

    def fake_load(**payload):
        calls.append(payload)
        return payload["output"]

    monkeypatch.setattr(provider, "provider_load", fake_load)
    output = coop.ThreadData(3)
    group = coop.this_block()

    result = coop.load(
        group,
        object(),
        output,
        valid_items=17,
        oob_default=0,
        offset=4,
    )

    assert result is output
    assert calls[0]["output"].items_per_thread == 3
    assert calls[0]["valid_items_binding"] == ArgumentBinding.static(17)
    assert calls[0]["oob_default_binding"] == ArgumentBinding.static(0)
    assert calls[0]["offset_binding"] == ArgumentBinding.static(4)
    metadata = thread_data_metadata(output)
    assert metadata is not None
    assert metadata.visibility is ResultVisibility.PER_MEMBER
    assert metadata.defined_domain == DefinedThreadDomain.all_callers()


@pytest.mark.parametrize(
    ("value", "expected_items"),
    [
        (Int32(7), 1),
        (coop.ThreadData.from_values(Int32(1), Int32(2)), 2),
    ],
)
def test_root_store_scalar_is_one_item_and_thread_data_is_sole_count(
    monkeypatch,
    set_cutlass_launch_facts,
    value,
    expected_items,
):
    set_cutlass_launch_facts(64)
    calls = []
    monkeypatch.setattr(
        provider, "provider_store", lambda **payload: calls.append(payload)
    )

    coop.store(
        coop.this_warp(),
        object(),
        value,
    )

    stored = calls[0]["value"]
    actual_items = stored.items_per_thread if isinstance(stored, coop.ThreadData) else 1
    assert actual_items == expected_items


def test_root_store_rejects_root_only_values_before_provider(
    monkeypatch, set_cutlass_launch_facts
):
    set_cutlass_launch_facts(64)
    monkeypatch.setattr(
        provider,
        "provider_store",
        lambda **payload: pytest.fail(f"provider called: {payload}"),
    )
    resolved = _resolve_collective_group_from_launch(
        coop.this_block(),
        LaunchFacts(exact_block_dim=64),
        feature="store",
    )
    value = attach_thread_data_metadata(
        coop.ThreadData.from_values(Int32(1)),
        metadata_for_group(resolved, visibility=ResultVisibility.GROUP_ROOT),
    )

    with pytest.raises(ValueError, match="defined only at group roots"):
        coop.store(
            coop.this_block(),
            object(),
            value,
        )


def test_root_load_store_frontend_errors_are_specific(set_cutlass_launch_facts):
    set_cutlass_launch_facts(64)
    with pytest.raises(TypeError, match="output must be ThreadData"):
        coop.load(
            coop.this_block(),
            object(),
            object(),
        )
    with pytest.raises(TypeError, match="oob_default requires valid_items"):
        coop.load(
            coop.this_block(),
            object(),
            coop.ThreadData(1),
            oob_default=0,
        )
    with pytest.raises(TypeError, match="valid_items must be an integer"):
        coop.store(
            coop.this_block(),
            object(),
            Int32(1),
            valid_items=True,
        )
    with pytest.raises(TypeError, match="valid_items must be an integer"):
        coop.store(
            coop.this_block(),
            object(),
            Int32(1),
            valid_items=1.5,
        )
    with pytest.raises(TypeError, match="oob_default must be numeric, not boolean"):
        coop.load(
            coop.this_block(),
            object(),
            coop.ThreadData(1),
            valid_items=1,
            oob_default=np.bool_(True),
        )
    with pytest.raises(ValueError, match="oob_default must be finite"):
        coop.load(
            coop.this_block(),
            object(),
            coop.ThreadData(1),
            valid_items=1,
            oob_default=float("inf"),
        )
    with pytest.raises(TypeError, match="oob_default must be a numeric scalar"):
        coop.load(
            coop.this_block(),
            object(),
            coop.ThreadData(1),
            valid_items=1,
            oob_default="zero",
        )
    with pytest.raises(ValueError, match="offset must be non-negative"):
        coop.store(
            coop.this_block(),
            object(),
            Int32(1),
            offset=-1,
        )
    with pytest.raises(ValueError, match="offset must fit a signed 64-bit integer"):
        coop.store(
            coop.this_block(),
            object(),
            Int32(1),
            offset=1 << 63,
        )
    with pytest.raises(
        NotImplementedError,
        match="physical block, physical-warp, and logical-warp",
    ):
        coop.store(
            coop.this_grid(),
            object(),
            Int32(1),
        )
    with pytest.raises(TypeError, match="unexpected keyword"):
        coop.load(
            coop.this_block(),
            object(),
            coop.ThreadData(1),
            items_per_thread=1,
        )
    with pytest.raises(NotImplementedError, match="complete"):
        set_cutlass_launch_facts(33)
        coop.load(
            coop.this_warp(),
            object(),
            coop.ThreadData(1),
        )


@pytest.mark.parametrize("primitive", ("load", "store"))
def test_root_load_store_rejects_private_launch_metadata(primitive):
    with pytest.raises(
        TypeError,
        match=r"unexpected keyword argument 'launch_metadata'",
    ):
        if primitive == "load":
            coop.load(
                coop.this_block(),
                object(),
                coop.ThreadData(1),
                launch_metadata={"block": 64},
            )
        else:
            coop.store(
                coop.this_block(),
                object(),
                Int32(1),
                launch_metadata={"block": 64},
            )


def test_root_load_store_requires_compiler_verified_exact_launch(monkeypatch):
    from cuda.coop.cutlass._dsl import _launch

    monkeypatch.setattr(
        _launch,
        "current_kernel_launch_facts",
        lambda: LaunchFacts(exact_block_dim=64),
    )
    monkeypatch.setattr(
        provider,
        "provider_store",
        lambda **payload: pytest.fail(f"provider called: {payload}"),
    )

    with pytest.raises(NotImplementedError, match="verified compiler launch facts"):
        coop.store(coop.this_block(), object(), Int32(1))


@pytest.mark.parametrize(
    ("private_scope", "scoped_frontend"),
    (
        (coop._block, scoped_block_load_store),
        (coop._warp, scoped_warp_load_store),
    ),
    ids=("block", "warp"),
)
def test_private_canonical_adapters_forward_launch_facts(
    monkeypatch,
    set_cutlass_launch_facts,
    private_scope,
    scoped_frontend,
):
    set_cutlass_launch_facts(None)
    calls = []

    def capture_load(**payload):
        calls.append(payload)
        return payload["output"]

    monkeypatch.setattr(provider, "provider_load", capture_load)
    monkeypatch.setattr(
        provider, "provider_store", lambda **payload: calls.append(payload)
    )
    monkeypatch.setattr(
        scoped_frontend,
        "classify_scoped_load_store_route",
        lambda *args, **kwargs: ScopedLoadStoreRouteDecision(
            route=ScopedLoadStoreRoute.CANONICAL_CUB,
            reason="test canonical route",
            exact_block_dim=(8, 4, 2),
        ),
    )

    launch_metadata = {"block": (8, 4, 2)}
    items = coop.ThreadData(2, dtype=Int32)
    assert (
        private_scope.load(
            object(),
            items,
            launch_metadata=launch_metadata,
        )
        is items
    )
    private_scope.store(
        object(),
        items,
        launch_metadata=launch_metadata,
    )

    assert [payload["launch"].exact_block_dim for payload in calls] == [
        (8, 4, 2),
        (8, 4, 2),
    ]


@pytest.mark.evidence_for("group.load", backend="cutlass", evidence="lowering")
@pytest.mark.evidence_for("group.store", backend="cutlass", evidence="lowering")
def test_plans_cover_block_and_physical_warp_cub_collectives():
    plans = {
        ("block", "load"): _plan(
            coop.this_block(),
            kind="load",
            algorithm="striped",
            valid_items=ArgumentBinding.runtime(),
            oob_default=ArgumentBinding.static(0),
            offset=ArgumentBinding.static(4),
        ).require_supported(),
        ("block", "store"): _plan(
            coop.this_block(),
            kind="store",
            valid_items=ArgumentBinding.runtime(),
        ).require_supported(),
        ("warp", "load"): _plan(
            coop.this_warp(),
            kind="load",
            items_per_thread=1,
        ).require_supported(),
        ("warp", "store"): _plan(
            coop.this_warp(),
            kind="store",
            items_per_thread=1,
            algorithm="vectorize",
            valid_items=ArgumentBinding.runtime(),
        ).require_supported(),
    }

    for (group_kind, operation), plan in plans.items():
        target = (
            GroupLoweringTarget.CUB_BLOCK
            if group_kind == "block"
            else GroupLoweringTarget.CUB_WARP
        )
        assert plan.target is target
        assert plan.provenance.cpp_class == (
            f"cub::{group_kind.title()}{operation.title()}"
        )
        if operation == "load":
            assert (
                plan.result.result_items_per_thread
                == plan.call.operation.items_per_thread
            )
        else:
            assert plan.result is None


def test_warp_rejects_block_only_algorithm_through_shared_plan():
    plan = _plan(
        coop.this_warp(),
        kind="load",
        algorithm="warp_transpose",
    )

    assert plan.target is GroupLoweringTarget.UNSUPPORTED
    with pytest.raises(NotImplementedError, match="does not support algorithm"):
        plan.require_supported()


def test_request_rejects_static_valid_count_past_group_tile():
    plan = _plan(
        coop.this_warp(),
        kind="store",
        items_per_thread=2,
        valid_items=ArgumentBinding.static(65),
    ).require_supported()

    with pytest.raises(ValueError, match=r"group tile size \(64\)"):
        provider._CubLoadStoreRequest(plan=plan, value_type=Int32)


def test_request_rejects_static_offset_past_signed_int64():
    plan = _plan(
        coop.this_block(),
        kind="store",
        offset=ArgumentBinding.static(1 << 63),
    ).require_supported()

    with pytest.raises(ValueError, match="fit a signed 64-bit integer"):
        provider._CubLoadStoreRequest(plan=plan, value_type=Int32)


@pytest.mark.parametrize(
    ("dtype", "value", "error", "message"),
    (
        (Uint8, -1, ValueError, "not representable"),
        (Uint8, 256, ValueError, "not representable"),
        (Int32, 1.5, TypeError, "dtype does not match"),
        (Float32, 1e39, ValueError, "not representable"),
        (Float32, float("inf"), ValueError, "must be finite"),
        (Float32, Float32(float("inf")), ValueError, "must be finite"),
        (Float32, Float32(float("nan")), ValueError, "must be finite"),
        (Int32, np.float32(1.0), TypeError, "memory dtype"),
    ),
)
def test_request_validates_static_oob_default_against_memory_dtype(
    dtype,
    value,
    error,
    message,
):
    plan = _plan(
        coop.this_block(),
        dtype=dtype,
        valid_items=ArgumentBinding.static(1),
        oob_default=ArgumentBinding.static(value),
    ).require_supported()

    with pytest.raises(error, match=message):
        provider._CubLoadStoreRequest(plan=plan, value_type=dtype)


def test_request_renders_finite_typed_oob_default():
    plan = _plan(
        coop.this_block(),
        dtype=Float32,
        valid_items=ArgumentBinding.static(1),
        oob_default=ArgumentBinding.static(Float32(1.25)),
    ).require_supported()
    request = provider._CubLoadStoreRequest(plan=plan, value_type=Float32)

    assert provider._cpp_oob_literal(request) == "static_cast<float>(1.25)"


def test_runtime_oob_default_requires_exact_compiler_dtype():
    assert provider._coerce_runtime_oob_default(Int32(1), Int32).__class__ is Int32
    with pytest.raises(TypeError, match="must match the memory dtype"):
        provider._coerce_runtime_oob_default(Float32(1), Int32)


def test_block_load_renderer_models_partial_offset_and_cub_collective():
    plan = _plan(
        coop.this_block(),
        kind="load",
        algorithm="striped",
        valid_items=ArgumentBinding.runtime(),
        oob_default=ArgumentBinding.static(0),
        offset=ArgumentBinding.static(4),
    ).require_supported()
    request = provider._CubLoadStoreRequest(plan=plan, value_type=Int32)
    source = "\n".join(provider._render_cub_load_store(request))

    assert "::cub::BlockLoad<int, 64, 2, ::cub::BLOCK_LOAD_STRIPED" in source
    assert "const int* base, int valid_items, int* result_items" in source
    assert "tile_ptr += 4;" in source
    assert ".Load(tile_ptr, items, valid_items, static_cast<int>(0));" in source
    assert "cuda_coop_cutlass_block_sync();" in source
    assert "result_items[1] = items[1];" in source


def test_warp_store_renderer_offsets_each_physical_warp_tile():
    plan = _plan(
        coop.this_warp(),
        kind="store",
        dtype=Float32,
        items_per_thread=3,
        algorithm="transpose",
        offset=ArgumentBinding.runtime(),
    ).require_supported()
    request = provider._CubLoadStoreRequest(plan=plan, value_type=Float32)
    source = "\n".join(provider._render_cub_load_store(request))

    assert "::cub::WarpStore<float, 3, ::cub::WARP_STORE_TRANSPOSE, 32>" in source
    assert "storage[2]" in source
    assert "int item" not in source
    assert "float item0, float item1, float item2, long long offset" in source
    assert "storage_instance) * 96ll" in source
    assert ".Store(tile_ptr, items);" in source
    assert "cuda_coop_cutlass_warp_sync();" in source


def test_memory_pointer_diagnostic_covers_cute_and_prims_without_raw_pointer():
    class TensorLike:
        element_type = Int32

    assert provider._memory_dtype(TensorLike()) is Int32
    with pytest.raises(NotImplementedError, match="raw contiguous iterator/pointer"):
        provider._memory_pointer(TensorLike(), primitive_name="load")


def test_contiguous_layout_proof_is_strict_and_import_light():
    class Layout:
        def __init__(self, shape, stride):
            self.shape = shape
            self.stride = stride

    assert contiguous_layout_reason(Layout((128,), (1,))) is None
    assert contiguous_layout_reason(Layout((8, 4, 2), (8, 2, 1))) is None
    assert "not a compact" in contiguous_layout_reason(Layout((8, 4, 2), (12, 3, 1)))
    assert contiguous_layout_reason(Layout(((8, 4), 2), ((1, 8), 32))) is None
    assert "incongruent" in contiguous_layout_reason(Layout(((8, 4), 2), (1, 8)))
    assert "not a compact" in contiguous_layout_reason(
        Layout(((8, 4), 2), ((1, 9), 36))
    )
    assert "not statically provable" in contiguous_layout_reason(
        Layout((object(), 4), (4, 1))
    )
    assert "no inspectable" in contiguous_layout_reason(object())


def test_required_static_extent_covers_block_and_every_warp_instance(monkeypatch):
    block = provider._CubLoadStoreRequest(
        _plan(
            coop.this_block(),
            items_per_thread=2,
            valid_items=ArgumentBinding.static(17),
            offset=ArgumentBinding.static(4),
        ).require_supported(),
        Int32,
    )
    warp = provider._CubLoadStoreRequest(
        _plan(
            coop.this_warp(),
            items_per_thread=2,
            valid_items=ArgumentBinding.static(17),
            offset=ArgumentBinding.static(4),
        ).require_supported(),
        Int32,
    )
    assert provider._required_static_elements(block) == 21
    assert provider._required_static_elements(warp) == 85

    pointer = object()
    monkeypatch.setattr(
        provider,
        "_contiguous_memory_proof",
        lambda *args, **kwargs: (
            provider._ContiguousMemoryProof(pointer, 84),
            "proven",
        ),
    )
    with pytest.raises(ValueError, match="requires 85 elements.*provides 84"):
        provider._memory_pointer(
            object(),
            primitive_name="load",
            required_elements=85,
        )
    assert (
        provider._memory_pointer(
            object(),
            primitive_name="store",
            required_elements=84,
        )
        is pointer
    )


@pytest.mark.parametrize(
    ("dtype", "left", "right"),
    (
        (Int32, 1, np.int32(1)),
        (Float32, 0.0, -0.0),
    ),
)
def test_request_identity_preserves_static_scalar_representation(
    dtype,
    left,
    right,
):
    def request(value):
        plan = _plan(
            coop.this_block(),
            dtype=dtype,
            valid_items=ArgumentBinding.static(1),
            oob_default=ArgumentBinding.static(value),
        ).require_supported()
        return provider._CubLoadStoreRequest(plan, dtype)

    left_request = request(left)
    right_request = request(right)
    assert left_request != right_request
    assert left_request.symbol_name != right_request.symbol_name


def test_scoped_route_classification_is_fail_closed_before_provider_registration(
    monkeypatch,
):
    class Memory:
        element_type = Int32
        shape = ((8, 4), 8)
        stride = ((1, 8), 32)

        def data_ptr(self):
            return object()

    monkeypatch.setattr(
        provider,
        "_contiguous_memory_proof",
        lambda *args, **kwargs: (object(), "proven"),
    )
    launch_kwargs = {"launch_metadata": {"block": (8, 4, 2)}}
    block = classify_scoped_load_store_route(
        Memory(),
        scope="cuda.coop.cutlass._block",
        primitive_name="load",
        launch_kwargs=launch_kwargs,
        dtype=Int32,
        items_per_thread=2,
    )
    warp = classify_scoped_load_store_route(
        Memory(),
        scope="cuda.coop.cutlass._warp",
        primitive_name="store",
        launch_kwargs=launch_kwargs,
        dtype=Int32,
        items_per_thread=2,
        threads_in_warp=32,
    )
    assert block.route is ScopedLoadStoreRoute.CANONICAL_CUB
    assert block.exact_block_dim == (8, 4, 2)
    assert warp.route is ScopedLoadStoreRoute.CANONICAL_CUB

    missing = classify_scoped_load_store_route(
        Memory(),
        scope="cuda.coop.cutlass._block",
        primitive_name="load",
        launch_kwargs={},
        dtype=Int32,
        items_per_thread=2,
    )
    logical = classify_scoped_load_store_route(
        Memory(),
        scope="cuda.coop.cutlass._warp",
        primitive_name="load",
        launch_kwargs=launch_kwargs,
        dtype=Int32,
        items_per_thread=2,
        threads_in_warp=16,
    )
    incomplete = classify_scoped_load_store_route(
        Memory(),
        scope="cuda.coop.cutlass._warp",
        primitive_name="load",
        launch_kwargs={"launch_metadata": {"block": 33}},
        dtype=Int32,
        items_per_thread=2,
        threads_in_warp=32,
    )
    assert missing.route is ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER
    assert "exact block dimensions" in missing.reason
    assert logical.route is ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER
    assert "logical warp" in logical.reason
    assert incomplete.route is ScopedLoadStoreRoute.CUTE_INDEXING_PAYLOAD_ADAPTER
    assert "complete physical warps" in incomplete.reason


def test_group_provider_rejects_noncontiguous_memory_before_registration(
    monkeypatch,
):
    class Noncontiguous:
        element_type = Int32
        shape = (8, 4)
        stride = (5, 1)

    monkeypatch.setattr(
        provider._provider_support,
        "register_request",
        lambda request: pytest.fail("request registered before pointer proof"),
    )
    with pytest.raises(NotImplementedError, match="not a compact contiguous layout"):
        provider.provider_load(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=32),
            source=Noncontiguous(),
            output=coop.ThreadData(2, dtype=Int32),
            algorithm=provider.GroupLoadStoreAlgorithm.DIRECT,
            valid_items=None,
            valid_items_binding=ArgumentBinding.omitted(),
            oob_default=None,
            oob_default_binding=ArgumentBinding.omitted(),
            offset=None,
            offset_binding=ArgumentBinding.omitted(),
        )


def test_group_provider_rejects_static_extent_before_registration(monkeypatch):
    class TooSmall:
        element_type = Int32
        shape = (20,)
        stride = (1,)

    monkeypatch.setattr(
        provider._provider_support,
        "register_request",
        lambda request: pytest.fail("request registered before extent proof"),
    )
    monkeypatch.setattr(
        provider,
        "_contiguous_memory_proof",
        lambda *args, **kwargs: (
            provider._ContiguousMemoryProof(object(), 20),
            "proven",
        ),
    )
    with pytest.raises(ValueError, match="requires 21 elements.*provides 20"):
        provider.provider_load(
            group=coop.this_block(),
            launch=LaunchFacts(exact_block_dim=32),
            source=TooSmall(),
            output=coop.ThreadData(2, dtype=Int32),
            algorithm=provider.GroupLoadStoreAlgorithm.DIRECT,
            valid_items=17,
            valid_items_binding=ArgumentBinding.static(17),
            oob_default=0,
            oob_default_binding=ArgumentBinding.static(0),
            offset=4,
            offset_binding=ArgumentBinding.static(4),
        )


def test_root_and_scoped_sources_share_load_store_artifact_identity():
    root = _plan(
        coop.this_block(),
        kind="load",
        items_per_thread=3,
        offset=ArgumentBinding.runtime(),
        source="cutlass_root",
    ).require_supported()
    scoped = _plan(
        coop.this_block(),
        kind="load",
        items_per_thread=3,
        offset=ArgumentBinding.runtime(),
        source="scoped_block",
    ).require_supported()
    root_request = provider._CubLoadStoreRequest(root, Int32)
    scoped_request = provider._CubLoadStoreRequest(scoped, Int32)

    assert root.artifact_key == scoped.artifact_key
    assert root_request.semantic_key == scoped_request.semantic_key
    assert root_request.symbol_name == scoped_request.symbol_name
