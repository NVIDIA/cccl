# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Host-only contract tests for the qualified CUTLASS backend."""

from __future__ import annotations

import inspect
from dataclasses import replace
from types import SimpleNamespace

import numpy as np
import pytest

coop = pytest.importorskip("cuda.coop.cutlass", exc_type=ImportError)
cutlass_types = pytest.importorskip("cutlass.base_dsl.typing")

from cuda import coop as root_coop  # noqa: E402
from cuda.coop._core import (  # noqa: E402
    ArgumentBinding,
    GroupLoadStoreKind,
    GroupLoadStoreSemantics,
    LaunchFactOrigin,
    LaunchFacts,
    make_group_primitive_call,
    plan_group_primitive,
)
from cuda.coop.cutlass import _provider  # noqa: E402
from cuda.coop.cutlass._compiler import (  # noqa: E402
    register_trace_context,
    trace_context,
)
from cuda.coop.cutlass._load_store import (  # noqa: E402
    _integer_binding,
    _oob_binding,
)
from cuda.coop.cutlass._runtime import (  # noqa: E402
    _missing_capabilities,
    validate_cutlass_runtime,
)


def _load_plan(
    *,
    dtype: type | None = None,
    oob_default: ArgumentBinding | None = None,
):
    if dtype is None:
        dtype = cutlass_types.Int32
    if oob_default is None:
        oob_default = ArgumentBinding.static(-1)
    operation = GroupLoadStoreSemantics(
        kind=GroupLoadStoreKind.LOAD,
        dtype=dtype,
        items_per_thread=2,
        valid_items=ArgumentBinding.static(61),
        oob_default=oob_default,
        offset=ArgumentBinding.static(3),
    )
    launch = LaunchFacts(
        exact_block_dim=(32, 1, 1),
        provenance=LaunchFactOrigin(
            fact="exact_block_dim",
            source="test_compiler",
            verified=True,
        ),
    )
    return plan_group_primitive(
        make_group_primitive_call(coop.this_block(), operation),
        launch,
    ).require_supported()


def _binding(kind: str, value: object) -> ArgumentBinding:
    if kind == "omitted":
        return ArgumentBinding.omitted()
    if kind == "static":
        return ArgumentBinding.static(value)
    return ArgumentBinding.runtime()


_CONTROL_CASES = [
    (operation, valid, oob, offset)
    for operation in ("load", "store")
    for valid in ("omitted", "static", "runtime")
    for oob in (
        ("omitted", "static", "runtime")
        if operation == "load" and valid != "omitted"
        else ("omitted",)
    )
    for offset in ("omitted", "static", "runtime")
]


def test_public_surface_signatures_and_summaries_are_locked() -> None:
    assert coop.__all__ == ["ThreadData", "ThreadGroup", "this_block", "load", "store"]
    expected_summaries = {
        "ThreadData": "Create an uninitialized per-thread register payload.",
        "ThreadGroup": "Descriptor for the current CUDA thread block.",
        "this_block": "Return a descriptor for the current CUDA thread block.",
        "load": "Collectively load one block tile into a per-thread payload.",
        "store": "Collectively store one per-thread payload as one block tile.",
    }
    for name, expected in expected_summaries.items():
        assert inspect.getdoc(getattr(coop, name)).splitlines()[0] == expected
        assert inspect.getdoc(getattr(coop, name)) == inspect.getdoc(
            getattr(root_coop, name)
        )

    assert tuple(inspect.signature(coop.load).parameters) == (
        "group",
        "source",
        "items",
        "valid_items",
        "oob_default",
        "offset",
    )
    assert tuple(inspect.signature(coop.store).parameters) == (
        "group",
        "destination",
        "items",
        "valid_items",
        "offset",
    )


@pytest.mark.parametrize(
    ("dtype", "expected"),
    [
        (np.uint8, cutlass_types.Uint8),
        (np.int32, cutlass_types.Int32),
        (np.uint32, cutlass_types.Uint32),
        (np.int64, cutlass_types.Int64),
        (np.uint64, cutlass_types.Uint64),
        (np.float32, cutlass_types.Float32),
        (np.float64, cutlass_types.Float64),
    ],
)
def test_thread_data_accepts_exactly_the_portable_numeric_profile(
    dtype: type,
    expected: type,
) -> None:
    assert coop.ThreadData(2, dtype=dtype).dtype is expected


def test_thread_data_rejects_invalid_sizes_dtypes_and_indices() -> None:
    for value in (0, -1, True, 1.5):
        with pytest.raises((TypeError, ValueError)):
            coop.ThreadData(value)
    with pytest.raises(TypeError):
        coop.ThreadData(2, dtype=np.complex64)

    items = coop.ThreadData(2, dtype=np.int32)
    for index in (-1, 2):
        with pytest.raises(IndexError):
            items[index]
    for index in (True, 1.5):
        with pytest.raises(TypeError):
            items[index]
    with pytest.raises(RuntimeError, match="before it was initialized"):
        items[0]
    items[0] = 7
    assert items[0] == 7
    with pytest.raises(AttributeError):
        items.items_per_thread = 3


@pytest.mark.parametrize(
    "value",
    (
        True,
        np.bool_(True),
        cutlass_types.Boolean(True),
        1.5,
        np.float32(1.5),
        "1",
    ),
)
def test_integer_controls_reject_non_integer_scalars(value: object) -> None:
    with pytest.raises(TypeError, match="valid_items must be an integer"):
        _integer_binding(value, name="valid_items")


def test_integer_controls_accept_cutlass_runtime_integers() -> None:
    binding = _integer_binding(cutlass_types.Int32(7), name="valid_items")
    assert binding.kind.value == "runtime"


@pytest.mark.parametrize(
    "value",
    (True, np.bool_(True), cutlass_types.Boolean(True)),
)
def test_oob_bool_rejected(value: object) -> None:
    with pytest.raises(TypeError, match="oob_default must be numeric, not boolean"):
        _oob_binding(value)


@pytest.mark.parametrize("value", ("not numeric", [1], object()))
def test_oob_rejects_non_numeric_runtime_values(value: object) -> None:
    with pytest.raises(TypeError, match="oob_default must be a numeric scalar"):
        _oob_binding(value)


def test_oob_accepts_cutlass_runtime_numeric_values() -> None:
    assert _oob_binding(cutlass_types.Int32(7)).kind.value == "runtime"
    assert _oob_binding(cutlass_types.Float32(1.5)).kind.value == "runtime"
    assert _oob_binding(1 << 100_000).kind.value == "static"


@pytest.mark.parametrize(
    ("dtype", "values"),
    [
        (cutlass_types.Uint8, (7, 250)),
        (cutlass_types.Int32, (7, -3)),
        (cutlass_types.Uint32, (7, 4_000_000_000)),
        (cutlass_types.Int64, (7, -(1 << 40))),
        (cutlass_types.Uint64, (7, 1 << 63)),
        (cutlass_types.Float32, (0.5, -3.25)),
        (cutlass_types.Float64, (0.5, -3.25)),
    ],
)
def test_store_converts_plain_literals_to_declared_dtype(
    dtype: type,
    values: tuple[int | float, int | float],
) -> None:
    items = coop.ThreadData(2, dtype=dtype)
    items[0], items[1] = values

    converted = _provider._thread_data_values(items, dtype)

    assert all(isinstance(item, dtype) for item in converted)


@pytest.mark.parametrize(
    ("dtype", "value", "message"),
    [
        (cutlass_types.Uint8, 256, "not representable"),
        (cutlass_types.Uint8, -1, "not representable"),
        (cutlass_types.Int32, 1.5, "dtype does not match"),
        (cutlass_types.Float32, 1e39, "not representable"),
    ],
)
def test_store_rejects_incompatible_plain_literals(
    dtype: type,
    value: int | float,
    message: str,
) -> None:
    items = coop.ThreadData(1, dtype=dtype)
    items[0] = value

    with pytest.raises((TypeError, ValueError), match=message):
        _provider._thread_data_values(items, dtype)


def test_store_preserves_nonfinite_float_payloads() -> None:
    items = coop.ThreadData(2, dtype=cutlass_types.Float32)
    items[0] = float("inf")
    items[1] = float("nan")

    converted = _provider._thread_data_values(items, cutlass_types.Float32)

    assert all(isinstance(item, cutlass_types.Float32) for item in converted)


def test_store_requires_exact_explicit_item_dtypes() -> None:
    items = coop.ThreadData(2, dtype=cutlass_types.Int32)
    items[0] = cutlass_types.Int32(1)
    items[1] = np.int32(2)
    assert _provider._thread_data_values(items, cutlass_types.Int32) == (
        items[0],
        items[1],
    )

    items[1] = cutlass_types.Float32(2.0)
    with pytest.raises(TypeError, match="dtype does not match initialized item types"):
        _provider._thread_data_values(items, cutlass_types.Int32)


@pytest.mark.parametrize(
    ("dtype", "value", "error", "message"),
    [
        (cutlass_types.Uint8, 300, ValueError, "not representable"),
        pytest.param(
            cutlass_types.Uint8,
            1 << 100_000,
            ValueError,
            "not representable",
            id="uint8-huge-int",
        ),
        (cutlass_types.Int32, -1.5, TypeError, "dtype does not match"),
        (cutlass_types.Float32, 1e39, ValueError, "not representable"),
        (cutlass_types.Float32, float("inf"), ValueError, "must be finite"),
        (cutlass_types.Int64, np.int32(1), TypeError, "tensor dtype"),
    ],
)
def test_static_oob_default_must_match_memory_dtype(
    dtype: type,
    value: object,
    error: type[Exception],
    message: str,
) -> None:
    plan = _load_plan(
        dtype=dtype,
        oob_default=ArgumentBinding.static(value),
    )

    with pytest.raises(error, match=message):
        _provider._LoadStoreRequest.from_plan(plan, value_type=dtype)


@pytest.mark.parametrize(
    ("dtype", "value"),
    [
        (cutlass_types.Uint8, np.uint8(3)),
        (cutlass_types.Int64, 3),
        (cutlass_types.Float64, 0.1),
    ],
)
def test_static_oob_default_accepts_compatible_values(
    dtype: type,
    value: object,
) -> None:
    plan = _load_plan(
        dtype=dtype,
        oob_default=ArgumentBinding.static(value),
    )

    request = _provider._LoadStoreRequest.from_plan(plan, value_type=dtype)

    assert request.oob_default_value is value


def test_runtime_oob_default_requires_exact_compiler_dtype() -> None:
    plan = _load_plan(oob_default=ArgumentBinding.runtime())
    request = _provider._LoadStoreRequest.from_plan(
        plan,
        value_type=cutlass_types.Int32,
    )

    with pytest.raises(TypeError, match="oob_default must match the tensor dtype"):
        _provider._runtime_arguments(
            request,
            valid_items=None,
            oob_default=cutlass_types.Float32(1.0),
            offset=None,
        )

    parameter_types, arguments = _provider._runtime_arguments(
        request,
        valid_items=None,
        oob_default=cutlass_types.Int32(1),
        offset=None,
    )
    assert parameter_types == [cutlass_types.Int32]
    assert isinstance(arguments[0], cutlass_types.Int32)


def test_store_signless_ints() -> None:
    item = SimpleNamespace(type="i32", signed=None)
    items = coop.ThreadData(1, dtype=cutlass_types.Uint32)
    items[0] = item

    assert _provider._thread_data_values(items, cutlass_types.Uint32) == (item,)


def test_renderer_is_direct_internal_scratch_and_semantically_deduplicated() -> None:
    request = _provider._LoadStoreRequest.from_plan(
        _load_plan(),
        value_type=cutlass_types.Int32,
    )
    source = _provider._render_bundle_source({request, request})
    assert source.count(f"void {request.symbol_name}(") == 1
    assert "::cub::BlockLoad<int, 32, 2, ::cub::BLOCK_LOAD_DIRECT, 1, 1>" in source
    assert "__shared__ typename block_type::TempStorage temp_storage;" in source
    assert "tile += 3;" in source
    assert ".Load(tile, items, 61, static_cast<int>(-1));" in source
    assert "__syncthreads();" in source


@pytest.mark.parametrize(
    ("left", "right"),
    (
        (1, np.int32(1)),
        (0.0, -0.0),
        (True, 1),
    ),
)
def test_request_identity_preserves_static_scalar_representation(
    left: object,
    right: object,
) -> None:
    request = _provider._LoadStoreRequest.from_plan(
        _load_plan(),
        value_type=cutlass_types.Int32,
    )
    left_request = replace(request, oob_default_value=left)
    right_request = replace(request, oob_default_value=right)

    assert left_request != right_request
    assert left_request.symbol_name != right_request.symbol_name
    source = _provider._render_bundle_source({left_request, right_request})
    assert source.count("void cuda_coop_cutlass_cub_load_block_") == 2


@pytest.mark.parametrize(
    ("operation", "valid_kind", "oob_kind", "offset_kind"),
    _CONTROL_CASES,
)
def test_renderer_covers_every_control_binding_combination(
    operation: str,
    valid_kind: str,
    oob_kind: str,
    offset_kind: str,
) -> None:
    semantics = GroupLoadStoreSemantics(
        kind=operation,
        dtype=cutlass_types.Int32,
        items_per_thread=2,
        valid_items=_binding(valid_kind, 61),
        oob_default=_binding(oob_kind, -1),
        offset=_binding(offset_kind, 3),
    )
    launch = LaunchFacts(
        exact_block_dim=(32, 1, 1),
        provenance=LaunchFactOrigin(
            fact="exact_block_dim",
            source="test_compiler",
            verified=True,
        ),
    )
    plan = plan_group_primitive(
        make_group_primitive_call(coop.this_block(), semantics),
        launch,
    ).require_supported()
    request = _provider._LoadStoreRequest.from_plan(
        plan,
        value_type=cutlass_types.Int32,
    )
    source = _provider._render_bundle_source({request})
    signature = source.split(f"void {request.symbol_name}(", 1)[1].split(")", 1)[0]

    assert ("int valid_items" in signature) == (valid_kind == "runtime")
    assert ("int oob_default" in signature) == (oob_kind == "runtime")
    assert ("long long offset" in signature) == (offset_kind == "runtime")
    assert "TempStorage" not in signature
    assert "__shared__ typename block_type::TempStorage temp_storage;" in source


def test_per_trace_registration_deduplicates_identical_requests(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    request = _provider._LoadStoreRequest.from_plan(
        _load_plan(),
        value_type=cutlass_types.Int32,
    )

    class CompileOptions:
        pass

    compile_options = CompileOptions()
    dsl = SimpleNamespace(compile_options=compile_options)
    module = object()
    monkeypatch.setattr(_provider, "_ensure_trace_hook", lambda: dsl)
    monkeypatch.setattr(_provider, "_active_module_op", lambda: module)
    try:
        _provider._register_request(request)
        _provider._register_request(request)
        assert len(_provider._SESSIONS[compile_options].requests) == 1
    finally:
        _provider._SESSIONS.pop(compile_options, None)


@pytest.mark.parametrize(
    ("shape", "stride", "reason"),
    [
        ((4, 8), (8, 1), None),
        ((4, 8), (1, 4), None),
        (((2, 4), 8), ((1, 2), 8), None),
        (((2, 4), 8), ((1, 3), 8), "layout is not compact"),
        (
            ((2, 4), 8),
            (1, 2),
            "shape and stride layouts are not congruent",
        ),
        ((4, 8), (9, 1), "layout is not compact"),
        ((4, 8), (None, 1), "layout is not statically known"),
    ],
)
def test_compact_layout_validation_accepts_any_static_dense_order(
    shape: tuple[object, ...],
    stride: tuple[object, ...],
    reason: str | None,
) -> None:
    tensor = SimpleNamespace(shape=shape, stride=stride)
    assert _provider._compact_layout_reason(tensor) == reason


def test_pointer_validation_reports_non_compact_and_missing_pointer() -> None:
    non_compact = SimpleNamespace(shape=(4, 8), stride=(9, 1))
    with pytest.raises(NotImplementedError, match="statically compact"):
        _provider._require_contiguous_pointer(non_compact, feature="load")

    pointerless = SimpleNamespace(shape=(4, 8), stride=(8, 1))
    with pytest.raises(TypeError, match="raw LLVM pointer"):
        _provider._require_contiguous_pointer(pointerless, feature="store")


def test_static_operand_extent_covers_offset_and_valid_prefix() -> None:
    tensor = SimpleNamespace(shape=(64,), stride=(1,))

    assert _provider._required_static_elements(_load_plan()) == 64
    with pytest.raises(ValueError, match="requires 65 elements.*provides 64"):
        _provider._require_contiguous_pointer(
            tensor,
            feature="load",
            required_elements=65,
        )


def test_provider_rejects_a_plan_without_the_cub_block_target() -> None:
    plan = SimpleNamespace(target=SimpleNamespace(value=None))
    with pytest.raises(ValueError, match="CUB block lowering plan"):
        _provider._LoadStoreRequest.from_plan(
            plan,
            value_type=cutlass_types.Int32,
        )


def test_provider_rejects_a_plan_with_an_incompatible_dtype() -> None:
    with pytest.raises(TypeError, match="plan dtype does not match"):
        _provider._LoadStoreRequest.from_plan(
            _load_plan(),
            value_type=cutlass_types.Float32,
        )


def test_runtime_and_compiler_trace_context_use_validated_capabilities() -> None:
    runtime = validate_cutlass_runtime()

    dsl = runtime.dsl_type._get_dsl()
    before = tuple(dsl._trace_context_factories)
    register_trace_context()
    register_trace_context()
    after = tuple(dsl._trace_context_factories)
    assert len(after) == len(before)
    assert getattr(dsl, "_cuda_coop_cutlass_trace_context_target") is trace_context


def test_runtime_capabilities_match_the_provider_link_contract() -> None:
    runtime = validate_cutlass_runtime()
    assert not _missing_capabilities(
        runtime.cutlass_dsl,
        runtime.cute,
        runtime.compiler,
    )

    without_arch = SimpleNamespace(LinkLibraries=runtime.compiler.LinkLibraries)
    assert "cutlass.base_dsl.compiler.GPUArch" in _missing_capabilities(
        runtime.cutlass_dsl,
        runtime.cute,
        without_arch,
    )

    class RenamedLinkLibraries:
        _option_name = "renamed-link-libraries"

    renamed_link_attribute = SimpleNamespace(
        GPUArch=runtime.compiler.GPUArch,
        LinkLibraries=RenamedLinkLibraries,
    )
    assert (
        "cutlass.base_dsl.compiler.LinkLibraries._option_name=link-libraries"
        in _missing_capabilities(
            runtime.cutlass_dsl,
            runtime.cute,
            renamed_link_attribute,
        )
    )
