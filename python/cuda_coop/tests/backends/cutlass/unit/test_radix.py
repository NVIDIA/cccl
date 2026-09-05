# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import inspect
from types import SimpleNamespace

import numpy as np
import pytest


def _provider_dependencies() -> None:
    pytest.importorskip("cutlass")
    pytest.importorskip("cutlass.cute.ffi")


def test_radix_public_api() -> None:
    _provider_dependencies()

    import cuda.coop.cutlass as coop

    for name in ("radix_rank", "radix_sort_keys", "radix_sort_pairs"):
        assert name in coop.__all__
        function = getattr(coop, name)
        assert function.__module__ == "cuda.coop.cutlass._group_radix"
        assert all(
            not parameter.startswith("_")
            for parameter in inspect.signature(function).parameters
        )


def test_radix_requests_are_block_only_and_render_public_cub() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32, Int64

    import cuda.coop.cutlass as coop
    from cuda.coop._core import GroupOperandKind, LaunchFacts
    from cuda.coop.cutlass._lowering import _radix as provider
    from cuda.coop.cutlass._lowering._core import (
        CutlassCoreAdapter,
        CutlassRuntimeIntRange,
        render_cutlass_core_artifact,
    )

    launch = LaunchFacts(exact_block_dim=(64, 1, 1))
    sort_request = provider._make_sort_request(
        group=coop.this_block(),
        launch=launch,
        key_type=Int32,
        value_type=Int32,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        descending=True,
        source="cutlass_root",
        external_scratch=True,
    )
    sort_source = "\n".join(render_cutlass_core_artifact(sort_request))
    assert sort_request.external_scratch is True
    assert "#include <cub/block/block_radix_sort.cuh>" not in sort_source
    assert "::cub::BlockRadixSort<int, 64, 2, int, 4, true" in sort_source
    assert "cudaSharedMemBankSizeFourByte, 1, 1>" in sort_source
    assert "unsigned int temp_storage_smem_addr" in sort_source
    assert "temp_storage_bytes < required_temp_bytes" in sort_source
    assert "required_temp_alignment - 1ull" in sort_source
    assert "cuda_coop_cutlass_shared_ptr(temp_storage_smem_addr)" in sort_source
    assert "if (temp_storage_auto_sync != 0)" in sort_source
    assert "if (begin_bit < 0 || begin_bit > 31)" in sort_source
    assert "if (end_bit < 1 || end_bit > 32)" in sort_source
    assert "if (begin_bit >= end_bit)" in sort_source
    sort_call = sort_source.index(".SortDescending(")
    assert sort_source.index("if (begin_bit < 0") < sort_call
    assert sort_source.index("if (end_bit < 1") < sort_call
    assert sort_source.index("if (begin_bit >= end_bit)") < sort_call

    wide_sort_request = provider._make_sort_request(
        group=coop.this_block(),
        launch=launch,
        key_type=Int64,
        value_type=None,
        items_per_thread=1,
        operand_kind=GroupOperandKind.SCALAR,
        descending=False,
        source="cutlass_root",
    )
    wide_sort_source = "\n".join(render_cutlass_core_artifact(wide_sort_request))
    assert "if (begin_bit < 0 || begin_bit > 63)" in wide_sort_source
    assert "if (end_bit < 1 || end_bit > 64)" in wide_sort_source
    assert "if (begin_bit >= end_bit)" in wide_sort_source

    with pytest.raises(ValueError, match="distinct parameters"):
        CutlassRuntimeIntRange(
            "begin_bit",
            0,
            31,
            less_than_parameter="begin_bit",
        )
    with pytest.raises(ValueError, match="input scalar parameters"):
        CutlassCoreAdapter().materialize(
            sort_request.specialization,
            plan=sort_request.plan,
            kind=sort_request.kind,
            runtime_int_ranges=(
                CutlassRuntimeIntRange(
                    "begin_bit",
                    0,
                    31,
                    less_than_parameter="keys",
                ),
            ),
            external_scratch=True,
        )

    rank_request = provider._make_rank_request(
        group=coop.this_block(),
        launch=launch,
        input_type=Int32,
        items_per_thread=2,
        operand_kind=GroupOperandKind.ARRAY,
        begin_bit=24,
        end_bit=32,
        descending=False,
        prefix_items=4,
        source="cutlass_root",
    )
    rank_source = "\n".join(render_cutlass_core_artifact(rank_request))
    assert "::cub::BlockRadixRank<64, 8, false, true" in rank_source
    assert "cudaSharedMemBankSizeEightByte, 1, 1>" in rank_source
    assert "^ 0x80000000u" in rank_source
    assert "exclusive_digit_prefix" in rank_source
    assert "temp_storage_smem_addr" not in rank_source

    for group in (coop.this_warp(), coop.this_warp().group_by(8)):
        with pytest.raises(NotImplementedError, match="only this_block"):
            from cuda.coop.cutlass import _group_radix

            _group_radix._validate_group(group, primitive_name="radix_sort_keys")


@pytest.mark.parametrize("invalid", [True, np.bool_(True), 1.5, "3"])
def test_static_rank_controls_reject_nonintegers(invalid) -> None:
    _provider_dependencies()

    from cuda.coop.cutlass import _group_radix

    with pytest.raises(TypeError, match="trace-time static integer"):
        _group_radix._resolve_rank_bits(
            begin_bit=invalid,
            end_bit=None,
            radix_bits=None,
        )


@pytest.mark.parametrize(
    ("kwargs", "message"),
    [
        ({"begin_bit": -1}, "begin_bit must be non-negative"),
        ({"begin_bit": 4, "end_bit": 4}, "greater than begin_bit"),
        ({"begin_bit": 0, "radix_bits": 0}, "radix_bits must be positive"),
        ({"begin_bit": 0, "end_bit": 9}, "bit width must be <= 8"),
        (
            {"begin_bit": 3, "end_bit": 8, "radix_bits": 4},
            "radix_bits must match",
        ),
    ],
)
def test_static_rank_control_ranges_are_strict(kwargs, message) -> None:
    _provider_dependencies()

    from cuda.coop.cutlass import _group_radix

    controls = {"begin_bit": 0, "end_bit": None, "radix_bits": None}
    controls.update(kwargs)
    with pytest.raises(ValueError, match=message):
        _group_radix._resolve_rank_bits(**controls)


def test_sort_bit_ranges_accept_only_integer_static_or_dsl_values() -> None:
    _provider_dependencies()
    from cutlass.base_dsl.typing import Int32

    from cuda.coop.cutlass._compiler._types import validate_radix_bit_range

    runtime_begin = Int32(3)
    runtime_end = Int32(13)
    assert validate_radix_bit_range(0, None, Int32) == 32
    assert validate_radix_bit_range(np.int64(3), np.int64(13), Int32) == 13
    assert validate_radix_bit_range(runtime_begin, runtime_end, Int32) is runtime_end
    for invalid in (True, np.bool_(True), 1.5, np.float32(1.5), object()):
        with pytest.raises(TypeError, match="int-like scalars"):
            validate_radix_bit_range(invalid, None, Int32)
    for begin_bit, end_bit, message in (
        (-1, None, "non-negative"),
        (32, None, "must be < 32"),
        (0, 33, "must be <= 32"),
        (16, 16, "greater than begin_bit"),
    ):
        with pytest.raises(ValueError, match=message):
            validate_radix_bit_range(begin_bit, end_bit, Int32)


def test_fixed_radix_storage_forwards_address_capacity_alignment_and_sync(
    monkeypatch,
) -> None:
    _provider_dependencies()

    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._lowering import _radix as provider

    storage = coop.TempStorage(4096, alignment=32, auto_sync=False)
    assert (
        provider._temp_storage_for_radix_sort(
            group=coop.this_block(),
            source="cutlass_root",
            explicit_temp_storage=storage,
        )
        is storage
    )
    with pytest.raises(ValueError, match="only for block groups"):
        provider._temp_storage_for_radix_sort(
            group=coop.this_warp(),
            source="cutlass_root",
            explicit_temp_storage=storage,
        )
    with pytest.raises(ValueError, match="group-first call"):
        provider._temp_storage_for_radix_sort(
            group=coop.this_block(),
            source="private",
            explicit_temp_storage=storage,
        )
    exclusive = coop.TempStorage(
        4096,
        alignment=16,
        auto_sync=False,
        sharing="exclusive",
    )
    with pytest.raises(ValueError, match="does not support sharing='exclusive'"):
        provider._temp_storage_for_radix_sort(
            group=coop.this_block(),
            source="cutlass_root",
            explicit_temp_storage=exclusive,
        )

    address = object()
    observed: dict[str, object] = {}

    def materialize(temp_storage, **kwargs):
        observed["storage"] = temp_storage
        observed.update(kwargs)
        return SimpleNamespace(
            smem_addr_u32=address,
            size_in_bytes=4096,
            alignment=32,
            auto_sync=False,
        )

    monkeypatch.setattr(
        provider._provider_storage,
        "materialize_temp_storage_binding",
        materialize,
    )
    monkeypatch.setattr(provider, "Int32", lambda value: value)
    assert provider._external_scratch_args(
        storage,
        requirement_key=("radix-sort",),
    ) == (address, 4096, 0)
    assert observed == {
        "storage": storage,
        "scope": "cuda.coop.cutlass",
        "implicit_alignment": 16,
    }
