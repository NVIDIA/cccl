# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import pytest


def test_numba_mlir_block_exchange_adapter_preserves_both_overloads():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)
    from numba_cuda_mlir import types

    from cuda.coop._core.block import make_block_exchange_spec
    from cuda.coop.numba_mlir._core_adapter import NumbaMlirCoreAdapter

    core_spec = make_block_exchange_spec(
        dtype=types.int32,
        block_dim=(16, 2, 1),
        items_per_thread=3,
        mode="scatter_to_striped_flagged",
        value_form="both",
        warp_time_slicing=True,
        rank_dtype=types.int32,
        valid_flag_dtype=types.uint8,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)

    assert specialization.c_name.startswith("cuda_coop_numba_mlir_block_exchange__")
    assert specialization.method_name == "ScatterToStripedFlagged"
    assert [
        [type(parameter).__name__ for parameter in method]
        for method in specialization.parameters
    ] == [
        ["Pointer", "Array", "Array", "Array"],
        ["Pointer", "Array", "Array", "Array", "Array"],
    ]
    assert not specialization.parameters[0][1].is_output
    assert not specialization.parameters[1][2].is_output

    same_shape_modes = [
        NumbaMlirCoreAdapter().materialize(
            make_block_exchange_spec(
                dtype=types.int32,
                block_dim=(32, 1, 1),
                items_per_thread=2,
                mode=mode,
                value_form="in_place",
            ).specialization
        )
        for mode in ("striped_to_blocked", "blocked_to_striped")
    ]
    assert same_shape_modes[0].c_name == same_shape_modes[1].c_name
    assert same_shape_modes[0].mangled_name(
        same_shape_modes[0].parameters[0]
    ) != same_shape_modes[1].mangled_name(same_shape_modes[1].parameters[0])


@pytest.mark.parametrize(
    ("runtime_arg_count", "expected_use_output_items"),
    [(1, False), (2, True)],
)
def test_single_phase_block_exchange_selects_one_value_form(
    runtime_arg_count,
    expected_use_output_items,
):
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    factory_kwargs = {}
    seen_factory_kwargs = set()
    rewrite._finalize_exchange_factory_kwargs(
        runtime_args=[object()] * runtime_arg_count,
        runtime_arg_count=runtime_arg_count,
        seen_factory_kwargs=seen_factory_kwargs,
        factory_kwargs=factory_kwargs,
    )

    assert factory_kwargs["use_output_items"] is expected_use_output_items
    assert "use_output_items" in seen_factory_kwargs


def test_single_phase_block_exchange_rejects_conflicting_value_form():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    with pytest.raises(
        rewrites.CoopSinglePhaseRewriteError,
        match="use_output_items does not match the runtime argument form",
    ):
        rewrite._finalize_exchange_factory_kwargs(
            runtime_args=[object(), object()],
            runtime_arg_count=2,
            seen_factory_kwargs={"use_output_items"},
            factory_kwargs={"use_output_items": False},
        )


def test_single_phase_block_exchange_resolves_explicit_auto_value_form():
    pytest.importorskip("numba_cuda_mlir", exc_type=ImportError)

    from cuda.coop.numba_mlir import _single_phase_rewrites as rewrites

    rewrite = object.__new__(rewrites.CoopSinglePhaseRewrite)
    factory_kwargs = {"use_output_items": None}
    rewrite._finalize_exchange_factory_kwargs(
        runtime_args=[object(), object()],
        runtime_arg_count=2,
        seen_factory_kwargs={"use_output_items"},
        factory_kwargs=factory_kwargs,
    )

    assert factory_kwargs["use_output_items"] is True
