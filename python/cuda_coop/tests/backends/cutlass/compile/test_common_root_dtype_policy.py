# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import pytest

import cuda.coop.cutlass as cutlass_coop
from cuda import coop

cute = pytest.importorskip("cutlass.cute")
runtime = pytest.importorskip("cutlass.cute.runtime")
Int32 = pytest.importorskip("cutlass.base_dsl.typing").Int32


@cute.kernel
def _nonportable_explicit_complex_thread_data_kernel(output: cute.Tensor):
    coop.ThreadData(1, dtype=complex)
    output[0] = Int32(0)


@cute.kernel
def _nonportable_inferred_complex_thread_data_kernel(output: cute.Tensor):
    items = coop.ThreadData(1)
    items[0] = 1 + 2j
    output[0] = Int32(0)


@cute.kernel
def _nonportable_inferred_bool_thread_data_kernel(output: cute.Tensor):
    items = coop.ThreadData(1)
    items[0] = True
    output[0] = Int32(0)


class _CustomValue:
    pass


@cute.kernel
def _qualified_generic_thread_data_kernel(output: cute.Tensor):
    complex_items = cutlass_coop.ThreadData(1, dtype=complex)
    complex_items[0] = 1 + 2j
    bool_items = cutlass_coop.ThreadData(1, dtype=bool)
    bool_items[0] = True
    custom_value = _CustomValue()
    custom_items = cutlass_coop.ThreadData(1, dtype=_CustomValue)
    custom_items[0] = custom_value
    if complex_items[0] == 1 + 2j and bool_items[0] and custom_items[0] is custom_value:
        output[0] = Int32(0)


@cute.jit
def _run_nonportable_explicit_complex_thread_data(output: cute.Tensor):
    _nonportable_explicit_complex_thread_data_kernel(output).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


@cute.jit
def _run_nonportable_inferred_complex_thread_data(output: cute.Tensor):
    _nonportable_inferred_complex_thread_data_kernel(output).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


@cute.jit
def _run_nonportable_inferred_bool_thread_data(output: cute.Tensor):
    _nonportable_inferred_bool_thread_data_kernel(output).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


@cute.jit
def _run_qualified_cleanup_after_explicit_complex(output: cute.Tensor):
    _qualified_generic_thread_data_kernel(output).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


@cute.jit
def _run_qualified_cleanup_after_inferred_complex(output: cute.Tensor):
    _qualified_generic_thread_data_kernel(output).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


@cute.jit
def _run_qualified_cleanup_after_inferred_bool(output: cute.Tensor):
    _qualified_generic_thread_data_kernel(output).launch(
        grid=(1, 1, 1),
        block=(1, 1, 1),
    )


def _assert_rejected_then_qualified_compile_succeeds(
    rejected,
    cleanup,
) -> None:
    fake_output = runtime.make_fake_compact_tensor(Int32, (1,))

    with pytest.raises(
        TypeError,
        match=r"cuda\.coop\.ThreadData common V1 supports dtypes",
    ):
        cute.compile(rejected, fake_output)

    assert cute.compile(cleanup, fake_output) is not None


def test_common_thread_data_rejects_explicit_complex_at_compile() -> None:
    _assert_rejected_then_qualified_compile_succeeds(
        _run_nonportable_explicit_complex_thread_data,
        _run_qualified_cleanup_after_explicit_complex,
    )


def test_common_thread_data_rejects_inferred_complex_at_compile() -> None:
    _assert_rejected_then_qualified_compile_succeeds(
        _run_nonportable_inferred_complex_thread_data,
        _run_qualified_cleanup_after_inferred_complex,
    )


def test_common_thread_data_rejects_inferred_bool_at_compile() -> None:
    _assert_rejected_then_qualified_compile_succeeds(
        _run_nonportable_inferred_bool_thread_data,
        _run_qualified_cleanup_after_inferred_bool,
    )
