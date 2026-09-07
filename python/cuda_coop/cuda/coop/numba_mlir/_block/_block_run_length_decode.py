# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from .. import _require_runtime

_require_runtime()

from numba_cuda_mlir import types

from cuda.coop._core.block import (
    BlockRunLengthDecodeStage,
    make_block_run_length_decode_semantics,
    make_block_run_length_decode_spec,
)

from .._common import (
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import NumbaMlirCoreAdapter
from .._types import make_invocable_from_specialization


class _RunLengthParent:
    """Compile-time placeholder rewritten by the single-phase coop pass."""

    def __init__(
        self,
        run_values,
        run_lengths,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype=None,
        temp_storage=None,
    ):
        self._run_values = run_values
        self._run_lengths = run_lengths
        self._runs_per_thread = runs_per_thread
        self._decoded_items_per_thread = decoded_items_per_thread
        self._total_decoded_size = total_decoded_size
        self._decoded_offset_dtype = decoded_offset_dtype
        self._temp_storage = temp_storage

    def decode(self, decoded_items, decoded_window_offset=None, relative_offsets=None):
        raise RuntimeError(
            "coop._block.run_length(...).decode(...) is only supported inside a "
            "@cuda.jit kernel with coop single-phase rewrite enabled."
        )


class _RunLengthFactory:
    """Two-phase BlockRunLengthDecode factory result."""

    def __init__(
        self,
        item_dtype,
        threads_per_block,
        runs_per_thread,
        decoded_items_per_thread,
        decoded_offset_dtype=None,
    ):
        item_dtype = normalize_dtype_param(item_dtype)
        threads_per_block = normalize_dim_param(threads_per_block)
        decoded_offset_dtype = (
            types.uint32
            if decoded_offset_dtype is None
            else normalize_dtype_param(decoded_offset_dtype)
        )
        self.core_semantics = make_block_run_length_decode_semantics(
            item_dtype=item_dtype,
            decoded_offset_dtype=decoded_offset_dtype,
            runs_per_thread=runs_per_thread,
            decoded_items_per_thread=decoded_items_per_thread,
        )
        self.item_dtype = item_dtype
        self.threads_per_block = threads_per_block
        self.runs_per_thread = self.core_semantics.runs_per_thread
        self.decoded_items_per_thread = self.core_semantics.decoded_items_per_thread
        self.decoded_offset_dtype = decoded_offset_dtype
        self._temp_storage_probe = None
        self.__name__ = "run_length"

    def _temp_storage_invocable(self):
        if self._temp_storage_probe is None:
            self._temp_storage_probe = self.decode(types.uint32, types.uint32)
        return self._temp_storage_probe

    @property
    def temp_storage_bytes(self):
        """Number of bytes required for this run-length decoder's TempStorage."""
        return self._temp_storage_invocable().temp_storage_bytes

    @property
    def temp_storage_alignment(self):
        """Required alignment for this run-length decoder's TempStorage."""
        return self._temp_storage_invocable().temp_storage_alignment

    def __call__(
        self,
        run_values,
        run_lengths,
        runs_per_thread=None,
        decoded_items_per_thread=None,
        total_decoded_size=None,
        decoded_offset_dtype=None,
        temp_storage=None,
    ):
        """Create a kernel-local run-length parent placeholder.

        A parent factory may be created outside the kernel, then called inside
        the kernel with runtime arrays before invoking ``.decode(...)``. The
        Numba-CUDA-MLIR rewrite consumes the placeholder before type inference.
        """

        return _RunLengthParent(
            run_values,
            run_lengths,
            self.runs_per_thread if runs_per_thread is None else runs_per_thread,
            self.decoded_items_per_thread
            if decoded_items_per_thread is None
            else decoded_items_per_thread,
            total_decoded_size,
            decoded_offset_dtype=(
                self.decoded_offset_dtype
                if decoded_offset_dtype is None
                else decoded_offset_dtype
            ),
            temp_storage=temp_storage,
        )

    def decode(
        self,
        run_length_dtype=types.uint32,
        total_decoded_size_dtype=types.uint32,
        *,
        with_relative_offsets=False,
        with_decoded_window_offset=False,
        relative_offset_dtype=None,
    ):
        return _run_length_decode(
            self.item_dtype,
            run_length_dtype,
            self.decoded_offset_dtype,
            total_decoded_size_dtype,
            self.threads_per_block,
            self.runs_per_thread,
            self.decoded_items_per_thread,
            with_relative_offsets=with_relative_offsets,
            with_decoded_window_offset=with_decoded_window_offset,
            relative_offset_dtype=relative_offset_dtype,
        )


def run_length(
    run_values=None,
    run_lengths=None,
    runs_per_thread=1,
    decoded_items_per_thread=1,
    total_decoded_size=None,
    decoded_offset_dtype=None,
    temp_storage=None,
    threads_per_block=None,
    dim=None,
    item_dtype=None,
):
    """Create a BlockRunLengthDecode parent placeholder.

    The returned object is consumed by the Numba-CUDA-MLIR single-phase rewrite when a
    kernel calls ``.decode(...)``. It records the run-value, run-length, decoded
    item, and offset metadata needed to build the CUB decode invocable.
    ``temp_storage`` may be a ``coop.TempStorage`` placeholder to share or
    explicitly size the storage used by the decode child call.
    """
    if item_dtype is not None:
        if run_values is not None:
            raise ValueError("run_values and item_dtype cannot both be provided")
        run_values = item_dtype

    if threads_per_block is None and dim is None and run_lengths is not None:
        try:
            normalize_dtype_param(run_values)
        except (TypeError, ValueError):
            pass
        else:
            threads_per_block = run_lengths
            run_lengths = None

    if threads_per_block is not None or dim is not None:
        if temp_storage is not None:
            raise NotImplementedError(
                "Explicit temp_storage is only supported for single-phase "
                "run_length constructors."
            )
        threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
        return _RunLengthFactory(
            run_values,
            threads_per_block,
            runs_per_thread,
            decoded_items_per_thread,
            decoded_offset_dtype=decoded_offset_dtype,
        )

    return _RunLengthParent(
        run_values,
        run_lengths,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
        temp_storage=temp_storage,
    )


def _common_run_length(
    run_values,
    run_lengths,
    runs_per_thread,
    decoded_items_per_thread,
    total_decoded_size,
    decoded_offset_dtype=None,
    _static_decoded_window_offset=None,
):
    """Compiler-only parent marker for common V1 Run Length Decode."""

    return _RunLengthParent(
        run_values,
        run_lengths,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
    )


def _qualified_group_run_length(
    run_values,
    run_lengths,
    runs_per_thread,
    decoded_items_per_thread,
    total_decoded_size,
    decoded_offset_dtype=None,
    _static_decoded_window_offset=None,
):
    """Compiler-only parent marker for qualified group-first decoding."""

    return _RunLengthParent(
        run_values,
        run_lengths,
        runs_per_thread,
        decoded_items_per_thread,
        total_decoded_size,
        decoded_offset_dtype=decoded_offset_dtype,
    )


def _run_length_decode(
    item_dtype,
    run_length_dtype,
    decoded_offset_dtype,
    total_decoded_size_dtype,
    threads_per_block,
    runs_per_thread,
    decoded_items_per_thread,
    with_relative_offsets=False,
    with_decoded_window_offset=False,
    relative_offset_dtype=None,
):
    dim = normalize_dim_param(threads_per_block)
    item_dtype = normalize_dtype_param(item_dtype)
    run_length_dtype = normalize_dtype_param(run_length_dtype)
    decoded_offset_dtype = normalize_dtype_param(decoded_offset_dtype)
    total_decoded_size_dtype = normalize_dtype_param(total_decoded_size_dtype)
    if relative_offset_dtype is not None:
        relative_offset_dtype = normalize_dtype_param(relative_offset_dtype)

    core_spec = make_block_run_length_decode_spec(
        item_dtype=item_dtype,
        run_length_dtype=run_length_dtype,
        decoded_offset_dtype=decoded_offset_dtype,
        total_decoded_size_dtype=total_decoded_size_dtype,
        block_dim=tuple(dim),
        runs_per_thread=runs_per_thread,
        decoded_items_per_thread=decoded_items_per_thread,
        stage=BlockRunLengthDecodeStage.FUSED,
        with_relative_offsets=with_relative_offsets,
        relative_offset_dtype=relative_offset_dtype,
        with_decoded_window_offset=with_decoded_window_offset,
        returns_total_decoded_size=True,
    )
    specialization = NumbaMlirCoreAdapter().materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)
