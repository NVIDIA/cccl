# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

from typing import TYPE_CHECKING, Tuple, Union

from .. import _require_runtime

_require_runtime()

from cuda.coop._core.block import (
    make_block_radix_rank_spec,
    resolve_static_radix_end_bit,
)

from .._common import (
    _validate_common_integer_key_dtype,
    dim3,
    normalize_dim_param,
    normalize_dtype_param,
    resolve_threads_per_block_alias,
)
from .._core_adapter import (
    NumbaMlirArrayInputTransform,
    NumbaMlirCoreAdapter,
)
from .._types import make_invocable_from_specialization

if TYPE_CHECKING:
    import numpy as np
    from numba_cuda_mlir import types


_SIGNED_CUB_KEY_DTYPES = {
    "int32": "uint32",
    "int64": "uint64",
}
_SIGNED_INPUT_TRANSFORMS = {
    "int32": "(static_cast<unsigned int>({value}) ^ 0x80000000u)",
    "int64": ("(static_cast<unsigned long long>({value}) ^ 0x8000000000000000ull)"),
}


def radix_rank(
    dtype: Union[str, type, "np.dtype", "types.Type"],
    threads_per_block: Union[int, Tuple[int, int], Tuple[int, int, int], dim3] = None,
    items_per_thread: int = 1,
    begin_bit: int = 0,
    end_bit: int = None,
    descending: bool = False,
    exclusive_digit_prefix=None,
    dim=None,
):
    """Build a block-wide radix-rank invocable.

    The invocable wraps CUB ``BlockRadixRank`` and writes each key's rank within
    the block tile. ``begin_bit`` and ``end_bit`` select the radix window, and
    ``descending`` reverses digit ordering for descending sort pipelines.
    ``exclusive_digit_prefix`` may be provided to select the overload that also
    writes each per-thread digit-bin prefix.
    """
    threads_per_block = resolve_threads_per_block_alias(threads_per_block, dim)
    dim = normalize_dim_param(threads_per_block)
    dtype = normalize_dtype_param(dtype)

    bitwidth = getattr(dtype, "bitwidth", None)
    if bitwidth is not None:
        bitwidth = int(bitwidth)
    end_bit = resolve_static_radix_end_bit(
        begin_bit=begin_bit,
        end_bit=end_bit,
        bit_width=bitwidth,
        default_radix_bits=4,
        clamp_default=False,
    )
    dtype_name = str(dtype)
    cub_dtype = dtype
    input_transforms = None
    if dtype_name in _SIGNED_CUB_KEY_DTYPES:
        from numba_cuda_mlir import types as numba_mlir_types

        cub_dtype = getattr(numba_mlir_types, _SIGNED_CUB_KEY_DTYPES[dtype_name])
        input_transforms = {
            "keys": NumbaMlirArrayInputTransform(
                source_dtype=dtype,
                cpp_expression=_SIGNED_INPUT_TRANSFORMS[dtype_name],
            )
        }

    core_spec = make_block_radix_rank_spec(
        key_dtype=cub_dtype,
        block_dim=tuple(dim),
        items_per_thread=items_per_thread,
        begin_bit=begin_bit,
        end_bit=end_bit,
        key_bit_width=bitwidth,
        descending=descending,
        with_exclusive_digit_prefix=exclusive_digit_prefix is not None,
    )
    specialization = NumbaMlirCoreAdapter(
        input_transforms=input_transforms
    ).materialize(core_spec.specialization)
    return make_invocable_from_specialization(specialization)


def _common_radix_rank(**kwargs):
    """Materialize one portable rank specialization after dtype inference."""

    kwargs = dict(kwargs)
    kwargs["dtype"] = _validate_common_integer_key_dtype(
        kwargs["dtype"], operation="radix_rank"
    )
    return radix_rank(**kwargs)
