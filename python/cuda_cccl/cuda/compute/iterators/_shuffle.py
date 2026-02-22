# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""ShuffleIterator implementation."""

from __future__ import annotations

import struct
from textwrap import dedent

import numpy as np

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_to_ltoir
from ..types import from_numpy_dtype
from ._base import IteratorBase
from ._common import CUDA_PREAMBLE

_SHUFFLE_STATE_STRUCT = """\
struct ShuffleState {
    int64_t current_index;
    uint64_t num_items;
    uint64_t seed;
};"""


class ShuffleIterator(IteratorBase):
    """
    Iterator that produces a deterministic random permutation of indices.

    At position ``i``, yields ``bijection(i)`` where the bijection is a random
    permutation of ``[0, num_items)`` parameterized by ``seed``.

    Example:
        The code snippet below demonstrates the usage of a ``ShuffleIterator``
        to randomly permute indices:

        .. literalinclude:: ../../python/cuda_cccl/tests/compute/examples/iterator/shuffle_iterator_basic.py
            :language: python
            :start-after: # example-begin

    Args:
        num_items: Number of elements in the domain to permute. Must be > 0.
        seed: Seed for the random permutation. Different seeds produce
            different (deterministic) permutations. Defaults to 0.
    """

    __slots__ = ["_num_items", "_seed", "_current_index"]

    def __init__(self, num_items: int, seed: int = 0, *, _current_index: int = 0):
        if num_items <= 0:
            raise ValueError("num_items must be > 0")

        self._num_items = int(num_items)
        self._seed = int(seed)
        self._current_index = int(_current_index)

        # State layout matches C++ ShuffleState:
        #   int64_t  current_index  (offset 0,  size 8)
        #   uint64_t num_items      (offset 8,  size 8)
        #   uint64_t seed           (offset 16, size 8)
        state_bytes = struct.pack(
            "<qQQ", self._current_index, self._num_items, self._seed
        )

        super().__init__(
            state_bytes=state_bytes,
            state_alignment=8,
            value_type=from_numpy_dtype(np.dtype("int64")),
        )

    def _make_advance_op(self) -> Op:
        symbol = self._make_advance_symbol()

        source = dedent(f"""
            {CUDA_PREAMBLE}

            {_SHUFFLE_STATE_STRUCT}

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                auto s = static_cast<ShuffleState*>(state);
                auto dist = *static_cast<int64_t*>(offset);
                s->current_index += dist;
            }}
        """)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=compile_cpp_to_ltoir(source),
            extra_ltoirs=[],
        )

    def _make_input_deref_op(self) -> Op | None:
        symbol = self._make_input_deref_symbol()

        # Note: a potential optimization is to avoid constructing
        # `cuda::random_bijection` objects upon every dereference,
        # instead constructing it once and using it as the state
        # object. The tradeoff is that it would require a C++
        # extension providing a constructor for
        # `cuda::random_bijection` objects, since we would now be
        # doing it on the host.  See discussion in #7721.
        source = dedent(f"""
            #include <cuda/__random/random_bijection.h>
            #include <cuda/__random/pcg_engine.h>
            {CUDA_PREAMBLE}

            {_SHUFFLE_STATE_STRUCT}

            // __noinline__ is required to prevent the compiler from merging
            // this function's register usage into the calling kernel during LTO
            // inlining. feistel_bijection constructs 24 round keys with
            // UNROLL_FULL, which exhausts the kernel's register budget and
            // causes spilling to local memory (LDL/STL instructions).
            // Keeping it non-inlined gives it an isolated register frame.
            __device__ __noinline__ int64_t __shuffle_apply(uint64_t num_items, uint64_t seed, uint64_t idx) {{
                cuda::pcg64 rng(seed);
                cuda::random_bijection<uint64_t> bijection(num_items, rng);
                return static_cast<int64_t>(bijection(idx));
            }}

            extern "C" __device__ void {symbol}(void* state, void* result) {{
                const auto* s = static_cast<const ShuffleState*>(state);
                *static_cast<int64_t*>(result) = __shuffle_apply(
                    s->num_items, s->seed, static_cast<uint64_t>(s->current_index));
            }}
        """)

        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=compile_cpp_to_ltoir(source),
            extra_ltoirs=[],
        )

    def _make_output_deref_op(self) -> Op | None:
        return None

    def __add__(self, offset: int) -> "ShuffleIterator":
        return ShuffleIterator(
            self._num_items,
            self._seed,
            _current_index=self._current_index + offset,
        )
