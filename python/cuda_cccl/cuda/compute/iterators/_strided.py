# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import operator
import struct
from textwrap import dedent

from .._bindings import Op, OpKind
from .._cpp_compile import compile_cpp_op_code
from ._base import IteratorBase, compose_state_blobs
from ._common import CUDA_PREAMBLE, ensure_iterator


class StridedIterator(IteratorBase):
    __slots__ = ["_underlying", "_stride", "_stride_offset"]

    def __init__(self, underlying, stride: int):
        self._underlying = ensure_iterator(underlying)
        self._stride = operator.index(stride)
        state_bytes, state_alignment, offsets = compose_state_blobs(
            [
                (bytes(self._underlying.state), self._underlying.state_alignment),
                (struct.pack("=q", self._stride), 8),
            ]
        )
        self._stride_offset = offsets[1]
        super().__init__(state_bytes, state_alignment, self._underlying.value_type)

    def _make_advance_op(self) -> Op:
        child_op = self._underlying.get_advance_op()
        symbol = self._make_advance_symbol()
        source = dedent(f"""
            {CUDA_PREAMBLE}

            extern "C" __device__ void {child_op.name}(void* state, void* offset);

            extern "C" __device__ void {symbol}(void* state, void* offset) {{
                const auto stride = *reinterpret_cast<uint64_t*>(static_cast<char*>(state) + {self._stride_offset});
                uint64_t distance = *static_cast<uint64_t*>(offset) * stride;
                {child_op.name}(state, &distance);
            }}
        """).strip()
        return Op(
            operator_type=OpKind.STATELESS,
            name=symbol,
            ltoir=compile_cpp_op_code(source),
            extra_ltoirs=[child_op.code, *child_op.extra_code],
        )

    def get_input_deref_op(self) -> Op | None:
        return self._underlying.get_input_deref_op()

    def get_output_deref_op(self) -> Op | None:
        return self._underlying.get_output_deref_op()

    @property
    def children(self):
        return (self._underlying,)

    @property
    def kind(self):
        return ("StridedIterator", self._underlying.kind)

    def __add__(self, offset: int):
        return StridedIterator(self._underlying + offset * self._stride, self._stride)
