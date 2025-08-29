# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A toy example to illustrate how we can compose logical operations

import numba
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from cuda.cccl.experimental.stf._stf_bindings import (
    context,
    data_place,
    exec_place,
)


class Plaintext:
    # Initialize from actual values, or from a logical data
    def __init__(self, ctx, values=None, ld=None):
        self.ctx = ctx
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.l = ctx.logical_data(self.values)
        self.symbol = None

    def set_symbol(self, symbol: str):
        self.l.set_symbol(symbol)
        self.symbol = symbol

    def encrypt(self) -> "Ciphertext":
        encrypted = bytearray([c ^ 0x42 for c in self.values])  # toy XOR
        return Ciphertext(self.ctx, values=encrypted)

    def print_values(self):
        with ctx.task(exec_place.host(), self.l.read(data_place.managed())) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            hvalues = t.get_arg_numba(0)
            print([v for v in hvalues])


@cuda.jit
def and_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] & b[i]


@cuda.jit
def or_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] | b[i]


@cuda.jit
def not_kernel(a, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = ~a[i]


@cuda.jit
def xor_kernel(a, out, v):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = a[i] ^ v


class Ciphertext:
    def __init__(self, ctx, values=None, ld=None):
        self.ctx = ctx
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.l = ctx.logical_data(self.values)
        self.symbol = None

    # ~ operator
    def __invert__(self):
        result = self.like_empty()

        with ctx.task(self.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da = t.get_arg_numba(0)
            dresult = t.get_arg_numba(1)
            not_kernel[32, 16, nb_stream](da, dresult)

        return result

    # | operator
    def __or__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented

        result = self.like_empty()

        with ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da = t.get_arg_numba(0)
            db = t.get_arg_numba(1)
            dresult = t.get_arg_numba(2)
            or_kernel[32, 16, nb_stream](da, db, dresult)

        return result

    # & operator
    def __and__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented

        result = self.like_empty()

        with ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            nb_stream.synchronize()
            da = t.get_arg_numba(0)
            db = t.get_arg_numba(1)
            dresult = t.get_arg_numba(2)
            and_kernel[32, 16, nb_stream](da, db, dresult)

        return result

    def set_symbol(self, symbol: str):
        self.l.set_symbol(symbol)
        self.symbol = symbol

    def decrypt(self):
        result = self.like_empty()

        with ctx.task(self.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da = t.get_arg_numba(0)
            dresult = t.get_arg_numba(1)
            # reverse the toy XOR "encryption"
            xor_kernel[32, 16, nb_stream](da, dresult, 0x42)

        return Plaintext(self.ctx, ld=result.l)

    def like_empty(self):
        return Ciphertext(self.ctx, ld=self.l.like_empty())

def circuit(eA: Ciphertext, eB: Ciphertext) -> Ciphertext:
    return ~((eA | ~eB) & (~eA | eB))


ctx = context(use_graph=False)

vA = [3, 3, 2, 2, 17]
pA = Plaintext(ctx, vA)
pA.set_symbol("A")

vB = [1, 7, 7, 7, 49]
pB = Plaintext(ctx, vB)
pB.set_symbol("B")

eA = pA.encrypt()
eB = pB.encrypt()
out = circuit(eA, eB)

out.decrypt().print_values()
ctx.finalize()
