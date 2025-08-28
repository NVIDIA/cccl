# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# A toy example to illustrate how we can compose logical operations

import numba
import numpy as np
import pytest
from numba import cuda

numba.config.CUDA_ENABLE_PYNVJITLINK = 1
numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0

from cuda.cccl.experimental.stf._stf_bindings import (
    context,
    read,
    rw,
    write,
)

class Plaintext:
    def __init__(self, ctx, values=None, ld=None):
        self.ctx = ctx
        if not ld is None:
           self.l = ld
        if not values is None:
           self.values = bytearray(values)
           self.l = ctx.logical_data(self.values)
        self.symbol = None

    def set_symbol(self, symbol: str):
        self.l.set_symbol(symbol)
        self.symbol = symbol

    def convert_to_vector(self) -> bytearray:
        result = bytearray(self.l.buffer)
        return result

    def encrypt(self) -> "Ciphertext":
        # stub: should return a Ciphertext object wrapping a LogicalData
        encrypted = bytearray([c ^ 0x42 for c in self.values])  # toy XOR
        return Ciphertext(self.ctx, values=encrypted)

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

class Ciphertext:
    def __init__(self, ctx, values=None, ld=None):
        self.ctx = ctx
        if not ld is None:
           self.l = ld
        if values is not None:
           self.values = bytearray(values)
           self.l = ctx.logical_data(self.values)
        self.symbol = None

    # ~ operator
    def __invert__(self):
        result=Ciphertext(ctx, values=None, ld=self.l.like_empty())

        with ctx.task(self.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da = t.get_arg_numba(0)
            dresult = t.get_arg_numba(1)
            not_kernel[32, 16, nb_stream](da, dresult)

        return result

    # | operator
    def __or__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented

        result=Ciphertext(ctx, ld=self.l.like_empty())

        with ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da = t.get_arg_numba(0)
            db = t.get_arg_numba(1)
            dresult = t.get_arg_numba(2)
            or_kernel[32, 16, nb_stream](da, db, dresult)

        return result


    # & operator
    def __and__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented

        result=Ciphertext(ctx, ld=self.l.like_empty())

        with ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da = t.get_arg_numba(0)
            db = t.get_arg_numba(1)
            dresult = t.get_arg_numba(2)
            and_kernel[32, 16, nb_stream](da, db, dresult)

        return result

    def set_symbol(self, symbol: str):
        self.l.set_symbol(symbol)
        self.symbol = symbol

    def decrypt(self):
        # reverse the toy XOR "encryption"
        decrypted = bytearray([c ^ 0x42 for c in self.values])
        return Plaintext(self.ctx, decrypted)

def circuit(eA: Ciphertext, eB: Ciphertext) -> Ciphertext:
    return (~((eA | ~eB) & (~eA | eB)))

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

ctx.finalize()

# v_out = out.decrypt().values
# print("Output vector:", list(v_out))


