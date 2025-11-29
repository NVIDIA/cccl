# Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Toy Fully Homomorphic Encryption (FHE) example with addition encryption

import numba
from numba import cuda

import cuda.stf as cudastf

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


class Plaintext:
    def __init__(self, ctx, values=None, ld=None, key=0x42):
        self.ctx = ctx
        self.key = key
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
        encrypted = bytearray([(c + self.key) & 0xFF for c in self.values])
        return Ciphertext(self.ctx, values=encrypted, key=self.key)

    def print_values(self):
        with ctx.task(
            cudastf.exec_place.host(), self.l.read(cudastf.data_place.managed())
        ) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            nb_stream.synchronize()
            hvalues = t.numba_arguments()
            print([v for v in hvalues])


@cudastf.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] + b[i]) & 0xFF


@cudastf.jit
def sub_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] - b[i]) & 0xFF


@cudastf.jit
def sub_scalar_kernel(a, out, v):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] - v) & 0xFF


class Ciphertext:
    def __init__(self, ctx, values=None, ld=None, key=0x42):
        self.ctx = ctx
        self.key = key
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.l = ctx.logical_data(self.values)
        self.symbol = None

    def __add__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        result = self.like_empty()
        add_kernel[32, 16](self.l.read(), other.l.read(), result.l.write())
        return result

    def __sub__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        result = self.like_empty()
        sub_kernel[32, 16](self.l.read(), other.l.read(), result.l.write())
        return result

    def set_symbol(self, symbol: str):
        self.l.set_symbol(symbol)
        self.symbol = symbol

    def decrypt(self, num_operands=2):
        """Decrypt by subtracting num_operands * key"""
        result = self.like_empty()
        total_key = (num_operands * self.key) & 0xFF
        sub_scalar_kernel[32, 16](self.l.read(), result.l.write(), total_key)
        return Plaintext(self.ctx, ld=result.l, key=self.key)

    def like_empty(self):
        return Ciphertext(self.ctx, ld=self.l.like_empty())


def circuit(a, b):
    """Circuit: (A + B) + (B - A) = 2*B"""
    return (a + b) + (b - a)


def test_fhe_decorator():
    """Test FHE using @cudastf.jit decorators with addition encryption."""
    global ctx
    ctx = cudastf.context(use_graph=False)

    vA = [3, 3, 2, 2, 17]
    pA = Plaintext(ctx, vA)
    pA.set_symbol("A")

    vB = [1, 7, 7, 7, 49]
    pB = Plaintext(ctx, vB)
    pB.set_symbol("B")

    expected = [circuit(a, b) & 0xFF for a, b in zip(vA, vB)]

    eA = pA.encrypt()
    eB = pB.encrypt()
    encrypted_out = circuit(eA, eB)
    decrypted_out = encrypted_out.decrypt(num_operands=2)

    with ctx.task(
        cudastf.exec_place.host(), decrypted_out.l.read(cudastf.data_place.managed())
    ) as t:
        nb_stream = cuda.external_stream(t.stream_ptr())
        nb_stream.synchronize()
        hvalues = t.numba_arguments()
        actual = [int(v) for v in hvalues]

    ctx.finalize()

    assert actual == expected, (
        f"Decrypted result {actual} doesn't match expected {expected}"
    )


if __name__ == "__main__":
    test_fhe_decorator()
