# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Decorator-based variant of the STF FHE composability smoke test.

``test_fhe.py`` is the primary teaching example because its explicit
``ctx.task(...)`` calls show how logical-data dependencies build the task graph.
This file keeps the same toy encrypted arithmetic API, but uses the STF-aware
``@jit`` decorator to cover the ergonomic integration path.
"""

import numba
from numba import cuda

import cuda.stf._experimental as stf
from cuda.stf._experimental.interop.numba import jit

numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


class Plaintext:
    def __init__(self, ctx, values=None, ld=None, key=0x42, name=None):
        self.ctx = ctx
        self.key = key
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.l = ctx.logical_data(self.values, name=name)

    def encrypt(self) -> "Ciphertext":
        encrypted = bytearray([(c + self.key) & 0xFF for c in self.values])
        return Ciphertext(self.ctx, values=encrypted, key=self.key)

    def print_values(self):
        self.ctx.host_launch(self.l.read(), fn=lambda x: print(list(x)))


@jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] + b[i]) & 0xFF


@jit
def sub_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] - b[i]) & 0xFF


@jit
def sub_scalar_kernel(a, out, v):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] - v) & 0xFF


class Ciphertext:
    def __init__(self, ctx, values=None, ld=None, key=0x42, name=None):
        self.ctx = ctx
        self.key = key
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.l = ctx.logical_data(self.values, name=name)

    def __add__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        result = self.empty_like()
        add_kernel[32, 16](self.l.read(), other.l.read(), result.l.write())
        return result

    def __sub__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        result = self.empty_like()
        sub_kernel[32, 16](self.l.read(), other.l.read(), result.l.write())
        return result

    def decrypt(self, num_operands=2):
        """Decrypt by subtracting num_operands * key"""
        result = self.empty_like()
        total_key = (num_operands * self.key) & 0xFF
        sub_scalar_kernel[32, 16](self.l.read(), result.l.write(), total_key)
        return Plaintext(self.ctx, ld=result.l, key=self.key)

    def empty_like(self):
        return Ciphertext(self.ctx, ld=self.l.empty_like())


def circuit(a, b):
    """Circuit: (A + B) + (B - A) = 2*B"""
    return (a + b) + (b - a)


def test_fhe_decorator():
    """Exercise the decorator integration variant of the FHE example."""
    ctx = stf.context(use_graph=False)

    vA = [3, 3, 2, 2, 17]
    pA = Plaintext(ctx, vA, name="A")

    vB = [1, 7, 7, 7, 49]
    pB = Plaintext(ctx, vB, name="B")

    expected = [circuit(a, b) & 0xFF for a, b in zip(vA, vB)]

    eA = pA.encrypt()
    eB = pB.encrypt()
    encrypted_out = circuit(eA, eB)
    decrypted_out = encrypted_out.decrypt(num_operands=2)

    actual = []
    ctx.host_launch(
        decrypted_out.l.read(),
        fn=lambda x, out: out.extend(int(v) for v in x),
        args=[actual],
    )

    ctx.finalize()

    assert actual == expected, (
        f"Decrypted result {actual} doesn't match expected {expected}"
    )


def main():
    test_fhe_decorator()


if __name__ == "__main__":
    main()
