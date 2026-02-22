# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# Toy Fully Homomorphic Encryption (FHE) example with addition encryption

import numba
from numba import cuda

import cuda.stf as stf

numba.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0


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
        with self.ctx.task(
            stf.exec_place.host(), self.l.read(stf.data_place.managed())
        ) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            nb_stream.synchronize()
            hvalues = t.numba_arguments()
            print([v for v in hvalues])


@cuda.jit
def add_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] + b[i]) & 0xFF


@cuda.jit
def sub_kernel(a, b, out):
    i = cuda.grid(1)
    if i < out.size:
        out[i] = (a[i] - b[i]) & 0xFF


@cuda.jit
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
        with self.ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, db, dresult = t.numba_arguments()
            add_kernel[32, 16, nb_stream](da, db, dresult)
        return result

    def __sub__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        result = self.empty_like()
        with self.ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, db, dresult = t.numba_arguments()
            sub_kernel[32, 16, nb_stream](da, db, dresult)
        return result

    def decrypt(self, num_operands=2):
        """Decrypt by subtracting num_operands * key"""
        result = self.empty_like()
        total_key = (num_operands * self.key) & 0xFF
        with self.ctx.task(self.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, dresult = t.numba_arguments()
            sub_scalar_kernel[32, 16, nb_stream](da, dresult, total_key)
        return Plaintext(self.ctx, ld=result.l, key=self.key)

    def empty_like(self):
        return Ciphertext(self.ctx, ld=self.l.empty_like())


def circuit(a, b):
    """Circuit: (A + B) + (B - A) = 2*B"""
    return (a + b) + (b - a)


def test_fhe():
    """Test FHE using manual task creation with addition encryption."""
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

    with ctx.task(
        stf.exec_place.host(), decrypted_out.l.read(stf.data_place.managed())
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
    test_fhe()
