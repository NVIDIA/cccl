# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Toy encrypted arithmetic example demonstrating STF composability.

The user-facing API is ordinary Python arithmetic over ``Ciphertext`` objects:

    encrypted_out = circuit(eA, eB)

The circuit computes ``(A + B) + (B - A)``. The two inner operations, ``A + B``
and ``B - A``, are independent and may run concurrently; the final add depends
on both temporary results.

In a manual stream/event implementation, the caller would need to express that
scheduling explicitly, for example:

    tmp1 = add(A, B) on stream_add
    tmp2 = sub(B, A) on stream_sub
    record events for tmp1/tmp2
    final stream waits on both events
    out = add(tmp1, tmp2)

With STF, each ``Ciphertext`` operation instead creates a task over logical data.
Each task declares its reads and writes, and STF derives the task dependencies
from those declarations. The high-level circuit composes ordinary arithmetic
operations without exposing CUDA streams or events to the user.
"""

import pytest

numba = pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda  # noqa: E402
from numba_helpers import numba_arguments  # noqa: E402

import cuda.stf._experimental as stf  # noqa: E402

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
    """Encrypted byte array whose arithmetic records STF tasks."""

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
        # The explicit task API makes the dataflow visible: read both inputs,
        # write the result, and let STF schedule the resulting dependency graph.
        with self.ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, db, dresult = numba_arguments(t)
            add_kernel[32, 16, nb_stream](da, db, dresult)
        return result

    def __sub__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        result = self.empty_like()
        # This task is independent from a sibling add that reads the same inputs
        # and writes a different result, so STF may execute them concurrently.
        with self.ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, db, dresult = numba_arguments(t)
            sub_kernel[32, 16, nb_stream](da, db, dresult)
        return result

    def decrypt(self, num_operands=2):
        """Decrypt by subtracting num_operands * key"""
        result = self.empty_like()
        total_key = (num_operands * self.key) & 0xFF
        with self.ctx.task(self.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, dresult = numba_arguments(t)
            sub_scalar_kernel[32, 16, nb_stream](da, dresult, total_key)
        return Plaintext(self.ctx, ld=result.l, key=self.key)

    def empty_like(self):
        return Ciphertext(self.ctx, ld=self.l.empty_like())


def circuit(a, b):
    """Circuit: (A + B) + (B - A) = 2*B"""
    return (a + b) + (b - a)


def test_fhe():
    """Exercise the explicit STF task API used by the composability example."""
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


if __name__ == "__main__":
    test_fhe()
