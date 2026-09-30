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

import random

import numba
import pytest
from numba import cuda

# Skip if the compiled CUDASTF bindings are unavailable (e.g. Windows wheels).
pytest.importorskip("cuda.stf._experimental._stf_bindings")
import cuda.stf._experimental as stf  # noqa: E402
from cuda.stf._experimental.interop.numba import numba_arguments  # noqa: E402


class Plaintext:
    def __init__(self, ctx, values=None, ld=None, key=0x42, name=None, size=None):
        self.ctx = ctx
        self.key = key
        self.size = size
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.size = len(self.values)
            self.l = ctx.logical_data(self.values, name=name)

    def encrypt(self) -> "Ciphertext":
        encrypted = bytearray([(c + self.key) & 0xFF for c in self.values])
        return Ciphertext(self.ctx, values=encrypted, key=self.key)

    def print_values(self):
        self.ctx.host_launch(self.l.read(), fn=lambda x: print(list(x)))


# Grid-stride threads-per-block; the launch grid is sized from the operand
# length so the circuit works for arbitrary-length ciphertexts, not just the
# tiny demo vectors.
_THREADS_PER_BLOCK = 256


def _launch_grid(n: int):
    return (n + _THREADS_PER_BLOCK - 1) // _THREADS_PER_BLOCK, _THREADS_PER_BLOCK


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

    def __init__(self, ctx, values=None, ld=None, key=0x42, name=None, size=None):
        self.ctx = ctx
        self.key = key
        self.size = size
        if ld is not None:
            self.l = ld
        if values is not None:
            self.values = bytearray(values)
            self.size = len(self.values)
            self.l = ctx.logical_data(self.values, name=name)

    def _check_binary_operand(self, other):
        """Reject operands that cannot be combined element-wise.

        Silently proceeding with a mismatched context, length, or key would
        launch kernels over incompatible buffers (or a wrong-length grid) and
        yield garbage that only surfaces much later at decrypt time.
        """
        if other.ctx is not self.ctx:
            raise ValueError("Ciphertext operands belong to different STF contexts")
        if self.size != other.size:
            raise ValueError(f"Ciphertext length mismatch: {self.size} != {other.size}")
        if self.key != other.key:
            raise ValueError(
                f"Ciphertext key mismatch: {self.key:#x} != {other.key:#x}"
            )

    def __add__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        self._check_binary_operand(other)
        result = self.empty_like()
        blocks, tpb = _launch_grid(self.size)
        # The explicit task API makes the dataflow visible: read both inputs,
        # write the result, and let STF schedule the resulting dependency graph.
        with self.ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, db, dresult = numba_arguments(t)
            add_kernel[blocks, tpb, nb_stream](da, db, dresult)
        return result

    def __sub__(self, other):
        if not isinstance(other, Ciphertext):
            return NotImplemented
        self._check_binary_operand(other)
        result = self.empty_like()
        blocks, tpb = _launch_grid(self.size)
        # This task is independent from a sibling add that reads the same inputs
        # and writes a different result, so STF may execute them concurrently.
        with self.ctx.task(self.l.read(), other.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, db, dresult = numba_arguments(t)
            sub_kernel[blocks, tpb, nb_stream](da, db, dresult)
        return result

    def decrypt(self, num_operands=2):
        """Decrypt by subtracting num_operands * key"""
        result = self.empty_like()
        total_key = (num_operands * self.key) & 0xFF
        blocks, tpb = _launch_grid(self.size)
        with self.ctx.task(self.l.read(), result.l.write()) as t:
            nb_stream = cuda.external_stream(t.stream_ptr())
            da, dresult = numba_arguments(t)
            sub_scalar_kernel[blocks, tpb, nb_stream](da, dresult, total_key)
        return Plaintext(self.ctx, ld=result.l, key=self.key, size=self.size)

    def empty_like(self):
        # Preserve the key (and length) so a derived ciphertext still decrypts
        # with the same secret; dropping it would silently reset to the default.
        return Ciphertext(
            self.ctx, ld=self.l.empty_like(), key=self.key, size=self.size
        )


def circuit(a, b):
    """Circuit: (A + B) + (B - A) = 2*B"""
    return (a + b) + (b - a)


def _run_fhe(vA=None, vB=None):
    """Exercise the explicit STF task API used by the composability example."""
    ctx = stf.context(use_graph=False)

    if vA is None:
        vA = [3, 3, 2, 2, 17]
    if vB is None:
        vB = [1, 7, 7, 7, 49]

    pA = Plaintext(ctx, vA, name="A")
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


def test_fhe(monkeypatch):
    monkeypatch.setattr(numba.cuda.config, "CUDA_LOW_OCCUPANCY_WARNINGS", 0)
    _run_fhe()


@pytest.mark.parametrize("length", [1, 5, 256, 257, 1000])
def test_fhe_arbitrary_lengths(monkeypatch, length):
    """The circuit must work for lengths beyond a single thread block."""
    monkeypatch.setattr(numba.cuda.config, "CUDA_LOW_OCCUPANCY_WARNINGS", 0)
    rng = random.Random(length)
    vA = [rng.randint(0, 255) for _ in range(length)]
    vB = [rng.randint(0, 255) for _ in range(length)]
    _run_fhe(vA, vB)


def test_fhe_empty_like_preserves_key(monkeypatch):
    """A ciphertext derived via ``empty_like`` keeps the source key."""
    monkeypatch.setattr(numba.cuda.config, "CUDA_LOW_OCCUPANCY_WARNINGS", 0)
    ctx = stf.context(use_graph=False)
    e = Plaintext(ctx, [1, 2, 3], key=0x37).encrypt()
    derived = e.empty_like()
    assert derived.key == e.key
    assert derived.size == e.size
    ctx.finalize()


def test_fhe_rejects_mismatched_operands(monkeypatch):
    """Binary ops reject different contexts, lengths, or keys."""
    monkeypatch.setattr(numba.cuda.config, "CUDA_LOW_OCCUPANCY_WARNINGS", 0)
    ctx = stf.context(use_graph=False)
    ctx2 = stf.context(use_graph=False)

    a = Plaintext(ctx, [1, 2, 3], key=0x42).encrypt()
    diff_len = Plaintext(ctx, [1, 2], key=0x42).encrypt()
    diff_key = Plaintext(ctx, [1, 2, 3], key=0x11).encrypt()
    diff_ctx = Plaintext(ctx2, [1, 2, 3], key=0x42).encrypt()

    with pytest.raises(ValueError, match="length mismatch"):
        _ = a + diff_len
    with pytest.raises(ValueError, match="key mismatch"):
        _ = a + diff_key
    with pytest.raises(ValueError, match="different STF contexts"):
        _ = a + diff_ctx

    ctx.finalize()
    ctx2.finalize()


def main():
    previous = numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS
    numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = 0
    try:
        _run_fhe()
    finally:
        numba.cuda.config.CUDA_LOW_OCCUPANCY_WARNINGS = previous


if __name__ == "__main__":
    main()
