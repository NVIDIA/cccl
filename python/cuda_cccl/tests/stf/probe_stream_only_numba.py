"""Numba analog of c/experimental/stf/test/test_stream_ctx_override.cu.

Minimal back-to-back probe using Numba kernels (no Warp) on the Python
binding path:
  - single CUDA stream owned by the caller (cupy / cuda-python)
  - stream_ctx(stream) on the stream backend, NO handle
  - two contexts in sequence, each launches a slow token task that writes
    `value` into a device buffer
  - verify the final buffer contains the LAST write (2)
If the C-facade + binding layer honors the caller-stream chaining contract
end-to-end, this should always pass, matching the C unit test.
"""

from __future__ import annotations

import numpy as np
import pytest

from cuda.bindings import runtime as cudart

pytest.importorskip("numba")
pytest.importorskip("numba.cuda")
from numba import cuda

import cuda.stf._experimental as stf

N = 1 << 14
ITERS = 1 << 18
N_ROUNDS = 20


@cuda.jit
def slow_set(arr, value, iters):
    i = cuda.grid(1)
    if i >= arr.size:
        return
    acc = 0
    for k in range(iters):
        acc += (k * 1103515245 + 12345) & 0x7FFFFFFF
    # commit value; the (acc & 0) term keeps the busy loop from being elided
    arr[i] = value + (acc & 0)


def _check_cuda(err):
    err = err[0] if isinstance(err, tuple) else err
    if int(err) != 0:
        raise RuntimeError(f"cudart error {int(err)}")


def main() -> None:
    err, s_raw = cudart.cudaStreamCreate()
    _check_cuda(err)

    # Numba stream wrapping the caller's raw cudaStream_t.
    numba_stream = cuda.external_stream(int(s_raw))

    d_arr = cuda.device_array(N, dtype=np.int32, stream=numba_stream)
    ptr = int(d_arr.device_ctypes_pointer.value)
    cudart.cudaMemsetAsync(ptr, 0, N * d_arr.dtype.itemsize, s_raw)

    threads = 128
    blocks = (N + threads - 1) // threads

    ok = 0
    bad = 0
    for r in range(N_ROUNDS):
        # Context 1: write value 1 via slow kernel
        ctx = stf.context(stream=int(s_raw))
        tok = ctx.token()
        with ctx.task(tok.rw()) as t:
            task_stream = cuda.external_stream(int(t.stream_ptr()))
            slow_set[blocks, threads, task_stream](d_arr, 1, ITERS)
        ctx.finalize()

        # Context 2: write value 2 via slow kernel. Same caller stream,
        # NO handle; ctx2's task MUST wait on ctx1's work through `s_raw`.
        ctx = stf.context(stream=int(s_raw))
        tok = ctx.token()
        with ctx.task(tok.rw()) as t:
            task_stream = cuda.external_stream(int(t.stream_ptr()))
            slow_set[blocks, threads, task_stream](d_arr, 2, ITERS)
        ctx.finalize()

        ret = cudart.cudaStreamSynchronize(s_raw)
        _check_cuda(ret)

        h_arr = d_arr.copy_to_host()
        if np.all(h_arr == 2):
            ok += 1
        else:
            bad += 1
            mismatches = int(np.sum(h_arr != 2))
            first_idx = int(np.argmax(h_arr != 2))
            print(
                f"  round {r:3d}  FAIL: {mismatches}/{N} slots != 2, "
                f"first mismatch idx={first_idx} val={int(h_arr[first_idx])}"
            )

    print(
        f"\nnumba + C-facade stream-only back-to-back: {ok} OK, {bad} FAIL (rounds={N_ROUNDS})"
    )


if __name__ == "__main__":
    main()
