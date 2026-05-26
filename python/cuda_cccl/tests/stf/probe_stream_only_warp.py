"""Warp analog of probe_stream_only_numba.py.

Same pattern as the Numba probe and the C-facade unit test:
  - caller-owned CUDA stream,
  - two back-to-back `stf.context(stream=s)` on the stream backend,
  - NO handle, so each context gets a fresh async_resources pool,
  - each context submits one token task that launches a slow kernel
    writing `value` into a device buffer.

The only difference from the Numba probe is that kernels are launched
via Warp (`wp.launch(stream=wp.Stream(cuda_stream=<task ptr>))`). We
flip `sync_at_task_end` to check whether an extra sync on the task's
stream inside the task body is sufficient to make the back-to-back
chaining work with Warp.
"""

from __future__ import annotations

import numpy as np
import warp as wp

import cuda.stf._experimental as stf
from cuda.bindings import runtime as cudart

N = 1 << 12
ITERS = 1 << 16
N_ROUNDS = 20
N_TOKENS = 4  # ensemble members
N_STEPS = 4  # training steps per context
N_KERNELS = 5  # kernels chained inside each task body


@wp.kernel
def slow_set(arr: wp.array(dtype=wp.int32), value: wp.int32, iters: wp.int32):
    i = wp.tid()
    if i >= arr.shape[0]:
        return
    acc = wp.int32(0)
    for k in range(iters):
        acc += (k * 1103515245 + 12345) & 0x7FFFFFFF
    arr[i] = value + (acc & 0)


def _check_cuda(err) -> None:
    if isinstance(err, tuple):
        err = err[0]
    if int(err) != 0:
        raise RuntimeError(f"cudart error {int(err)}")


_wp_stream_cache: dict[int, "wp.Stream"] = {}


def _wrap(
    ptr: int, device: "wp.Device", caller: "wp.Stream | None" = None
) -> "wp.Stream":
    # Cache so that STF-provided pool-stream ptrs are wrapped exactly once,
    # and the caller's own wp.Stream (if any) is reused when STF hands the
    # same ptr back to a task (can happen on the stream backend).
    key = int(ptr)
    s = _wp_stream_cache.get(key)
    if s is not None:
        return s
    if caller is not None and int(caller.cuda_stream) == key:
        _wp_stream_cache[key] = caller
        return caller
    s = wp.Stream(device, cuda_stream=ptr)
    _wp_stream_cache[key] = s
    return s


def run(sync_at_task_end: bool) -> tuple[int, int]:
    err, s_raw = cudart.cudaStreamCreate()
    _check_cuda(err)

    device = wp.get_device("cuda:0")
    caller_wp = wp.Stream(device, cuda_stream=int(s_raw))
    _wp_stream_cache.clear()
    _wp_stream_cache[int(s_raw)] = caller_wp

    arrs = [wp.zeros(N, dtype=wp.int32, device=device) for _ in range(N_TOKENS)]

    ok = 0
    bad = 0
    for r in range(N_ROUNDS):
        for value in (1, 2):
            ctx = stf.context(stream=int(s_raw))
            toks = [ctx.token() for _ in range(N_TOKENS)]

            for _ in range(N_STEPS):
                for k in range(N_TOKENS):
                    with ctx.task(toks[k].rw()) as t:
                        task_stream = _wrap(int(t.stream_ptr()), device, caller_wp)
                        for _kern in range(N_KERNELS):
                            wp.launch(
                                kernel=slow_set,
                                dim=N,
                                inputs=[arrs[k], value, ITERS],
                                device=device,
                                stream=task_stream,
                            )
                        if sync_at_task_end:
                            wp.synchronize_stream(task_stream)
            ctx.finalize()

        ret = cudart.cudaStreamSynchronize(s_raw)
        _check_cuda(ret)

        round_bad = 0
        first_bad_k = -1
        first_bad_idx = -1
        first_bad_val = None
        for k in range(N_TOKENS):
            h_arr = arrs[k].numpy()
            if not np.all(h_arr == 2):
                round_bad += int(np.sum(h_arr != 2))
                if first_bad_k < 0:
                    first_bad_k = k
                    first_bad_idx = int(np.argmax(h_arr != 2))
                    first_bad_val = int(h_arr[first_bad_idx])
        if round_bad == 0:
            ok += 1
        else:
            bad += 1
            print(
                f"  round {r:3d}  FAIL: {round_bad}/{N_TOKENS * N} slots != 2, "
                f"first mismatch tok={first_bad_k} idx={first_bad_idx} val={first_bad_val}"
            )
    return ok, bad


def main() -> None:
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        print("variant: NO per-task sync (expect failures)")
        ok, bad = run(sync_at_task_end=False)
        print(f"  -> {ok} OK, {bad} FAIL")
        print()
        print("variant: sync_at_task_end=True (wp.synchronize_stream inside task)")
        ok, bad = run(sync_at_task_end=True)
        print(f"  -> {ok} OK, {bad} FAIL")


if __name__ == "__main__":
    main()
