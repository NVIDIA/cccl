"""Ablation: isolate which part of the `_wp_stream_cache` machinery fixes
the stream-backend / +stream / no-handle back-to-back Warp race.

Cases:
  A. No cache at all (build fresh wp.Stream every wrap).
  B. Cache ON, pre-populate OFF (only pool ptrs cached; caller ptr
     still triggers a fresh wrap on first hit).
  C. Cache ON, pre-populate ON (current baseline that passes).

All three run the same back-to-back repro, check correctness vs ref.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import cuda.stf._experimental as stf

sys.path.insert(0, "/home/caugonnet/git/caugonnet_cccl/python/cuda_cccl/tests/stf")
import test_mlp_ensemble_warp as demo  # noqa: E402


def _run_once(
    use_cache: bool,
    pre_populate: bool,
    use_handle: bool,
    sync_between: bool = False,
) -> bool:
    n, steps = 4, 4
    ens_ref = demo.Ensemble(n, seed=7)
    ens_stf = demo.Ensemble(n, seed=7)
    demo.clone_weights(ens_ref, ens_stf)

    device = ens_ref.device
    ref_stream = wp.Stream(device)
    stream = wp.Stream(device)
    handle = stf.async_resources() if use_handle else None

    demo.ref_train_ensemble(ref_stream, ens_ref, steps)
    demo.ref_train_ensemble(ref_stream, ens_ref, steps)
    wp.synchronize_stream(ref_stream)

    _orig_wrap = demo._wrap_stream
    demo._wp_stream_cache.clear()

    if not use_cache:

        def no_cache_wrap(raw_ptr, dev):
            return wp.Stream(dev, cuda_stream=int(raw_ptr))

        demo._wrap_stream = no_cache_wrap
    # else: use the normal cache as-is.

    def call():
        if pre_populate and use_cache:
            demo._wp_stream_cache[(id(device), int(stream.cuda_stream))] = stream
        ctx = stf.context(stream=stream.cuda_stream, handle=handle)
        tokens = [ctx.token() for _ in range(n)]
        BLOCKS_W2 = demo.D_OUT * demo.D_HID
        BLOCKS_W1 = demo.D_HID * demo.D_IN
        for _ in range(steps):
            for k in range(n):
                with ctx.task(tokens[k].rw()) as t:
                    s = demo._wrap_stream(t.stream_ptr(), device)
                    wp.launch(
                        kernel=demo.fwd_L1,
                        dim=demo.D_HID,
                        inputs=[ens_stf.W1[k], ens_stf.x[k], ens_stf.z[k]],
                        device=device,
                        stream=s,
                    )
                    wp.launch(
                        kernel=demo.fwd_L2,
                        dim=demo.D_OUT,
                        inputs=[ens_stf.W2[k], ens_stf.z[k], ens_stf.y[k]],
                        device=device,
                        stream=s,
                    )
                    wp.launch(
                        kernel=demo.bwd_gz,
                        dim=demo.D_HID,
                        inputs=[
                            ens_stf.y[k],
                            ens_stf.target[k],
                            ens_stf.W2[k],
                            ens_stf.z[k],
                            ens_stf.gz[k],
                        ],
                        device=device,
                        stream=s,
                    )
                    wp.launch(
                        kernel=demo.upd_W2,
                        dim=BLOCKS_W2,
                        inputs=[
                            ens_stf.y[k],
                            ens_stf.target[k],
                            ens_stf.z[k],
                            ens_stf.W2[k],
                            demo.LR,
                        ],
                        device=device,
                        stream=s,
                    )
                    wp.launch(
                        kernel=demo.upd_W1,
                        dim=BLOCKS_W1,
                        inputs=[ens_stf.gz[k], ens_stf.x[k], ens_stf.W1[k], demo.LR],
                        device=device,
                        stream=s,
                    )
        ctx.finalize()

    try:
        call()
        if sync_between:
            wp.synchronize_stream(stream)
        call()
        wp.synchronize_stream(stream)
        W1r, W2r = ens_ref.snapshot_weights()
        W1s, W2s = ens_stf.snapshot_weights()
        ok = all(
            np.array_equal(W1r[k], W1s[k]) and np.array_equal(W2r[k], W2s[k])
            for k in range(n)
        )
        return ok
    finally:
        demo._wrap_stream = _orig_wrap


def main() -> None:
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        cases = [
            # (label, cache, pre, handle, sync_between)
            ("NOhandle, no sync", True, True, False, False),
            ("NOhandle, sync between", True, True, False, True),
            ("handle,   no sync", True, True, True, False),
            ("handle,   sync between", True, True, True, True),
        ]
        for label, use_cache, pre, use_handle, sync in cases:
            ok_runs = sum(_run_once(use_cache, pre, use_handle, sync) for _ in range(5))
            print(f"{label:<32s} {ok_runs}/5 OK")


if __name__ == "__main__":
    main()
