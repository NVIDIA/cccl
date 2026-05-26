"""Single-case reproducer: stream backend, +stream override, no handle.

This is the only variant of probe_warp_btb.py that diverges. We isolate it
so compute-sanitizer / cuda-gdb don't have to wade through the passing
variants too.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

sys.path.insert(0, "/home/caugonnet/git/caugonnet_cccl/python/cuda_cccl/tests/stf")
from test_mlp_ensemble_warp import (  # noqa: E402
    Ensemble,
    clone_weights,
    ref_train_ensemble,
    stf_train_ensemble,
)


def main() -> None:
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        device = wp.get_device()
        stream = wp.Stream(device)

        n = 4
        steps = 4

        ens_ref = Ensemble(n, seed=7)
        ens_stf = Ensemble(n, seed=7)
        clone_weights(ens_ref, ens_stf)

        stream_for_ref = wp.Stream(device)
        ref_train_ensemble(stream_for_ref, ens_ref, steps)
        ref_train_ensemble(stream_for_ref, ens_ref, steps)
        wp.synchronize_stream(stream_for_ref)

        # The repro: two back-to-back STF calls on the same caller stream,
        # stream backend, NO shared async_resources_handle. We sync on the
        # caller stream only (not wp.synchronize()) because that's the
        # exact outbound-finalize contract we're testing: "after finalize,
        # all task work is observable on the caller stream."
        stf_train_ensemble(ens_stf, steps, stream=stream)
        stf_train_ensemble(ens_stf, steps, stream=stream)
        wp.synchronize_stream(stream)

        W1_ref, W2_ref = ens_ref.snapshot_weights()
        W1_stf, W2_stf = ens_stf.snapshot_weights()
        ok = all(
            np.array_equal(W1_ref[k], W1_stf[k])
            and np.array_equal(W2_ref[k], W2_stf[k])
            for k in range(n)
        )
        print("RESULT:", "OK" if ok else "FAIL")
        if not ok:
            for k in range(n):
                d1 = np.abs(W1_ref[k].astype(np.float64) - W1_stf[k].astype(np.float64))
                d2 = np.abs(W2_ref[k].astype(np.float64) - W2_stf[k].astype(np.float64))
                print(f"  member {k}: max|dW1|={d1.max():.3e} max|dW2|={d2.max():.3e}")


if __name__ == "__main__":
    main()
