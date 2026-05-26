"""Back-to-back probe: run ref vs stf_train_ensemble twice in a row on
every combination of (backend, overrides), no explicit sync between the
two STF calls. If the outbound-finalize contract is honored end-to-end,
every variant should match the reference.
"""

from __future__ import annotations

import sys

import numpy as np
import warp as wp

import cuda.stf._experimental as stf

sys.path.insert(0, "/home/caugonnet/git/caugonnet_cccl/python/cuda_cccl/tests/stf")
from test_mlp_ensemble_warp import (  # noqa: E402
    Ensemble,
    clone_weights,
    ref_train_ensemble,
    stf_train_ensemble,
)


def _check(variant: str, kwargs: dict) -> bool:
    n = 4
    steps = 4

    ens_ref = Ensemble(n, seed=7)
    ens_stf = Ensemble(n, seed=7)
    clone_weights(ens_ref, ens_stf)

    stream_for_ref = wp.Stream(ens_ref.device)
    ref_train_ensemble(stream_for_ref, ens_ref, steps)
    ref_train_ensemble(stream_for_ref, ens_ref, steps)
    wp.synchronize_stream(stream_for_ref)

    stf_train_ensemble(ens_stf, steps, **kwargs)
    stf_train_ensemble(ens_stf, steps, **kwargs)
    wp.synchronize()

    W1_ref, W2_ref = ens_ref.snapshot_weights()
    W1_stf, W2_stf = ens_stf.snapshot_weights()
    ok = all(
        np.array_equal(W1_ref[k], W1_stf[k]) and np.array_equal(W2_ref[k], W2_stf[k])
        for k in range(n)
    )
    print(f"  {variant:<24s} {'OK' if ok else 'FAIL'}")
    return ok


def main() -> None:
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        device = wp.get_device()
        stream = wp.Stream(device)
        handle = stf.async_resources()

        print("stream backend")
        _check("plain", {})
        _check("+handle", {"handle": handle})
        _check("+stream", {"stream": stream})
        _check("+stream+handle", {"stream": stream, "handle": handle})

        print("\ngraph backend")
        _check("plain", {"use_graph": True})
        _check("+handle", {"use_graph": True, "handle": handle})
        _check("+stream", {"use_graph": True, "stream": stream})
        _check(
            "+stream+handle", {"use_graph": True, "stream": stream, "handle": handle}
        )


if __name__ == "__main__":
    main()
