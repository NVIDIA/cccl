"""Count unique (device, raw_ptr) pairs observed by `_wrap_stream`
across the 8 probe_warp_btb variants. Tells us whether the cache is
ever serving more than the caller's one pre-populated entry.
"""

from __future__ import annotations

import sys

import warp as wp

import cuda.stf._experimental as stf

sys.path.insert(0, "/home/caugonnet/git/caugonnet_cccl/python/cuda_cccl/tests/stf")
import test_mlp_ensemble_warp as demo  # noqa: E402

_orig_wrap = demo._wrap_stream
_obs: dict[str, set[tuple[int, int]]] = {}


def _probe(variant: str):
    seen: set[tuple[int, int]] = set()
    _obs[variant] = seen

    def tracking_wrap(raw_ptr, device):
        seen.add((id(device), int(raw_ptr)))
        return _orig_wrap(raw_ptr, device)

    demo._wrap_stream = tracking_wrap


def main() -> None:
    wp.init()
    with wp.ScopedDevice("cuda:0"):
        device = wp.get_device()
        stream = wp.Stream(device)
        handle = stf.async_resources()

        n, steps = 4, 4

        cases = {
            "stream / plain": {},
            "stream / +handle": {"handle": handle},
            "stream / +stream": {"stream": stream},
            "stream / +stream+handle": {"stream": stream, "handle": handle},
            "graph  / plain": {"use_graph": True},
            "graph  / +handle": {"use_graph": True, "handle": handle},
            "graph  / +stream": {"use_graph": True, "stream": stream},
            "graph  / +stream+handle": {
                "use_graph": True,
                "stream": stream,
                "handle": handle,
            },
        }

        for name, kwargs in cases.items():
            _probe(name)
            ens = demo.Ensemble(n, seed=7)
            # two back-to-back calls, no explicit sync between them
            demo.stf_train_ensemble(ens, steps, **kwargs)
            demo.stf_train_ensemble(ens, steps, **kwargs)
            wp.synchronize()

        caller_ptr = int(stream.cuda_stream)
        print(f"caller raw_ptr = 0x{caller_ptr:016x}\n")
        for name, seen in _obs.items():
            n_unique = len(seen)
            caller_seen = any(p == caller_ptr for (_, p) in seen)
            other_ptrs = sorted(p for (_, p) in seen if p != caller_ptr)
            print(f"{name}")
            print(f"   unique (device, ptr) pairs : {n_unique}")
            print(f"   caller ptr observed        : {caller_seen}")
            print(f"   other (pool) ptrs          : {len(other_ptrs)}")
            for p in other_ptrs[:8]:
                print(f"     0x{p:016x}")
            if len(other_ptrs) > 8:
                print(f"     ... and {len(other_ptrs) - 8} more")
            print()


if __name__ == "__main__":
    main()
