# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Generate reference data from Mamba's CUDA selective_scan kernel.

Run with a CUDA-capable PyTorch environment and a local mamba checkout.
Example:
  CUDA_VISIBLE_DEVICES=1 \
  MAMBA_SCAN_OPTION=cuda \
  /path/to/python tests/coop/generate_mamba_selective_scan_fwd_ref.py \
    --mamba-root ~/src/mamba \
    --output tests/coop/data/mamba_selective_scan_fwd_ref.npz
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import numpy as np


def _resolve_mamba_root(value: str | None) -> Path:
    if value:
        root = Path(value).expanduser()
    else:
        root = Path("~/src/mamba").expanduser()
    if not root.exists():
        raise FileNotFoundError(f"Mamba root not found: {root}")
    return root


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--mamba-root",
        default=None,
        help="Path to the mamba repo (default: ~/src/mamba)",
    )
    parser.add_argument(
        "--output",
        default="tests/coop/data/mamba_selective_scan_fwd_ref.npz",
        help="Output .npz path",
    )
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--seqlen", type=int, default=512)
    args = parser.parse_args()

    mamba_root = _resolve_mamba_root(args.mamba_root)
    os.environ.setdefault("MAMBA_SCAN_OPTION", "cuda")

    import sys

    sys.path.insert(0, str(mamba_root))

    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn

    torch.manual_seed(args.seed)
    device = "cuda"

    batch = 1
    dim = 1
    seqlen = args.seqlen

    u = torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)
    delta = 0.1 * torch.randn(batch, dim, seqlen, device=device, dtype=torch.float32)

    A = torch.tensor([[-0.2]], device=device, dtype=torch.float32)
    B = torch.tensor([[0.7]], device=device, dtype=torch.float32)
    C = torch.tensor([[-0.3]], device=device, dtype=torch.float32)
    D = torch.tensor([0.5], device=device, dtype=torch.float32)
    delta_bias = torch.tensor([0.01], device=device, dtype=torch.float32)

    out = selective_scan_fn(
        u,
        delta,
        A,
        B,
        C,
        D=D,
        z=None,
        delta_bias=delta_bias,
        delta_softplus=False,
        return_last_state=False,
    )

    output_path = Path(args.output).expanduser()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    np.savez(
        output_path,
        u=u.detach().cpu().numpy().reshape(-1),
        delta=delta.detach().cpu().numpy().reshape(-1),
        A=A.detach().cpu().numpy().reshape(-1),
        B=B.detach().cpu().numpy().reshape(-1),
        C=C.detach().cpu().numpy().reshape(-1),
        D=D.detach().cpu().numpy().reshape(-1),
        delta_bias=delta_bias.detach().cpu().numpy().reshape(-1),
        out=out.detach().cpu().numpy().reshape(-1),
        seqlen=np.array([seqlen], dtype=np.int32),
        seed=np.array([args.seed], dtype=np.int32),
    )

    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
