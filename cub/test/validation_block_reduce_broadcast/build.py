# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the identical downstream consumer against selected CCCL headers."""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import torch
from torch.utils.cpp_extension import load

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--variant", choices=["baseline", "patch"], required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
args.build.mkdir(parents=True, exist_ok=True)
harness = Path(__file__).resolve().parent
source = args.source.resolve()
extension = load(
    name=f"cccl_{args.variant}_h20",
    sources=[str(harness / "bindings.cpp"), str(harness / "normalization.cu")],
    extra_include_paths=[
        str(source / name) for name in ["cub", "libcudacxx/include", "thrust"]
    ],
    extra_cflags=["-O3", "-std=c++17"],
    extra_cuda_cflags=[
        "-O3",
        "-lineinfo",
        "-std=c++17",
        f"-DUSE_BROADCAST={int(args.variant == 'patch')}",
        "-gencode=arch=compute_90,code=sm_90",
    ],
    build_directory=str(args.build.resolve()),
    verbose=True,
)
library = Path(extension.__file__)
files = [
    source / "cub/cub/block/block_reduce.cuh",
    source / "cub/cub/block/specializations/block_reduce_warp_reductions.cuh",
    harness / "bindings.cpp",
    harness / "normalization.cu",
    library,
    args.build / "build.ninja",
]
record = {
    "variant": args.variant,
    "torch": torch.__version__,
    "cuda": torch.version.cuda,
    "source_sha": subprocess.check_output(
        ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
    ).strip(),
    "source_diff_sha256": hashlib.sha256(
        subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
    ).hexdigest(),
    "files": [
        {
            "path": str(path),
            "bytes": path.stat().st_size,
            "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        }
        for path in files
    ],
    "library": str(library),
}
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
