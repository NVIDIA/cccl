# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import hashlib
import importlib
import os
import subprocess
import sys
from pathlib import Path
from typing import Protocol, cast

import torch


class Consumer(Protocol):
    __file__: str

    def normalize(
        self, input: torch.Tensor, first_thread: bool = False, broadcast: bool = True
    ) -> torch.Tensor: ...


def load_consumer(build: Path, variant: str) -> Consumer:
    sys.path.insert(0, str(build.resolve()))
    return cast(Consumer, importlib.import_module(f"cccl_{variant}_h20"))


def digest(tensor: torch.Tensor) -> str:
    return hashlib.sha256(
        tensor.detach().contiguous().view(torch.uint8).cpu().numpy().tobytes()
    ).hexdigest()


def provenance(source: Path, consumer: Consumer) -> dict[str, object]:
    library = Path(consumer.__file__)
    return {
        "pid": os.getpid(),
        "source_sha": subprocess.check_output(
            ["git", "-C", str(source), "rev-parse", "HEAD"], text=True
        ).strip(),
        "source_diff_sha256": hashlib.sha256(
            subprocess.check_output(["git", "-C", str(source), "diff", "HEAD"])
        ).hexdigest(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "gpu": subprocess.check_output(
            [
                "nvidia-smi",
                "--query-gpu=index,uuid,name,driver_version",
                "--format=csv,noheader",
            ],
            text=True,
        ).strip(),
        "visible_device": os.environ["CUDA_VISIBLE_DEVICES"],
        "library": str(library),
        "library_sha256": hashlib.sha256(library.read_bytes()).hexdigest(),
    }
