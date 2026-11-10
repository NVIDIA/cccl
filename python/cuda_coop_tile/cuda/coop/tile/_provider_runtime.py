# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Find the CCCL and CUDA headers used by Python-triggered provider builds.

The lowering serializes these include paths into the child process's
environment. Native C++ consumers also have their own include resolution;
this module does not load the provider or compile device code.
"""

from __future__ import annotations

import os
from pathlib import Path

from ._headers import resolve_include_paths


def _include_paths(*, required_headers: tuple[str, ...] = ()) -> list[str]:
    """Resolve one CCCL header source and append available CUDA headers.

    Explicit CCCL roots take precedence. Source imports use their checkout;
    installed imports use their own header bundle, even inside a checkout.
    Invalid roots and missing required headers raise ``RuntimeError`` without
    falling back to a different CCCL source.
    """

    paths = resolve_include_paths(
        start=Path(__file__),
        configured_roots=(
            os.environ.get(name)
            for name in ("CUDA_COOP_CCCL_ROOT", "CCCL_ROOT", "CCCL_DIR")
        ),
        required_headers=("cub/version.cuh", *required_headers),
    )
    return [str(path) for path in (*paths.cccl, *paths.cuda)]
