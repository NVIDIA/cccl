# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Locate the CUDASTF C development headers and shared library.

These helpers let external C/CUDA projects compile and link against the same
STF C ABI that the Python bindings use. Importing this module is cheap: it does
*not* load the STF extension (``_stf_bindings_impl``) or preload CUDA libraries,
so it is safe to use from build scripts.

The shipped include root (see :func:`get_include_paths`) contains both the C STF
header ``cccl/c/experimental/stf/stf.h`` and the cudax headers
``cuda/experimental/*.cuh``, alongside libcudacxx/CUB/Thrust.
"""

from __future__ import annotations

from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path

from cuda.cccl.headers import get_include_paths as _get_cccl_include_paths
from cuda.cccl.headers.include_paths import iter_site_roots

# Shared library produced by the cccl.c.experimental.stf target (Linux-only).
_STF_LIBRARY_NAME = "libcccl.c.experimental.stf.so"
_CUDA_EXTRAS = ("cu12", "cu13")


def get_include_paths():
    """Return the CCCL include paths needed to compile against the STF C API.

    The returned :class:`~cuda.cccl.headers.include_paths.IncludePaths` exposes
    an include root that contains the C STF header
    (``cccl/c/experimental/stf/stf.h``) and the cudax headers
    (``cuda/experimental/*.cuh``), in addition to libcudacxx/CUB/Thrust.
    """
    return _get_cccl_include_paths(probe_file="cuda/experimental/places.cuh")


@lru_cache()
def get_library_dir() -> Path:
    """Return the directory containing the STF C shared library."""
    preferred_extra = _detect_preferred_extra()

    extras = list(_CUDA_EXTRAS)
    if preferred_extra in extras:
        extras.remove(preferred_extra)
        extras.insert(0, preferred_extra)

    candidate_roots = []
    with as_file(files("cuda.stf._experimental")) as f:
        candidate_roots.append(Path(f))

    # Editable installs and pip build isolation may place compiled artifacts
    # outside the import package tree. Scan site roots as a fallback.
    for root in iter_site_roots():
        candidate_roots.append(root / "cuda" / "stf" / "_experimental")

    seen = set()
    for base in candidate_roots:
        for extra in extras:
            lib_dir = base / extra / "cccl"
            key = str(lib_dir.resolve()) if lib_dir.exists() else str(lib_dir)
            if key in seen:
                continue
            seen.add(key)
            if (lib_dir / _STF_LIBRARY_NAME).exists():
                return lib_dir

    raise RuntimeError(
        f"Unable to locate the CUDASTF library '{_STF_LIBRARY_NAME}'. "
        "Searched for cu12/cu13 layouts under the installed package and site roots. "
        "Reinstall cuda-cccl with a CUDA extra (e.g. `pip install cuda-cccl[cu13]`)."
    )


@lru_cache()
def get_library_path() -> Path:
    """Return the full path to the STF C shared library."""
    return get_library_dir() / _STF_LIBRARY_NAME


def _detect_preferred_extra() -> str | None:
    """Best-effort preferred CUDA extra from runtime bindings.

    This intentionally imports ``cuda.bindings`` lazily (through
    ``cuda.cccl._cuda_version_utils``) so importing this module stays lightweight
    in build-isolation environments where runtime bindings may be absent.
    """
    try:
        from cuda.cccl._cuda_version_utils import (
            detect_cuda_version,
            get_recommended_extra,
        )
    except Exception:
        return None

    try:
        extra = get_recommended_extra(detect_cuda_version())
    except Exception:
        return None
    return extra if extra in _CUDA_EXTRAS else None
