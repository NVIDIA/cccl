# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Locate the CUDASTF C development headers and shared library.

These helpers let external C/CUDA projects compile and link against the same
STF C ABI that the Python bindings use. Importing this module is cheap: it does
*not* load the STF extension (``_stf_bindings_impl``) or preload CUDA libraries,
so it is safe to use from build scripts.

cuda-stf ships its own include root (see :func:`get_include_paths`) containing
the C STF header ``cccl/c/experimental/stf/stf.h`` and the cudax headers
``cuda/experimental/*.cuh``. The lower-level libcudacxx/CUB/Thrust headers those
cudax headers ``#include`` are *not* shipped here; they are owned by cuda-cccl.
When cuda-cccl is installed, :func:`get_include_paths` transparently adds its
include root so C++ compilation works; otherwise only the STF root is returned.
"""

from __future__ import annotations

import site
import sys
from dataclasses import dataclass
from functools import lru_cache
from importlib.resources import as_file, files
from pathlib import Path
from typing import Optional

# Shared library produced by the cccl.c.experimental.stf target (Linux-only).
_STF_LIBRARY_NAME = "libcccl.c.experimental.stf.so"
_CUDA_EXTRAS = ("cu12", "cu13")

# A header that only exists in the STF include root, used to validate a
# candidate directory actually contains the shipped headers.
_STF_PROBE_FILE = Path("cuda/experimental/places.cuh")


def iter_site_roots():
    """Yield unique candidate roots under which an installed ``cuda`` package
    may live.

    Scans ``sys.path`` plus the interpreter's site directories. The site
    directories are required for pip build isolation, which strips the venv
    site-packages from ``sys.path`` while the package remains installed there
    (``sys.prefix`` still points at the venv, so ``site.getsitepackages()``
    recovers it). ``getsitepackages`` is missing in some virtualenv setups, so
    it is probed defensively.
    """
    try:
        site_dirs = site.getsitepackages()
    except AttributeError:
        site_dirs = []
    try:
        site_dirs = [*site_dirs, site.getusersitepackages()]
    except AttributeError:
        pass

    seen: set[Path] = set()
    for sp in [*sys.path, *site_dirs]:
        root = Path(sp).resolve()
        if root in seen:
            continue
        seen.add(root)
        yield root


@dataclass
class IncludePaths:
    cuda: Optional[Path]
    libcudacxx: Optional[Path]
    cub: Optional[Path]
    thrust: Optional[Path]
    stf: Optional[Path]

    def as_tuple(self):
        # Note: higher-level ... lower-level order:
        return (self.stf, self.thrust, self.cub, self.libcudacxx, self.cuda)


@lru_cache()
def get_stf_include_dir() -> Path:
    """Return cuda-stf's own include root (cudax + C STF headers)."""
    candidate_roots = []
    with as_file(files("cuda.stf._experimental")) as f:
        candidate_roots.append(Path(f) / "include")

    # Editable installs and pip build isolation may place CMake-installed
    # headers outside the import package tree. Scan site roots as a fallback.
    for root in iter_site_roots():
        candidate_roots.append(root / "cuda" / "stf" / "_experimental" / "include")

    seen = set()
    for root in candidate_roots:
        key = str(root.resolve()) if root.exists() else str(root)
        if key in seen:
            continue
        seen.add(key)
        if (root / _STF_PROBE_FILE).exists():
            return root

    raise RuntimeError(
        "Unable to locate the CUDASTF include directory. "
        "Reinstall cuda-stf with a CUDA extra (e.g. `pip install cuda-stf[cu13]`)."
    )


def _cccl_base_include_root() -> Optional[Path]:
    """Best-effort libcudacxx/CUB/Thrust include root from cuda-cccl.

    cuda-cccl is an optional peer: it provides the lower-level headers the STF
    cudax headers ``#include``. Returns ``None`` when cuda-cccl is not
    installed, in which case only the STF headers are available.
    """
    try:
        from cuda.cccl.headers.include_paths import (  # noqa: PLC0415
            get_include_paths as _get_cccl_include_paths,
        )
    except Exception:
        return None
    try:
        return _get_cccl_include_paths().libcudacxx
    except Exception:
        return None


def _cuda_toolkit_include() -> Optional[Path]:
    try:
        from cuda.pathfinder import (  # noqa: PLC0415  # type: ignore[import-not-found]
            find_nvidia_header_directory,
        )
    except Exception:
        return None
    try:
        return find_nvidia_header_directory("cudart")
    except Exception:
        return None


def get_include_paths() -> IncludePaths:
    """Return the include paths needed to compile against the STF C/C++ API.

    The ``stf`` field points at cuda-stf's own include root (containing the C
    STF header ``cccl/c/experimental/stf/stf.h`` and the cudax headers
    ``cuda/experimental/*.cuh``). The ``libcudacxx``/``cub``/``thrust`` fields
    point at cuda-cccl's include root when it is installed (required to compile
    the cudax C++ headers); they are ``None`` otherwise.
    """
    stf_incl = get_stf_include_dir()
    cccl_incl = _cccl_base_include_root()
    return IncludePaths(
        cuda=_cuda_toolkit_include(),
        libcudacxx=cccl_incl,
        cub=cccl_incl,
        thrust=cccl_incl,
        stf=stf_incl,
    )


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
        "Reinstall cuda-stf with a CUDA extra (e.g. `pip install cuda-stf[cu13]`)."
    )


@lru_cache()
def get_library_path() -> Path:
    """Return the full path to the STF C shared library."""
    return get_library_dir() / _STF_LIBRARY_NAME


def _detect_preferred_extra() -> str | None:
    """Best-effort preferred CUDA extra from runtime bindings.

    This intentionally imports ``cuda.bindings`` lazily (through the local
    ``_cuda_version_utils`` helper) so importing this module stays lightweight
    in build-isolation environments where runtime bindings may be absent.
    """
    try:
        from ._cuda_version_utils import (  # noqa: PLC0415
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
