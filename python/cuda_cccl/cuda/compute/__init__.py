# Copyright (c) 2024, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations


# When built against the v2 (HostJIT) backend, the JIT loads Clang's CUDA
# headers and our cuda_minimal stubs from paths that don't exist on the
# user's machine. The wheel bundles both under cuda/cccl/headers/{clang,…};
# point hostjit at them via the env vars its detectDefaultConfig() reads.
# Only sets vars that aren't already configured by the user, and skips
# silently if the bundled directories are absent (e.g. v1 builds).
def _configure_hostjit_paths() -> None:
    import os
    from pathlib import Path

    try:
        from ._build_info import USING_V2  # type: ignore[import-not-found]
    except ImportError:
        return
    if not USING_V2:
        return

    # Probe for actual file presence, not just directory existence: editable
    # (`pip install -e`) installs leave behind empty placeholder dirs in the
    # source tree (with just `__pycache__`), so `is_dir()` succeeds but the
    # bundled headers are absent. In that case, leave the env vars unset and
    # let the C library use its build-time CLANG_HEADERS_DIR / HOSTJIT_INCLUDE_DIR
    # macros (pointing at the LLVM source tree under the CMake build dir).
    headers_dir = Path(__file__).resolve().parent.parent / "cccl" / "headers"
    clang_dir = headers_dir / "clang"
    if (
        clang_dir / "__clang_cuda_math_forward_declares.h"
    ).is_file() and not os.environ.get("HOSTJIT_CLANG_PATH"):
        os.environ["HOSTJIT_CLANG_PATH"] = str(clang_dir)
    if (
        headers_dir / "hostjit" / "cuda_minimal" / "__clang_cuda_runtime_wrapper.h"
    ).is_file() and not os.environ.get("HOSTJIT_INCLUDE_PATH"):
        os.environ["HOSTJIT_INCLUDE_PATH"] = str(headers_dir)


_configure_hostjit_paths()

from ._bindings import _BINDINGS_AVAILABLE  # type: ignore[attr-defined]  # noqa: E402

if not _BINDINGS_AVAILABLE:
    __all__ = ["_BINDINGS_AVAILABLE"]

    def __getattr__(name):
        raise AttributeError(
            f"Cannot access 'cuda.compute.{name}' because CUDA bindings are not available."
            "This typically means you're running on a CPU-only machine without CUDA drivers installed."
        )
else:
    from ._caching import clear_all_caches
    from .algorithms import (
        DoubleBuffer,
        SortOrder,
        binary_transform,
        exclusive_scan,
        histogram_even,
        inclusive_scan,
        lower_bound,
        make_binary_transform,
        make_exclusive_scan,
        make_histogram_even,
        make_inclusive_scan,
        make_lower_bound,
        make_merge_sort,
        make_radix_sort,
        make_reduce_into,
        make_segmented_reduce,
        make_segmented_sort,
        make_select,
        make_three_way_partition,
        make_unary_transform,
        make_unique_by_key,
        make_upper_bound,
        merge_sort,
        radix_sort,
        reduce_into,
        segmented_reduce,
        segmented_sort,
        select,
        three_way_partition,
        unary_transform,
        unique_by_key,
        upper_bound,
    )
    from .determinism import Determinism
    from .iterators import (
        CacheModifiedInputIterator,
        ConstantIterator,
        CountingIterator,
        DiscardIterator,
        PermutationIterator,
        ReverseIterator,
        ShuffleIterator,
        TransformIterator,
        TransformOutputIterator,
        ZipIterator,
    )
    from .op import OpKind
    from .struct import gpu_struct

    __all__ = [
        "_BINDINGS_AVAILABLE",
        "binary_transform",
        "clear_all_caches",
        "CacheModifiedInputIterator",
        "ConstantIterator",
        "CountingIterator",
        "DiscardIterator",
        "DoubleBuffer",
        "exclusive_scan",
        "gpu_struct",
        "histogram_even",
        "inclusive_scan",
        "lower_bound",
        "make_binary_transform",
        "make_exclusive_scan",
        "make_select",
        "make_histogram_even",
        "make_inclusive_scan",
        "make_lower_bound",
        "make_merge_sort",
        "make_radix_sort",
        "make_reduce_into",
        "make_segmented_reduce",
        "make_segmented_sort",
        "make_three_way_partition",
        "make_unary_transform",
        "make_unique_by_key",
        "make_upper_bound",
        "merge_sort",
        "OpKind",
        "Determinism",
        "PermutationIterator",
        "radix_sort",
        "reduce_into",
        "ReverseIterator",
        "ShuffleIterator",
        "segmented_reduce",
        "segmented_sort",
        "select",
        "SortOrder",
        "TransformIterator",
        "TransformOutputIterator",
        "three_way_partition",
        "unary_transform",
        "unique_by_key",
        "upper_bound",
        "ZipIterator",
    ]
