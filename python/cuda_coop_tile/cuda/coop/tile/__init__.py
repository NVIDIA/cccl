# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative operations on logical Tiles inside ``cuda.tile`` kernels.

Use ``import cuda.coop.tile as coop`` in kernel modules. Most operations
accept inputs of compile-time one-dimensional shape ``(N,)`` and numeric
element types:
signed or unsigned 8-, 16-, 32-, or 64-bit integers, ``float16``, ``bfloat16``,
``float32``, or ``float64``. Individual operations narrow these choices.
The compiler and native provider select physical participation and scratch
storage. Supported input extents depend on the operation. Operations other
than reductions require a width of 32, 64, 128, or 256 to divide ``N``.
Exchange's explicit width defines its permutation. Reductions also support
multidimensional inputs
and explicit axes, including 1, 2, 4, 8 or 16 elements per result.

Operations return new Tile values and leave inputs unchanged. Reductions
remove reduced axes by default; ``keepdims=True`` retains them with extent
one, and ``broadcast=True`` broadcasts aggregates to the input shape. TopK
preserves capacity while defining only a leading prefix. See each function
for its result and boundary contracts.
Examples in the function docstrings assume the ``coop`` import above and
already loaded Tiles inside a kernel.

The optional ``methods`` argument is a compile-time mapping or flat tuple
of alternating names and values, such as ``("algorithm", "raking")``.
Prefer named keyword arguments. A non-``None`` constant keyword overrides
the corresponding method value; a runtime keyword cannot also appear in
``methods``. Unknown method keys are errors. Numeric runtime controls
convert to the required dtype; integer controls use signed 32-bit values,
except Run Length Decode offsets, which use the unsigned length dtype.
For integer data, constant seeds and padding keys must be integers, not
floating-point values or booleans. Invalid constant controls fail during
compilation; runtime controls must satisfy each family's documented bounds.

Importing this package also exposes the C++ configuration helpers without
loading the Python compiler. Accessing a primitive loads cuTile and registers
its lowering once. Install ``cuda-coop-tile`` with compatible ``cuda-tile`` and
``cuda-tile-preview`` packages
for Python kernel compilation.
"""

from __future__ import annotations

from collections.abc import Mapping
from threading import Lock

from ._cpp import (
    CppTileConfig,
    get_cmake_dir,
    get_cpp_config,
    get_cpp_include_path,
    get_cpp_interface_path,
    get_cpp_tile_library_path,
)

_API_SYMBOLS = (
    "exclusive_sum",
    "inclusive_sum",
    "scan",
    "exclusive_scan",
    "inclusive_scan",
    "reduce",
    "sum",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_sort_keys",
    "radix_sort_pairs",
    "histogram_256",
    "histogram",
    "run_length_decode",
    "run_length_decode_with_offsets",
    "discontinuity_flag_heads",
    "discontinuity_flag_tails",
    "discontinuity_flag_heads_and_tails",
    "adjacent_difference_subtract_left",
    "adjacent_difference_subtract_right",
    "exchange_striped_to_blocked",
    "exchange_blocked_to_striped",
    "shuffle_up",
    "shuffle_down",
    "shuffle_offset",
    "shuffle_rotate",
    "radix_rank_keys",
    "topk_max_keys",
    "topk_min_keys",
    "topk_max_pairs",
    "topk_min_pairs",
)

# The primitive tuple is shared with lazy loading; every entry is a string.
__all__ = [  # noqa: PLE0604
    *_API_SYMBOLS,
    "CppTileConfig",
    "get_cmake_dir",
    "get_cpp_config",
    "get_cpp_include_path",
    "get_cpp_interface_path",
    "get_cpp_tile_library_path",
    "register",
]

_loaded_api: Mapping[str, object] | None = None
_load_api_lock = Lock()


def _cuda_tile_import_failed(exc: ImportError) -> bool:
    name = getattr(exc, "name", None)
    return name == "cuda.tile" or bool(name and name.startswith("cuda.tile."))


def _missing_cuda_tile_error(exc: ImportError) -> ImportError:
    return ImportError(
        "cuda.coop.tile requires a compatible cuda-tile package for Python "
        "primitives. Install compatible cuda-tile and cuda-tile-preview "
        "packages with cuda-coop-tile. "
        "If cuda-tile is already installed, "
        "check that its compiler API is compatible with cuda-coop-tile."
    ).with_traceback(exc.__traceback__)


def _load_api() -> Mapping[str, object]:
    global _loaded_api
    if _loaded_api is not None:
        return _loaded_api

    with _load_api_lock:
        if _loaded_api is not None:
            return _loaded_api

        try:
            from . import _api
            from ._register import register
        except ImportError as exc:
            if _cuda_tile_import_failed(exc):
                raise _missing_cuda_tile_error(exc) from exc
            raise

        register()
        values: dict[str, object] = {
            name: getattr(_api, name) for name in _API_SYMBOLS
        }
        values["register"] = register
        _loaded_api = values
        return values


def __getattr__(name: str) -> object:
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

    value = _load_api()[name]
    globals()[name] = value
    return value


def __dir__() -> list[str]:
    module_attrs = set(globals())
    module_attrs.update(__all__)
    return sorted(module_attrs)
