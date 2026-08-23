# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private warp-scope compatibility primitives for the Numba backend."""

import importlib

__all__ = [
    "reduce",
    "sum",
    "max",
    "min",
    "load",
    "store",
    "warp_reduce",
    "warp_sum",
    "warp_max",
    "warp_min",
    "warp_load",
    "warp_store",
    "make_load",
    "make_max",
    "make_min",
    "make_reduce",
    "make_store",
    "make_sum",
]


def _factory_name(name):
    if name.startswith("make_"):
        return name[5:]
    return name


def __getattr__(name):
    if name in (
        "reduce",
        "sum",
        "max",
        "min",
        "warp_reduce",
        "warp_sum",
        "warp_max",
        "warp_min",
        "make_reduce",
        "make_sum",
        "make_max",
        "make_min",
    ):
        _warp_reduce = importlib.import_module(f"{__name__}._warp_reduce")
        value = getattr(_warp_reduce, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "exclusive_sum",
        "inclusive_sum",
        "exclusive_scan",
        "inclusive_scan",
        "warp_exclusive_sum",
        "warp_inclusive_sum",
        "warp_exclusive_scan",
        "warp_inclusive_scan",
        "make_exclusive_sum",
        "make_inclusive_sum",
        "make_exclusive_scan",
        "make_inclusive_scan",
    ):
        _warp_scan = importlib.import_module(f"{__name__}._warp_scan")
        value = getattr(_warp_scan, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("load", "store", "warp_load", "warp_store", "make_load", "make_store"):
        _warp_load_store = importlib.import_module(f"{__name__}._warp_load_store")
        value = getattr(_warp_load_store, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("WarpExchangeType", "exchange", "warp_exchange", "make_exchange"):
        _warp_exchange = importlib.import_module(f"{__name__}._warp_exchange")
        value = getattr(_warp_exchange, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "merge_sort_keys",
        "merge_sort_pairs",
        "warp_merge_sort_keys",
        "warp_merge_sort_pairs",
        "make_merge_sort_keys",
        "make_merge_sort_pairs",
    ):
        _warp_merge_sort = importlib.import_module(f"{__name__}._warp_merge_sort")
        value = getattr(_warp_merge_sort, _factory_name(name))
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
