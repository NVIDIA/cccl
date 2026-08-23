# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private block-scope compatibility primitives for the Numba backend."""

import importlib

__all__ = [
    "BlockLoadAlgorithm",
    "BlockStoreAlgorithm",
    "BlockScanAlgorithm",
    "BlockHistogramAlgorithm",
    "BlockExchangeType",
    "BlockAdjacentDifferenceType",
    "BlockShuffleType",
    "BlockDiscontinuityType",
    "adjacent_difference",
    "histogram",
    "run_length",
    "shuffle",
    "discontinuity",
    "exchange",
    "load",
    "reduce",
    "sum",
    "scan",
    "exclusive_sum",
    "inclusive_sum",
    "exclusive_scan",
    "inclusive_scan",
    "store",
    "make_adjacent_difference",
    "make_discontinuity",
    "make_exchange",
    "make_exclusive_scan",
    "make_exclusive_sum",
    "make_histogram",
    "make_inclusive_scan",
    "make_inclusive_sum",
    "make_load",
    "make_reduce",
    "make_run_length",
    "make_scan",
    "make_shuffle",
    "make_store",
    "make_sum",
]


def _factory_name(name):
    if name.startswith("make_"):
        return name[5:]
    return name


def __getattr__(name):
    if name in ("histogram", "make_histogram"):
        _block_histogram = importlib.import_module(f"{__name__}._block_histogram")
        value = getattr(_block_histogram, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("run_length", "make_run_length"):
        _block_run_length_decode = importlib.import_module(
            f"{__name__}._block_run_length_decode"
        )
        value = getattr(_block_run_length_decode, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("BlockExchangeType", "exchange", "make_exchange"):
        _block_exchange = importlib.import_module(f"{__name__}._block_exchange")
        value = getattr(_block_exchange, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "BlockAdjacentDifferenceType",
        "adjacent_difference",
        "make_adjacent_difference",
    ):
        _block_adjacent_difference = importlib.import_module(
            f"{__name__}._block_adjacent_difference"
        )
        value = getattr(_block_adjacent_difference, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("BlockShuffleType", "shuffle", "make_shuffle"):
        _block_shuffle = importlib.import_module(f"{__name__}._block_shuffle")
        value = getattr(_block_shuffle, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("BlockDiscontinuityType", "discontinuity", "make_discontinuity"):
        _block_discontinuity = importlib.import_module(
            f"{__name__}._block_discontinuity"
        )
        value = getattr(_block_discontinuity, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "BlockLoadAlgorithm",
        "BlockStoreAlgorithm",
        "BlockScanAlgorithm",
        "BlockHistogramAlgorithm",
    ):
        # Algorithm enums are intentionally runtime-free so callers can inspect
        # configuration choices without importing the Numba-CUDA-MLIR runtime.
        _enums = importlib.import_module(f"{__name__.rsplit('.', 1)[0]}._enums")
        value = getattr(_enums, name)
        globals()[name] = value
        return value
    if name in ("load", "store", "make_load", "make_store"):
        _block_load_store = importlib.import_module(f"{__name__}._block_load_store")
        value = getattr(_block_load_store, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("reduce", "sum", "make_reduce", "make_sum"):
        _block_reduce = importlib.import_module(f"{__name__}._block_reduce")
        value = getattr(_block_reduce, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "scan",
        "exclusive_sum",
        "inclusive_sum",
        "exclusive_scan",
        "inclusive_scan",
        "make_scan",
        "make_exclusive_sum",
        "make_inclusive_sum",
        "make_exclusive_scan",
        "make_inclusive_scan",
    ):
        _block_scan = importlib.import_module(f"{__name__}._block_scan")
        value = getattr(_block_scan, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "merge_sort_keys",
        "merge_sort_pairs",
        "make_merge_sort_keys",
        "make_merge_sort_pairs",
    ):
        _block_merge_sort = importlib.import_module(f"{__name__}._block_merge_sort")
        value = getattr(_block_merge_sort, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "radix_sort_keys",
        "radix_sort_keys_descending",
        "radix_sort_pairs",
        "radix_sort_pairs_descending",
        "make_radix_sort_keys",
        "make_radix_sort_keys_descending",
        "make_radix_sort_pairs",
        "make_radix_sort_pairs_descending",
    ):
        _block_radix_sort = importlib.import_module(f"{__name__}._block_radix_sort")
        value = getattr(_block_radix_sort, _factory_name(name))
        globals()[name] = value
        return value
    if name in ("radix_rank", "make_radix_rank"):
        _block_radix_rank = importlib.import_module(f"{__name__}._block_radix_rank")
        value = getattr(_block_radix_rank, _factory_name(name))
        globals()[name] = value
        return value
    if name in (
        "topk_max_keys",
        "topk_min_keys",
        "topk_max_pairs",
        "topk_min_pairs",
        "make_topk_max_keys",
        "make_topk_min_keys",
        "make_topk_max_pairs",
        "make_topk_min_pairs",
    ):
        _block_topk = importlib.import_module(f"{__name__}._block_topk")
        value = getattr(_block_topk, _factory_name(name))
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__():
    return sorted(__all__)
