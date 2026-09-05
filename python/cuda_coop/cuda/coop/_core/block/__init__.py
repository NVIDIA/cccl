# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block-scoped cooperative primitive semantic builders."""

from .._bindings import ArgumentBinding, BindingKind, binding
from ._common import normalize_block_dim, normalize_positive_int
from .load_store import (
    BlockLoadAlgorithm,
    BlockLoadStoreAlgorithm,
    BlockLoadStoreKind,
    BlockLoadStoreSemantics,
    BlockLoadStoreSpec,
    BlockStoreAlgorithm,
    make_block_load_spec,
    make_block_load_store_semantics,
    make_block_load_store_spec,
    make_block_store_spec,
)
from .reduce import (
    BlockReduceAlgorithm,
    BlockReduceOperation,
    BlockReduceSemantics,
    BlockReduceSpec,
    BlockReduceValueKind,
    make_block_reduce_semantics,
    make_block_reduce_spec,
    normalize_block_reduce_algorithm,
)
from .scan import (
    BlockScanAlgorithm,
    BlockScanSpec,
    ScanMode,
    ScanSemantics,
    ScanValueKind,
    make_block_scan_spec,
    make_scan_semantics,
    normalize_block_scan_algorithm,
)

__all__ = [
    "ArgumentBinding",
    "BindingKind",
    "BlockLoadAlgorithm",
    "BlockLoadStoreAlgorithm",
    "BlockLoadStoreKind",
    "BlockLoadStoreSemantics",
    "BlockLoadStoreSpec",
    "BlockReduceAlgorithm",
    "BlockReduceOperation",
    "BlockReduceSemantics",
    "BlockReduceSpec",
    "BlockReduceValueKind",
    "BlockScanAlgorithm",
    "BlockScanSpec",
    "BlockStoreAlgorithm",
    "ScanMode",
    "ScanSemantics",
    "ScanValueKind",
    "binding",
    "make_block_load_spec",
    "make_block_load_store_semantics",
    "make_block_load_store_spec",
    "make_block_reduce_semantics",
    "make_block_reduce_spec",
    "make_block_scan_spec",
    "make_block_store_spec",
    "make_scan_semantics",
    "normalize_block_dim",
    "normalize_block_reduce_algorithm",
    "normalize_block_scan_algorithm",
    "normalize_positive_int",
]
