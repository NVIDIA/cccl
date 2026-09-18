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

__all__ = [
    "ArgumentBinding",
    "BindingKind",
    "BlockLoadAlgorithm",
    "BlockLoadStoreAlgorithm",
    "BlockLoadStoreKind",
    "BlockLoadStoreSemantics",
    "BlockLoadStoreSpec",
    "BlockStoreAlgorithm",
    "binding",
    "make_block_load_spec",
    "make_block_load_store_semantics",
    "make_block_load_store_spec",
    "make_block_store_spec",
    "normalize_block_dim",
    "normalize_positive_int",
]
