# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Block-scoped cooperative primitive semantic builders."""

from .._bindings import ArgumentBinding, BindingKind, binding
from ._common import normalize_block_dim, normalize_positive_int
from .exchange import (
    BlockExchangeMode,
    BlockExchangeSemantics,
    BlockExchangeSpec,
    BlockExchangeValueForm,
    make_block_exchange_semantics,
    make_block_exchange_spec,
)
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
from .shuffle import (
    BlockShuffleMode,
    BlockShuffleSemantics,
    BlockShuffleSpec,
    BlockShuffleValueKind,
    make_block_shuffle_semantics,
    make_block_shuffle_spec,
)

__all__ = [
    "ArgumentBinding",
    "BindingKind",
    "BlockExchangeMode",
    "BlockExchangeSemantics",
    "BlockExchangeSpec",
    "BlockExchangeValueForm",
    "BlockLoadAlgorithm",
    "BlockLoadStoreAlgorithm",
    "BlockLoadStoreKind",
    "BlockLoadStoreSemantics",
    "BlockLoadStoreSpec",
    "BlockShuffleMode",
    "BlockShuffleSemantics",
    "BlockShuffleSpec",
    "BlockShuffleValueKind",
    "BlockStoreAlgorithm",
    "binding",
    "make_block_exchange_semantics",
    "make_block_exchange_spec",
    "make_block_load_spec",
    "make_block_load_store_semantics",
    "make_block_load_store_spec",
    "make_block_shuffle_semantics",
    "make_block_shuffle_spec",
    "make_block_store_spec",
    "normalize_block_dim",
    "normalize_positive_int",
]
