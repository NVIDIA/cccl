# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockLoad and BlockStore descriptions."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._symbols import semantic_token
from ..thread_group import Dim3, normalize_thread_dim


class BlockLoadStoreKind(str, Enum):
    LOAD = "load"
    STORE = "store"


class BlockLoadStoreAlgorithm(str, Enum):
    DIRECT = "direct"


@dataclass(frozen=True, eq=False)
class BlockLoadStoreSpec:
    """Fully specialized direct CUB BlockLoad or BlockStore description."""

    kind: BlockLoadStoreKind
    dtype: Any
    block_dim: Dim3
    items_per_thread: int
    valid_items: bool = False
    oob_default: bool = False
    pointer_offset: bool = False
    algorithm: BlockLoadStoreAlgorithm = BlockLoadStoreAlgorithm.DIRECT

    def __post_init__(self) -> None:
        object.__setattr__(self, "kind", BlockLoadStoreKind(self.kind))
        object.__setattr__(self, "algorithm", BlockLoadStoreAlgorithm(self.algorithm))
        object.__setattr__(
            self,
            "block_dim",
            normalize_thread_dim(
                self.block_dim,
                scope="BlockLoadStoreSpec",
                label="block",
            ),
        )
        if (
            not isinstance(self.items_per_thread, int)
            or isinstance(self.items_per_thread, bool)
            or self.items_per_thread <= 0
        ):
            raise ValueError("items_per_thread must be a positive integer")
        if self.kind is BlockLoadStoreKind.STORE and self.oob_default:
            raise ValueError("oob_default is valid only for BlockLoad")
        if self.oob_default and not self.valid_items:
            raise ValueError("oob_default requires valid_items")

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return (
            f"block_{self.kind.value}",
            semantic_token(self.dtype),
            self.block_dim,
            self.items_per_thread,
            self.algorithm.value,
            self.valid_items,
            self.oob_default,
            self.pointer_offset,
        )

    def __eq__(self, other: object) -> bool:
        if not isinstance(other, BlockLoadStoreSpec):
            return NotImplemented
        return self.semantic_key == other.semantic_key

    def __hash__(self) -> int:
        return hash(self.semantic_key)


def make_block_load_store_spec(
    *,
    kind: str | BlockLoadStoreKind,
    dtype: Any,
    block_dim: Dim3 | int | tuple[int, ...] | list[int],
    items_per_thread: int,
    algorithm: str | BlockLoadStoreAlgorithm = BlockLoadStoreAlgorithm.DIRECT,
    valid_items: bool = False,
    oob_default: bool = False,
    include_pointer_offset: bool = False,
) -> BlockLoadStoreSpec:
    """Build one direct CUB BlockLoad or BlockStore description."""

    return BlockLoadStoreSpec(
        kind=BlockLoadStoreKind(kind),
        dtype=dtype,
        block_dim=normalize_thread_dim(
            block_dim,
            scope="make_block_load_store_spec",
            label="block",
        ),
        items_per_thread=items_per_thread,
        algorithm=BlockLoadStoreAlgorithm(algorithm),
        valid_items=valid_items,
        oob_default=oob_default,
        pointer_offset=include_pointer_offset,
    )


def make_block_load_spec(**kwargs: Any) -> BlockLoadStoreSpec:
    return make_block_load_store_spec(kind=BlockLoadStoreKind.LOAD, **kwargs)


def make_block_store_spec(**kwargs: Any) -> BlockLoadStoreSpec:
    return make_block_load_store_spec(kind=BlockLoadStoreKind.STORE, **kwargs)


__all__ = [
    "BlockLoadStoreAlgorithm",
    "BlockLoadStoreKind",
    "BlockLoadStoreSpec",
    "make_block_load_spec",
    "make_block_load_store_spec",
    "make_block_store_spec",
]
