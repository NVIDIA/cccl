# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral exact CUDA launch facts."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from .thread_group import Dim3, normalize_thread_dim


def _normalize_optional_dim(value: Any, *, label: str) -> Dim3 | None:
    if value is None:
        return None
    return normalize_thread_dim(value, scope="LaunchFacts", label=label)


@dataclass(frozen=True)
class LaunchFactOrigin:
    """Diagnostic origin for one launch fact."""

    fact: str
    source: str
    detail: str | None = None
    verified: bool = False

    def __post_init__(self) -> None:
        if not self.fact:
            raise ValueError("LaunchFactOrigin fact cannot be empty")
        if not self.source:
            raise ValueError("LaunchFactOrigin source cannot be empty")
        if not isinstance(self.verified, bool):
            raise TypeError("LaunchFactOrigin verified must be a bool")


@dataclass(frozen=True)
class LaunchFacts:
    """Exact static launch knowledge supplied by a compiler backend."""

    exact_block_dim: Dim3 | int | tuple[int, ...] | list[int] | None = None
    provenance: tuple[LaunchFactOrigin, ...] | LaunchFactOrigin = field(
        default=(),
        compare=False,
        hash=False,
    )

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "exact_block_dim",
            _normalize_optional_dim(self.exact_block_dim, label="exact block"),
        )
        provenance = self.provenance
        if isinstance(provenance, LaunchFactOrigin):
            provenance = (provenance,)
        else:
            provenance = tuple(provenance)
        if any(not isinstance(item, LaunchFactOrigin) for item in provenance):
            raise TypeError("LaunchFacts provenance must contain LaunchFactOrigin")
        object.__setattr__(self, "provenance", provenance)

    @property
    def exact_block_threads(self) -> int | None:
        if self.exact_block_dim is None:
            return None
        x, y, z = self.exact_block_dim
        return x * y * z

    @property
    def semantic_key(self) -> tuple[Dim3 | None]:
        return (self.exact_block_dim,)

    def is_verified(self, fact: str) -> bool:
        return any(
            origin.fact == fact and origin.verified for origin in self.provenance
        )


__all__ = [
    "Dim3",
    "LaunchFactOrigin",
    "LaunchFacts",
]
