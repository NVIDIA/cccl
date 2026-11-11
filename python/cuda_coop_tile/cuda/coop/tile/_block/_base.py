# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class CoopPrimitive:
    name: str


@dataclass
class BlockPrimitive(CoopPrimitive):
    block_threads: int = 128
