# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from typing import Any

from cuda import coop


def partial_tile_copy(source: Any, destination: Any) -> None:
    block = coop.this_block()
    items = coop.ThreadData(2)
    loaded = coop.load(
        block,
        source,
        items,
        valid_items=63,
        oob_default=0,
        offset=1,
    )
    coop.store(
        block,
        destination,
        loaded,
        valid_items=63,
        offset=1,
    )
