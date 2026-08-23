# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict Pyright fixture for explicit CUTLASS scoped load/store controls."""

# pyright: strict, reportPrivateUsage=none, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cuda.coop.cutlass as coop
    from cuda.coop.cutlass._dsl._launch import CutlassLaunchMetadata

    output = coop.ThreadData(2, int)
    metadata: CutlassLaunchMetadata = {"threads_per_block": 128}

    coop._block.load(
        object(),
        output,
        valid_items=127,
        oob_default=0,
        offset=1,
        payload=coop.Payload.PRIMS,
        launch_metadata=metadata,
    )
    coop._block.store(
        object(),
        output,
        valid_items=127,
        offset=1,
        launch_config=metadata,
    )
    coop._warp.load(
        object(),
        output,
        valid_items=31,
        threads_in_warp=32,
    )
    coop._warp.store(
        object(),
        output,
        valid_items=31,
    )

    block_load = coop._block.make_load(
        int,
        128,
        2,
        "direct",
        offset=1,
        launch_metadata=metadata,
    )
    block_store = coop._block.make_store(
        int,
        128,
        2,
        "direct",
        offset=1,
        launch_config=metadata,
    )
    warp_load = coop._warp.make_load(int, 2, 32, "direct", offset=1)
    warp_store = coop._warp.make_store(int, 2, 32, "direct", offset=1)

    block_load(object(), output, offset=2, payload=coop.Payload.PRIMS)
    block_store(object(), output, offset=2, payload=coop.Payload.PRIMS)
    warp_load(object(), output, offset=2, dtype=int)
    warp_store(object(), output, offset=2, dtype=int)
