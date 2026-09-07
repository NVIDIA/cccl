# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Strict Pyright fixture proving CUTLASS scoped splats are not public API."""

# pyright: strict, reportPrivateUsage=none, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    import cuda.coop.cutlass as coop

    output = coop.ThreadData(2, int)

    coop._block.load(object(), output, 2)  # pyright: ignore[reportCallIssue]
    coop._block.load(object(), output, invented=True)  # pyright: ignore[reportCallIssue]
    coop._block.store(object(), output, 2)  # pyright: ignore[reportCallIssue]
    coop._block.store(object(), output, invented=True)  # pyright: ignore[reportCallIssue]
    coop._warp.load(object(), output, 2)  # pyright: ignore[reportCallIssue]
    coop._warp.load(object(), output, invented=True)  # pyright: ignore[reportCallIssue]
    coop._warp.load(object(), output, launch_metadata={})  # pyright: ignore[reportCallIssue]
    coop._warp.store(object(), output, 2)  # pyright: ignore[reportCallIssue]
    coop._warp.store(object(), output, invented=True)  # pyright: ignore[reportCallIssue]
    coop._warp.store(object(), output, launch_meta={})  # pyright: ignore[reportCallIssue]

    coop._block.make_load(None, None, 1, "direct", "extra")  # pyright: ignore[reportCallIssue]
    coop._block.make_load(invented=True)  # pyright: ignore[reportCallIssue]
    coop._block.make_store(None, None, 1, "direct", "extra")  # pyright: ignore[reportCallIssue]
    coop._block.make_store(invented=True)  # pyright: ignore[reportCallIssue]
    coop._warp.make_load(None, 1, 32, "direct", None, None, "extra")  # pyright: ignore[reportCallIssue]
    coop._warp.make_load(invented=True)  # pyright: ignore[reportCallIssue]
    coop._warp.make_load(launch={})  # pyright: ignore[reportCallIssue]
    coop._warp.make_store(None, 1, 32, "direct", None, "extra")  # pyright: ignore[reportCallIssue]
    coop._warp.make_store(invented=True)  # pyright: ignore[reportCallIssue]
    coop._warp.make_store(launch_config={})  # pyright: ignore[reportCallIssue]

    block_load = coop._block.make_load(items_per_thread=2)
    block_store = coop._block.make_store(items_per_thread=2)
    warp_load = coop._warp.make_load(items_per_thread=2)
    warp_store = coop._warp.make_store(items_per_thread=2)
    block_load(object(), output, 2)  # pyright: ignore[reportCallIssue]
    block_load(object(), invented=True)  # pyright: ignore[reportCallIssue]
    block_store(object(), output, 2)  # pyright: ignore[reportCallIssue]
    block_store(object(), output, invented=True)  # pyright: ignore[reportCallIssue]
    warp_load(object(), output, 2)  # pyright: ignore[reportCallIssue]
    warp_load(object(), invented=True)  # pyright: ignore[reportCallIssue]
    warp_load(object(), output, launch_config={})  # pyright: ignore[reportCallIssue]
    warp_store(object(), output, 2)  # pyright: ignore[reportCallIssue]
    warp_store(object(), output, invented=True)  # pyright: ignore[reportCallIssue]
    warp_store(object(), output, launch_metadata={})  # pyright: ignore[reportCallIssue]
