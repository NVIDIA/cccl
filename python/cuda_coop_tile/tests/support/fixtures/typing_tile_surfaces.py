# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

# pyright: strict, reportUnnecessaryTypeIgnoreComment=error

from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    # The pinned preview supplies inline types but no py.typed marker.
    from cuda.tile import Tile  # pyright: ignore[reportMissingTypeStubs]

    import cuda.coop.tile as tile_coop

    tile_value = cast(Tile, object())
    tile_reduce: Tile = tile_coop.reduce(tile_value)
    tile_scan: Tile = tile_coop.scan(tile_value, mode="inclusive")
    tile_discontinuity: tuple[Tile, Tile] = (
        tile_coop.discontinuity_flag_heads_and_tails(tile_value)
    )
    tile_decoded: Tile = tile_coop.run_length_decode(
        tile_value, tile_value, decoded_items=256
    )
    tile_decoded_offsets: tuple[Tile, Tile] = (
        tile_coop.run_length_decode_with_offsets(
            tile_value, tile_value, decoded_items=128, decoded_window_offset=3
        )
    )
