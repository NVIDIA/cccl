# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockRowReduce semantics."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._types import Dependency, Reference, TemplateParameter, TempStorageParameter
from ._common import normalize_positive_int

BLOCK_ROW_REDUCE_INCLUDES = ("cub/block/block_row_reduce.cuh",)
_WARP_THREADS = 32
_MAX_BLOCK_THREADS = 1024

_TEMPLATE_PARAMETERS = (
    TemplateParameter("T"),
    TemplateParameter("ROWS_PER_BLOCK"),
    TemplateParameter("WARPS_PER_ROW"),
)


@dataclass(frozen=True)
class BlockRowReduceGeometry:
    """Static row partition and its required CUDA block width."""

    rows_per_block: int
    warps_per_row: int

    @property
    def logical_warps(self) -> int:
        return self.rows_per_block * self.warps_per_row

    @property
    def block_threads(self) -> int:
        return self.logical_warps * _WARP_THREADS

    def validate_block_threads(self, block_threads: int) -> None:
        """Require the CUDA block width implied by this row partition."""

        block_threads = normalize_positive_int("block_threads", block_threads)
        if block_threads != self.block_threads:
            raise ValueError(
                f"block has {block_threads} threads; expected exactly "
                f"{self.block_threads} from rows_per_block={self.rows_per_block} "
                f"and warps_per_row={self.warps_per_row}"
            )


@dataclass(frozen=True)
class BlockRowReduceSpec:
    """Specialized CUB row-sum contract.

    The class is supplied by newer CUB header sets and is not present in every
    toolkit. Core describes its compile-time semantics; a backend remains
    responsible for checking header availability before compilation.
    """

    specialization: AlgorithmSpec
    geometry: BlockRowReduceGeometry

    @property
    def rows_per_block(self) -> int:
        return self.geometry.rows_per_block

    @property
    def warps_per_row(self) -> int:
        return self.geometry.warps_per_row

    @property
    def logical_warps(self) -> int:
        return self.geometry.logical_warps

    @property
    def block_threads(self) -> int:
        return self.geometry.block_threads

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def normalize_block_row_reduce_geometry(
    *,
    rows_per_block: int,
    warps_per_row: int,
) -> BlockRowReduceGeometry:
    """Normalize the static row geometry required by CUB's row reducer."""

    rows_per_block = normalize_positive_int("rows_per_block", rows_per_block)
    warps_per_row = normalize_positive_int("warps_per_row", warps_per_row)
    # One warp broadcasts the per-warp partials for a logical row, so the
    # number of participating warps in a row cannot exceed its 32 lanes. Keep
    # this check separate for a precise per-row diagnostic; the aggregate row
    # shape is independently limited by CUDA's maximum block width below.
    if warps_per_row > _WARP_THREADS:
        raise ValueError(
            f"warps_per_row must be <= {_WARP_THREADS} for CUB BlockRowReduce"
        )
    geometry = BlockRowReduceGeometry(rows_per_block, warps_per_row)
    if geometry.block_threads > _MAX_BLOCK_THREADS:
        raise ValueError(
            "rows_per_block * warps_per_row must fit in one CUDA thread block"
        )
    return geometry


def make_block_row_reduce_spec(
    *,
    dtype: Any,
    rows_per_block: int,
    warps_per_row: int,
) -> BlockRowReduceSpec:
    """Build the BlockRowReduceWarpBroadcast sum specialization."""

    if dtype is None:
        raise ValueError("dtype must be provided")
    geometry = normalize_block_row_reduce_geometry(
        rows_per_block=rows_per_block,
        warps_per_row=warps_per_row,
    )
    specialization = Algorithm(
        struct_name="BlockRowReduceWarpBroadcast",
        method_name="Sum",
        c_name="block_row_reduce",
        includes=BLOCK_ROW_REDUCE_INCLUDES,
        template_parameters=_TEMPLATE_PARAMETERS,
        parameters=(
            (
                TempStorageParameter(),
                Reference(Dependency("T"), name="value"),
                Reference(
                    Dependency("T"),
                    name="output",
                    is_output=True,
                    is_return=True,
                ),
            ),
        ),
    ).specialize(
        {
            "T": dtype,
            "ROWS_PER_BLOCK": geometry.rows_per_block,
            "WARPS_PER_ROW": geometry.warps_per_row,
        },
        metadata={
            "scope": "block",
            "primitive": "row_reduce",
            "operation": "sum",
            "rows_per_block": geometry.rows_per_block,
            "warps_per_row": geometry.warps_per_row,
        },
    )
    return BlockRowReduceSpec(
        specialization=specialization,
        geometry=geometry,
    )
