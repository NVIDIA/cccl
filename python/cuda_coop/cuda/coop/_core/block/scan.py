# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Backend-neutral CUB BlockScan semantics."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm, AlgorithmSpec
from .._types import (
    Array,
    CxxFunction,
    CxxOperator,
    Dependency,
    Pointer,
    PythonOperator,
    Reference,
    TemplateParameter,
    TempStorageParameter,
)
from ..scan import ScanMode, ScanSemantics, ScanValueKind, make_scan_semantics
from ._common import normalize_block_dim


class BlockScanAlgorithm(str, Enum):
    """Normalized CUB ``BlockScanAlgorithm`` values used by core plans."""

    RAKING = "::cub::BLOCK_SCAN_RAKING"
    RAKING_MEMOIZE = "::cub::BLOCK_SCAN_RAKING_MEMOIZE"
    WARP_SCANS = "::cub::BLOCK_SCAN_WARP_SCANS"


def normalize_block_scan_algorithm(
    algorithm: str | BlockScanAlgorithm,
) -> BlockScanAlgorithm:
    """Normalize an internal or fully scoped CUB algorithm spelling."""

    if isinstance(algorithm, BlockScanAlgorithm):
        return algorithm
    for candidate in BlockScanAlgorithm:
        scoped = candidate.value.replace(
            "::cub::",
            "::cub::BlockScanAlgorithm::",
            1,
        )
        if algorithm in {candidate.name.lower(), candidate.value, scoped}:
            return candidate
    raise ValueError(f"unsupported CUB BlockScan algorithm {algorithm!r}")


@dataclass(frozen=True)
class BlockScanSpec:
    """Fully specialized CUB BlockScan call semantics."""

    specialization: AlgorithmSpec
    call: ScanSemantics
    block_dim: tuple[int, int, int]
    algorithm: BlockScanAlgorithm

    @property
    def mode(self) -> ScanMode:
        return self.call.mode

    @property
    def value_kind(self) -> ScanValueKind:
        return self.call.value_kind

    @property
    def items_per_thread(self) -> int:
        return self.call.items_per_thread

    @property
    def has_initial_value(self) -> bool:
        return self.call.initial_value is not None

    @property
    def has_block_aggregate(self) -> bool:
        return self.call.aggregate

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def _block_scan_parameters(call: ScanSemantics) -> tuple[Any, ...]:
    parameters: list[Any] = [TempStorageParameter()]
    if call.value_kind is ScanValueKind.ARRAY:
        parameters.extend(
            (
                Array(
                    Dependency("T"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="input",
                ),
                Array(
                    Dependency("T"),
                    Dependency("ITEMS_PER_THREAD"),
                    name="output",
                    is_output=True,
                    is_return=False,
                ),
            )
        )
    else:
        parameters.extend(
            (
                Reference(Dependency("T"), name="input"),
                Reference(
                    Dependency("T"),
                    name="output",
                    is_output=True,
                    is_return=True,
                ),
            )
        )
    if call.initial_value is not None:
        parameters.append(call.initial_value)
    if call.scan_operator is not None:
        parameters.append(call.scan_operator)
    if call.aggregate:
        parameters.append(
            Pointer(
                Dependency("T"),
                name="block_aggregate",
                is_output=True,
                is_return=False,
                is_array_pointer=True,
                deref_on_call=True,
            )
        )
    return tuple(parameters)


def make_block_scan_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    mode: str | ScanMode,
    algorithm: str | BlockScanAlgorithm,
    value_kind: str | ScanValueKind,
    scan_operator: CxxOperator | PythonOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    block_aggregate: bool = False,
) -> BlockScanSpec:
    """Build canonical BlockScan semantics from frontend-normalized inputs."""

    algorithm = normalize_block_scan_algorithm(algorithm)
    block_dim = normalize_block_dim(block_dim)
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    if algorithm is BlockScanAlgorithm.WARP_SCANS and block_threads % 32 != 0:
        raise ValueError(
            "BLOCK_SCAN_WARP_SCANS requires a block size that is a multiple of 32"
        )
    call = make_scan_semantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        scan_operator=scan_operator,
        initial_value=initial_value,
        aggregate=block_aggregate,
    )
    if call.initial_value is not None and call.scan_operator is None:
        raise ValueError("BlockScan sum overloads do not accept an initial value")

    cpp_prefix = "Exclusive" if call.mode is ScanMode.EXCLUSIVE else "Inclusive"
    method_name = f"{cpp_prefix}{'Sum' if call.scan_operator is None else 'Scan'}"
    template_arguments = {
        "T": dtype,
        "BLOCK_DIM_X": block_dim[0],
        "ALGORITHM": algorithm.value,
        "BLOCK_DIM_Y": block_dim[1],
        "BLOCK_DIM_Z": block_dim[2],
    }
    if call.value_kind is ScanValueKind.ARRAY:
        template_arguments["ITEMS_PER_THREAD"] = items_per_thread

    specialization = Algorithm(
        struct_name="BlockScan",
        method_name=method_name,
        c_name="block_scan",
        includes=("cub/block/block_scan.cuh",),
        template_parameters=(
            TemplateParameter("T"),
            TemplateParameter("BLOCK_DIM_X"),
            TemplateParameter("ALGORITHM"),
            TemplateParameter("BLOCK_DIM_Y"),
            TemplateParameter("BLOCK_DIM_Z"),
        ),
        parameters=(_block_scan_parameters(call),),
        fake_return=call.value_kind is ScanValueKind.SCALAR,
    ).specialize(
        template_arguments,
        metadata={
            "scope": "block",
            "primitive": "scan",
            "mode": call.mode,
            "value_kind": call.value_kind,
            "initial_value": call.initial_value is not None,
            "operator": (
                None
                if call.scan_operator is None
                else type(call.scan_operator).__qualname__
            ),
            "aggregate": call.aggregate,
            "aggregate_excludes_initial": call.aggregate,
        },
    )
    return BlockScanSpec(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
        algorithm=algorithm,
    )


__all__ = [
    "BlockScanAlgorithm",
    "BlockScanSpec",
    "make_block_scan_spec",
    "normalize_block_scan_algorithm",
]
