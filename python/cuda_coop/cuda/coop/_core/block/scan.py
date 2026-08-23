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
    StatefulOperator,
    TemplateParameter,
    TempStorageParameter,
)
from ..scan import (
    ScanMode,
    ScanSemantics,
    ScanValueKind,
    make_scan_semantics,
)


class BlockScanAlgorithm(str, Enum):
    """Public CUB ``BlockScanAlgorithm`` enumerators."""

    RAKING = "::cub::BLOCK_SCAN_RAKING"
    RAKING_MEMOIZE = "::cub::BLOCK_SCAN_RAKING_MEMOIZE"
    WARP_SCANS = "::cub::BLOCK_SCAN_WARP_SCANS"


def normalize_block_scan_algorithm(
    algorithm: str | BlockScanAlgorithm,
) -> BlockScanAlgorithm:
    if isinstance(algorithm, BlockScanAlgorithm):
        return algorithm
    for candidate in BlockScanAlgorithm:
        scoped = candidate.value.replace(
            "::cub::",
            "::cub::BlockScanAlgorithm::",
            1,
        )
        if algorithm in {candidate.value, scoped}:
            return candidate
    raise ValueError(f"unsupported CUB BlockScan algorithm {algorithm!r}")


@dataclass(frozen=True)
class BlockScanSpec:
    """Fully specialized BlockScan call semantics."""

    specialization: AlgorithmSpec
    call: ScanSemantics
    mode: ScanMode
    value_kind: ScanValueKind
    block_dim: tuple[int, int, int]
    items_per_thread: int
    algorithm: BlockScanAlgorithm
    has_initial_value: bool
    has_prefix_callback: bool
    has_block_aggregate: bool

    @property
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def make_block_scan_spec(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    mode: str | ScanMode,
    algorithm: str | BlockScanAlgorithm,
    value_kind: str | ScanValueKind,
    scan_operator: CxxOperator | PythonOperator | StatefulOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    prefix_operator: PythonOperator | StatefulOperator | None = None,
    block_aggregate: bool = False,
) -> BlockScanSpec:
    """Build canonical BlockScan semantics from frontend-normalized inputs."""

    algorithm = normalize_block_scan_algorithm(algorithm)
    block_dim = tuple(block_dim)
    if len(block_dim) != 3 or any(dim < 1 for dim in block_dim):
        raise ValueError("block_dim must contain three positive dimensions")
    call = make_scan_semantics(
        dtype=dtype,
        mode=mode,
        value_kind=value_kind,
        items_per_thread=items_per_thread,
        scan_operator=scan_operator,
        initial_value=initial_value,
        aggregate=block_aggregate,
        prefix_callback=prefix_operator,
    )
    if call.initial_value is not None and call.scan_operator is None:
        raise ValueError("BlockScan sum overloads do not accept an initial value")
    if call.initial_value is not None and call.prefix_callback is not None:
        raise ValueError(
            "BlockScan initial value and prefix callback are mutually exclusive"
        )
    if (
        call.mode is ScanMode.INCLUSIVE
        and call.value_kind is ScanValueKind.SCALAR
        and call.initial_value is not None
    ):
        raise ValueError(
            "scalar CUB BlockScan InclusiveScan has no initial-value overload"
        )

    cpp_prefix = "Exclusive" if call.mode is ScanMode.EXCLUSIVE else "Inclusive"
    method_name = f"{cpp_prefix}{'Sum' if call.scan_operator is None else 'Scan'}"
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
    if call.prefix_callback is not None:
        parameters.append(call.prefix_callback)
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

    algorithm_spec = Algorithm(
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
        parameters=(tuple(parameters),),
        fake_return=call.value_kind is ScanValueKind.SCALAR,
    ).specialize(
        {
            "T": dtype,
            "BLOCK_DIM_X": block_dim[0],
            "ALGORITHM": algorithm.value,
            "BLOCK_DIM_Y": block_dim[1],
            "BLOCK_DIM_Z": block_dim[2],
            **(
                {"ITEMS_PER_THREAD": items_per_thread}
                if call.value_kind is ScanValueKind.ARRAY
                else {}
            ),
        },
        metadata={
            "scope": "block",
            "primitive": "scan",
            "mode": call.mode,
            "value_kind": call.value_kind,
            "has_initial_value": call.initial_value is not None,
            "scan_operator": (
                None
                if call.scan_operator is None
                else type(call.scan_operator).__qualname__
            ),
            "prefix_operator": (
                None
                if call.prefix_callback is None
                else type(call.prefix_callback).__qualname__
            ),
            "block_aggregate": call.aggregate,
        },
    )
    return BlockScanSpec(
        specialization=algorithm_spec,
        call=call,
        mode=call.mode,
        value_kind=call.value_kind,
        block_dim=block_dim,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        has_initial_value=call.initial_value is not None,
        has_prefix_callback=call.prefix_callback is not None,
        has_block_aggregate=call.aggregate,
    )
