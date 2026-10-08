# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Choose a CUB BlockScan signature and bind its template arguments.

Scalar and array scans use distinct input and output parameters. The backend
turns those descriptions into its result representation and allocates scratch.
This module selects the CUB method and argument order; it does not execute a
scan or determine the scratch layout.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any

from .._algorithm import Algorithm
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
from ..scan import ScanMode, ScanSemantics, ScanValueKind, make_scan_semantics
from ._common import normalize_block_dim


class BlockScanAlgorithm(str, Enum):
    """Name the CUB strategies available for a whole-block scan.

    ``RAKING`` combines shared partial reductions through a scan in one warp.
    ``RAKING_MEMOIZE`` keeps each raking segment in registers to reduce shared
    reads, at the cost of more registers. ``WARP_SCANS`` scans within each
    warp and combines the preceding warps' contributions. This core requires
    complete 32-thread warps for ``WARP_SCANS`` because CUB would silently
    substitute ``RAKING`` otherwise. Rejecting that request ensures the
    compiled algorithm is the one requested.
    """

    RAKING = "::cub::BLOCK_SCAN_RAKING"
    RAKING_MEMOIZE = "::cub::BLOCK_SCAN_RAKING_MEMOIZE"
    WARP_SCANS = "::cub::BLOCK_SCAN_WARP_SCANS"


def normalize_block_scan_algorithm(
    algorithm: str | BlockScanAlgorithm,
) -> BlockScanAlgorithm:
    """Resolve a short name or CUB-qualified spelling to one enum value.

    Accept an enum member, a lowercase member name such as ``"raking"``, or
    its CUB spelling with or without the ``BlockScanAlgorithm`` enum scope.
    Unknown spellings raise ``ValueError``. This helper does not strip
    whitespace or fold case. Frontends that accept user selectors normalize
    them first.
    """

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
class BlockScanSpecialization:
    """Keep a bound CUB call beside the Scan choices that produced it.

    Adapters consume ``specialization`` to generate the backend call. Planners
    can inspect the normalized shape and algorithm without decoding that
    call's template arguments or parameter list.

    Attributes
    ----------
    specialization : Algorithm
        Bound CUB method, template arguments, parameter order, and metadata.
    call : ScanSemantics
        Normalized shape, operator, seed, and aggregate request.
    block_dim : tuple of int
        Positive ``(x, y, z)`` block dimensions used by CUB's template.
    algorithm : BlockScanAlgorithm
        Canonical block implementation choice.
    """

    specialization: Algorithm
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
    def method_name(self) -> str:
        return self.specialization.method_name

    @property
    def semantic_key(self) -> tuple[Any, ...]:
        return self.specialization.semantic_key


def _block_scan_parameters(call: ScanSemantics) -> tuple[Any, ...]:
    """Build the CUB argument order with separate input and output storage.

    Both forms start with temporary storage. Scalar output is marked as the
    logical return value. Array output uses a separate buffer that the backend
    must provide. Append the optional seed, operator, prefix callback, and
    aggregate in CUB call order. A prefix callback computes a seed from the
    input aggregate; it cannot be combined with a seed or aggregate output.
    """

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
    return tuple(parameters)


def make_block_scan_specialization(
    *,
    dtype: Any,
    block_dim: tuple[int, int, int],
    items_per_thread: int,
    mode: str | ScanMode,
    algorithm: str | BlockScanAlgorithm,
    value_kind: str | ScanValueKind,
    scan_operator: CxxOperator | PythonOperator | None = None,
    initial_value: CxxFunction | Reference | None = None,
    prefix_operator: PythonOperator | StatefulOperator | None = None,
    block_aggregate: bool = False,
) -> BlockScanSpecialization:
    """Bind a block shape and Scan operation to a concrete CUB overload.

    An absent operator selects ``ExclusiveSum`` or ``InclusiveSum``. Supplying
    an operator selects the corresponding general Scan method. For arrays,
    ``ITEMS_PER_THREAD`` sizes the input and output array parameters. It is
    not a ``BlockScan`` class template parameter.

    Parameters
    ----------
    dtype : object
        Input and output dtype, forwarded to ``make_scan_semantics``.
    block_dim : tuple of int
        Positive ``(x, y, z)`` dimensions. Their product must be a multiple of
        32 when ``algorithm`` is ``WARP_SCANS``.
    items_per_thread : int
        Positive Python integer. Scalar form requires one item.
    mode : str or ScanMode
        ``"exclusive"`` or ``"inclusive"``.
    algorithm : str or BlockScanAlgorithm
        Block strategy accepted by ``normalize_block_scan_algorithm``.
    value_kind : str or ScanValueKind
        ``"scalar"`` or ``"array"``; arrays use blocked item order.
    scan_operator : CxxOperator or PythonOperator, optional
        Static operator descriptor. To seed addition, supply an explicit plus
        operator because the BlockScan sum overloads do not take a seed.
    initial_value : CxxFunction or Reference, optional
        Static expression or runtime scalar that seeds an exclusive scan.
        Its dtype must match the payload or refer to ``Dependency("T")``.
    prefix_operator : PythonOperator or StatefulOperator, optional
        Callback descriptor that computes a seed from the input aggregate.
        It cannot be combined with initial_value or block_aggregate.
    block_aggregate : bool, optional
        Request a scalar side output for all members, excluding the seed.

    Returns
    -------
    BlockScanSpecialization
        Bound algorithm and normalized call choices. Scalar calls mark their
        output reference as a return; array calls require an output buffer.

    Raises
    ------
    TypeError
        An operator, prefix, seed descriptor, or aggregate flag is invalid.
    ValueError
        Block dimensions, algorithm, or Scan shape are invalid, or a seed is
        supplied without an operator. See ``make_scan_semantics`` for shared
        shape, initial-value, and mutually exclusive callback checks.

    Notes
    -----
    This low-level builder can describe a custom exclusive scan with neither
    a seed nor a prefix callback. CUB leaves its first output undefined. Group
    planning requires one of those seed sources before exposing that form.
    """

    algorithm = normalize_block_scan_algorithm(algorithm)
    block_dim = normalize_block_dim(block_dim)
    block_threads = block_dim[0] * block_dim[1] * block_dim[2]
    if algorithm is BlockScanAlgorithm.WARP_SCANS and block_threads % 32 != 0:
        raise ValueError(
            "BLOCK_SCAN_WARP_SCANS requires a block size "
            "that is a multiple of 32"
        )
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
        raise ValueError(
            "BlockScan sum overloads do not accept an initial value"
        )

    cpp_prefix = "Exclusive" if call.mode is ScanMode.EXCLUSIVE else "Inclusive"
    method_name = (
        f"{cpp_prefix}{'Sum' if call.scan_operator is None else 'Scan'}"
    )
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
        template_arguments=template_arguments,
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
            "prefix_callback": (
                None
                if call.prefix_callback is None
                else type(call.prefix_callback).__qualname__
            ),
            "aggregate": call.aggregate,
            "aggregate_excludes_initial": call.aggregate,
        },
    )
    return BlockScanSpecialization(
        specialization=specialization,
        call=call,
        block_dim=block_dim,
        algorithm=algorithm,
    )


__all__ = [
    "BlockScanAlgorithm",
    "BlockScanSpecialization",
    "make_block_scan_specialization",
    "normalize_block_scan_algorithm",
]
