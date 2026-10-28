# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Validate Reduce calls for CuTe kernels and adapt register payloads.

Convert readable payloads, register tensors, and TensorSSA values to
ThreadData. The planner selects CUB block or warp reductions and records
root-only results and the scratch allocation policy.
"""

from __future__ import annotations

from collections.abc import Callable
from enum import Enum
from numbers import Integral
from typing import TypeVar

from cuda.coop.cutlass._planning import ArgumentBinding
from cuda.coop.cutlass._planning.thread_group import ThreadGroup
from cuda.coop.cutlass._typing import (
    CommonNumericScalar,
    CommonThreadDataLike,
    ReduceAlgorithm,
    ReduceOperator,
    TempStorageLike,
    ValidItems,
)

from ._compiler._launch import current_kernel_launch_facts
from ._group_load_store import _is_boolean
from ._thread_data import (
    CutlassTensorSample,
    CutlassTensorSSASample,
    _coerce_thread_payload,
)
from ._thread_group import _require_complete_warp_partition
from ._typing import BlockGroup, WarpGroup
from ._validation import _validate_temp_storage

_ItemT = TypeVar("_ItemT", bound=CommonNumericScalar)


_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset(
    {"raking_commutative_only", "raking", "warp_reductions"}
)


def _classify_valid_items(value, *, primitive="reduce"):
    """Separate omitted, static, and runtime counts for Reduce and Scan.

    Python and NumPy integral values bind a static count; CuTe integer values
    supply runtime operands. Booleans are rejected. ``primitive`` selects the
    diagnostic name. Group planning checks static bounds, and generated
    wrappers check runtime bounds; classification alone does not establish a
    valid count.
    """

    if value is None:
        return ArgumentBinding.omitted()
    if _is_boolean(value):
        raise TypeError(f"{_SCOPE}.{primitive} valid_items must be an integer")
    if isinstance(value, Integral):
        return ArgumentBinding.static(int(value))
    from cutlass.base_dsl.typing import Integer

    if isinstance(value, Integer):
        return ArgumentBinding.runtime()
    raise TypeError(f"{_SCOPE}.{primitive} valid_items must be an integer")


def _normalize_algorithm(algorithm):
    """Normalize a CUB block algorithm name.

    Strip whitespace, lowercase, and replace hyphens with underscores. Accept
    strings for the three deterministic strategies; Enum values and the
    nondeterministic warp-reduction strategy are rejected. Leave None so the
    shared planner can choose a route.
    """

    if algorithm is None:
        return None
    if not isinstance(algorithm, str) or isinstance(algorithm, Enum):
        raise TypeError(f"{_SCOPE}.reduce algorithm must be a string")
    token = algorithm.strip().lower().replace("-", "_")
    if token not in _ALGORITHMS:
        raise ValueError(
            f"{_SCOPE}.reduce algorithm must be one of: "
            + ", ".join(sorted(_ALGORITHMS))
        )
    return token


def reduce(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | _ItemT
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    binary_op: ReduceOperator
    | Callable[[object, object], object]
    | None = None,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Reduce scalars or per-thread register payloads with a built-in operator.

    Every member of the block or warp participates. Only group rank zero may
    read the result. CuTe register payloads contribute all their elements.

    Parameters
    ----------
    group : ThreadGroup
        Block, physical warp, or logical warp. Logical widths may be powers of
        two from 1 through 32, or widths from 17 through 31. For the latter,
        use ``group_by(width, exhaustive=False)`` and guard the call with
        ``group.is_member()``. Every member participates, including members
        excluded by ``valid_items``. The enclosing block must contain complete
        physical warps for warp reductions.
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Each member supplies one scalar or a fixed-size per-thread payload.
        Register tensors and ``TensorSSA`` values are converted with
        :meth:`cuda.coop.cutlass.ThreadData.from_payload`. Every payload element
        contributes to one scalar result. Inputs remain unchanged. Supported
        dtypes are signed and unsigned 8-, 16-, 32-, and 64-bit integers,
        ``Float32``, and ``Float64``.
    binary_op : str or built-in alias, optional
        Compile-time operator, default sum. Supported strings are ``"sum"``,
        ``"multiplies"``, ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"``,
        and ``"bit_xor"``. Also accepts the corresponding ``operator`` or
        NumPy aliases, such as ``operator.add`` and ``numpy.maximum``.
        Bitwise operators require an integer dtype. Custom device functions
        are not supported.

    valid_items : integer, optional
        Group-uniform count of contributing members, from one through the
        group size. Requires scalar input. ``None`` includes every member;
        empty reductions are unsupported.
    algorithm : str, optional
        Compile-time block algorithm: ``"raking_commutative_only"``,
        ``"raking"``, or ``"warp_reductions"``. ``None`` selects
        ``"warp_reductions"``. Warp groups require ``None``.
    temp_storage : TempStorage, optional
        Explicit block scratch descriptor. Size and alignment are inferred
        from its uses unless requested. Omission enables automatic allocation
        and reuse barriers. With ``auto_sync=False``, synchronize the block
        before reuse. Warp groups use automatic storage and require ``None``.

    Returns
    -------
    CuTe numeric scalar
        Reduced value with the input dtype, defined only at group rank zero.
        A NumPy dtype selector still produces a CuTe scalar inside the kernel.

    Notes
    -----
    Every reduction uses CUB. Blocks and warps accept scalars or per-thread
    payloads. A ``valid_items`` prefix counts contributing members, requires
    scalar input, and does not reduce required participation. Block reductions
    accept ``temp_storage`` with the same
    sharing, capacity, alignment, and synchronization policies as Load/Store.
    Floating-point results can differ from a sequential fold because reduction
    regroups operations.

    See Also
    --------
    cuda.coop.cutlass.sum
        Sum with the same operand and result behavior.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for block and warp reductions.

    Examples
    --------
    Compute a sum and a maximum over the same tile. Only the block root
    reads either result.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_reduce_examples.py
        :language: python
        :start-after: # qualified-reduce-example-begin
        :end-before: # qualified-reduce-example-end
        :dedent: 4
    """

    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.reduce group must be a ThreadGroup")
    if group.kind not in {"block", "warp", "threads_within_warp"}:
        raise NotImplementedError(
            f"{_SCOPE}.reduce requires a block, physical warp, "
            "or logical warp group"
        )
    if temp_storage is not None:
        if group.kind != "block":
            raise NotImplementedError(
                f"{_SCOPE}.reduce explicit TempStorage is supported only "
                "for block groups"
            )
        _validate_temp_storage("reduce", temp_storage)
    op = normalize_operator(binary_op)
    algorithm = _normalize_algorithm(algorithm)
    valid_binding = _classify_valid_items(valid_items)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="reduce",
        arg_name="value",
    )
    launch = current_kernel_launch_facts()
    _require_complete_warp_partition(
        group, feature="reduce", exact_block_dim=launch.exact_block_dim
    )
    from ._lowering._reduce import provider_reduce

    return provider_reduce(
        group=group,
        launch=launch,
        value=value,
        op=op,
        temp_storage=temp_storage,
        valid_items=valid_items,
        valid_items_binding=valid_binding,
        algorithm=algorithm,
    )


def sum(
    group: BlockGroup | WarpGroup,
    value: CommonThreadDataLike[_ItemT]
    | _ItemT
    | CutlassTensorSample
    | CutlassTensorSSASample,
    /,
    *,
    valid_items: ValidItems | None = None,
    algorithm: ReduceAlgorithm | None = None,
    temp_storage: TempStorageLike | None = None,
) -> _ItemT:
    """Sum scalars or per-thread register payloads with CUTLASS.

    Equivalent to :func:`cuda.coop.cutlass.reduce` with ``binary_op="sum"``.
    Its group, payload, valid-prefix, algorithm, and scratch contracts apply.
    Every group member participates; only rank zero may read the result.

    Parameters
    ----------
    group : ThreadGroup
        Block, physical warp, or logical warp supported by ``reduce``.
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Each member's contribution. Every payload item contributes; the input
        remains unchanged.
    valid_items : integer, optional
        Uniform count of contributing members, for scalar inputs only.
    algorithm : str, optional
        Block algorithm described by ``reduce``; omit for warp groups.
    temp_storage : TempStorage, optional
        Explicit block scratch descriptor; omit for warp groups. Omission
        enables automatic scratch and reuse synchronization.

    Returns
    -------
    CuTe numeric scalar
        Sum with the input dtype, defined only at group rank zero.

    See Also
    --------
    cuda.coop.cutlass.reduce
        Built-in operators, register payloads, and result ownership.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for block and warp reductions.

    Examples
    --------
    Write a tile sum and maximum from the block root.

    The launcher accepts device pointers and a compile-time
    ``items_per_thread`` value.

    .. literalinclude::
        ../../../python/cuda_coop_cutlass/tests/runtime/test_qualified_reduce_examples.py
        :language: python
        :start-after: # qualified-reduce-example-begin
        :end-before: # qualified-reduce-example-end
        :dedent: 4
    """
    return reduce(
        group,
        value,
        temp_storage=temp_storage,
        valid_items=valid_items,
        algorithm=algorithm,
    )


__all__ = [
    "_classify_valid_items",
    "reduce",
    "sum",
]
