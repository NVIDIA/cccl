# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""CUTLASS group-first built-in Reduce and Sum entry points."""

from enum import Enum
from numbers import Integral

from cuda.coop._core import ArgumentBinding
from cuda.coop._core.thread_group import ThreadGroup

from ._compiler._launch import current_kernel_launch_facts
from ._group_load_store import _is_boolean
from ._thread_data import _coerce_thread_payload
from ._thread_group import _require_complete_warp_partition

_SCOPE = "cuda.coop.cutlass"
_ALGORITHMS = frozenset({"raking_commutative_only", "raking", "warp_reductions"})


def _classify_valid_items(value, *, primitive="reduce"):
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
    group, value, /, *, binary_op=None, broadcast=True, valid_items=None, algorithm=None
):
    """Reduce scalars or per-thread register payloads with a built-in operator.

    See :func:`cuda.coop.reduce` for the shared parameters, supported groups,
    and participation rules. The qualified API adds CuTe register payloads and
    the operator aliases below.

    Parameters
    ----------
    value : numeric scalar, ThreadData, CuTe register tensor, or TensorSSA
        Each member supplies one scalar or a fixed-size per-thread payload.
        Register tensors and ``TensorSSA`` values are converted with
        :meth:`cuda.coop.cutlass.ThreadData.from_payload`. Every payload element
        contributes to one scalar result. Inputs remain unchanged.
    binary_op : str or built-in alias, optional
        Compile-time operator, default sum. Supported strings are ``"sum"``,
        ``"multiplies"``, ``"min"``, ``"max"``, ``"bit_and"``, ``"bit_or"``,
        and ``"bit_xor"``. Also accepts the corresponding ``operator`` or
        NumPy aliases, such as ``operator.add`` and ``numpy.maximum``.
        Bitwise operators require an integer dtype. Custom device functions
        are not supported.

    Returns
    -------
    CuTe numeric scalar
        Reduced value with the input dtype, defined at every member when
        ``broadcast=True`` and only at group rank zero otherwise. A NumPy
        dtype selector still produces a CuTe scalar inside the kernel.

    Notes
    -----
    Full-group built-in reductions use CUDAX. A ``valid_items`` prefix or an
    explicit block ``algorithm`` selects CUB and requires ``broadcast=False``.
    A prefix counts contributing members, requires scalar input, and does not
    reduce the required participation. For a mapped group of warps, every
    parent block thread must reach the call, including threads excluded by
    a non-exhaustive mapping.

    See Also
    --------
    cuda.coop.reduce
        Shared reduction contract and executable examples.
    cuda.coop.cutlass.sum
        Sum with the same operand and result behavior.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for valid prefixes and explicit block algorithms.
    """

    from ._operators import normalize_operator

    if not isinstance(group, ThreadGroup):
        raise TypeError(f"{_SCOPE}.reduce group must be a ThreadGroup")
    if not isinstance(broadcast, bool):
        raise TypeError(f"{_SCOPE}.reduce broadcast must be a bool")
    op = normalize_operator(binary_op)
    algorithm = _normalize_algorithm(algorithm)
    valid_binding = _classify_valid_items(valid_items)
    value = _coerce_thread_payload(
        value,
        scope=_SCOPE,
        primitive_name="reduce",
        arg_name="value",
        common_root_payload_kind="scalar_or_thread_data",
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
        broadcast=broadcast,
        valid_items=valid_items,
        valid_items_binding=valid_binding,
        algorithm=algorithm,
    )


def sum(group, value, /, *, broadcast=True, valid_items=None, algorithm=None):
    """Sum scalars or per-thread register payloads with CUTLASS.

    See :func:`cuda.coop.sum` for the shared parameters, defaults, supported
    groups, and examples. The qualified operand forms and participation rules
    are those of :func:`cuda.coop.cutlass.reduce`. Register tensors and
    ``TensorSSA`` values contribute all their elements, and inputs remain
    unchanged.

    Returns
    -------
    CuTe numeric scalar
        Sum with the input dtype, defined at every member when
        ``broadcast=True`` and only at group rank zero otherwise.

    See Also
    --------
    cuda.coop.cutlass.reduce
        Built-in operators, register payloads, and result ownership.
    :cpp:struct:`cub::BlockReduce`, :cpp:struct:`cub::WarpReduce`
        C++ primitives used for valid prefixes and explicit block algorithms.
        Full-group built-in reductions use CUDAX.
    """
    return reduce(
        group, value, broadcast=broadcast, valid_items=valid_items, algorithm=algorithm
    )


__all__ = ["reduce", "sum"]
