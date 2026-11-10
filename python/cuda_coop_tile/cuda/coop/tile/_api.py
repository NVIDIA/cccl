# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler-recognized Python entry points for Tile cooperative operations.

``_register`` associates these function objects with ``_lowerings``. Their
Python bodies describe return arity; compiled kernels obtain their behavior
from the lowering and native provider. Keep signatures and public help in
sync with ``__init__.pyi`` so runtime inspection and IDE help agree.
"""

from __future__ import annotations

from cuda.tile._execution import stub
from cuda.tile._stub import Tile


@stub(host=False, tile=True, simt=False)
def exclusive_sum(
    value: Tile,
    /,
    *,
    prefix_value=None,
    algorithm=None,
    methods=None,
) -> Tile:
    """Compute the sum of preceding elements at each Tile position.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    prefix_value : int, float, Tile
        Starting sum, defaulting to zero. Accepts a constant, runtime scalar,
        or single-element Tile and converts it to the input dtype.
    algorithm : str
        Compile-time choice of ``"raking"``, ``"raking_memoize"``, or
        ``"warp_scans"``. Defaults to ``"raking"``.
    methods : collections.abc.Mapping or tuple
        Compile-time options for ``algorithm`` and ``prefix_value``.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position zero contains the
        prefix; position ``i`` adds input positions ``[0, i)`` to it.
        Accumulation stays in the input dtype and within this Tile.

    Examples
    --------
    For a loaded integer Tile ``counts`` in a ``cuda.tile`` kernel:

    .. code-block:: python

        offsets = coop.exclusive_sum(counts)
        shifted_offsets = coop.exclusive_sum(counts, prefix_value=10)

    See Also
    --------
    inclusive_sum, exclusive_scan
    """

    return value


@stub(host=False, tile=True, simt=False)
def inclusive_sum(
    value: Tile,
    /,
    *,
    prefix_value=None,
    algorithm=None,
    methods=None,
) -> Tile:
    """Compute the sum through each Tile position.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    prefix_value : int, float, Tile
        Seed added before the first input. Accepts a constant, runtime
        scalar, or single-element Tile, converted to the input dtype.
        Omit it for an unseeded inclusive sum.
    algorithm : str
        Compile-time choice of ``"raking"``, ``"raking_memoize"``, or
        ``"warp_scans"``. Defaults to ``"raking"``.
    methods : collections.abc.Mapping or tuple
        Compile-time options for ``algorithm`` and ``prefix_value``.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position ``i`` sums input
        positions ``[0, i]`` and the optional prefix. Accumulation does not
        widen the dtype or continue into another Tile.

    Examples
    --------
    In a ``cuda.tile`` kernel, an inclusive sum of a 0/1 head mask numbers
    the runs, starting at one:

    .. code-block:: python

        heads = coop.discontinuity_flag_heads(values)
        run_ids = coop.inclusive_sum(heads)

    See Also
    --------
    exclusive_sum, inclusive_scan
    """

    return value


@stub(host=False, tile=True, simt=False)
def scan(
    value: Tile,
    /,
    *,
    mode: str = "exclusive",
    scan_op: str | None = None,
    initial_value=None,
    prefix_value=None,
    algorithm=None,
    methods=None,
) -> Tile:
    """Compute exclusive or inclusive prefixes with a named operation.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    mode : str
        Compile-time ``"exclusive"`` (default) or ``"inclusive"``.
    scan_op : str
        Compile-time ``"+"`` (default) or ``"sum"`` for addition, ``"*"``,
        ``"min"``, ``"max"``,
        ``"&"``, ``"|"``, or ``"^"``. Bitwise operations require integer
        inputs. Python callbacks are unsupported.
    initial_value : int, float, Tile
        Exclusive seed, converted to the input dtype. Addition defaults
        to zero; other exclusive operations require a seed here or through
        ``prefix_value``. Must be ``None`` in inclusive mode.
    prefix_value : int, float, Tile
        Seed applied before the inputs. Accepts a constant, runtime scalar,
        or single-element Tile. Cannot accompany ``initial_value``.
    algorithm : str
        Compile-time ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``.
        Defaults to ``"raking"``.
    methods : collections.abc.Mapping or tuple
        Compile-time options for ``scan_op``, ``initial_value``,
        ``prefix_value``, and ``algorithm``. Pass ``mode`` directly.

    Returns
    -------
    Tile
        Prefixes with the input shape and dtype. Exclusive prefixes omit
        the current input; inclusive prefixes include it. Each call starts
        a new scan, and accumulation stays in the input dtype.

    Examples
    --------
    For a loaded numeric Tile ``values`` in a ``cuda.tile`` kernel:

    .. code-block:: python

        products = coop.scan(values, scan_op="*", initial_value=1)
        maxima = coop.scan(values, mode="inclusive", scan_op="max")

    See Also
    --------
    exclusive_scan, inclusive_scan
    """

    return value


@stub(host=False, tile=True, simt=False)
def exclusive_scan(
    value: Tile,
    /,
    *,
    scan_op: str | None = None,
    initial_value=None,
    prefix_value=None,
    algorithm=None,
    methods=None,
) -> Tile:
    """Apply an operation to all elements before each Tile position.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    scan_op : str
        Compile-time ``"+"`` (default) or ``"sum"`` for addition, ``"*"``,
        ``"min"``, ``"max"``,
        ``"&"``, ``"|"``, or ``"^"``. Bitwise operations require integers;
        Python callbacks are unsupported.
    initial_value : int, float, Tile
        Seed for position zero, converted to the input dtype. Accepts a
        constant, runtime scalar, or single-element Tile. Addition defaults
        to zero; other operations require this seed or ``prefix_value``.
    prefix_value : int, float, Tile
        Alternative seed with the same accepted forms as ``initial_value``.
        Supply at most one of these two arguments.
    algorithm : str
        Compile-time ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``.
        Defaults to ``"raking"``.
    methods : collections.abc.Mapping or tuple
        Compile-time options for ``scan_op``, ``initial_value``,
        ``prefix_value``, and ``algorithm``.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. The first element is the seed;
        subsequent elements combine it with all preceding inputs.
        Accumulation stays in the input dtype and within this Tile.

    Examples
    --------
    In a ``cuda.tile`` kernel, use one as the seed for exclusive products:

    .. code-block:: python

        products = coop.exclusive_scan(values, scan_op="*", initial_value=1)

    See Also
    --------
    exclusive_sum, inclusive_scan
    """

    return value


@stub(host=False, tile=True, simt=False)
def inclusive_scan(
    value: Tile,
    /,
    *,
    scan_op: str | None = None,
    initial_value=None,
    prefix_value=None,
    algorithm=None,
    methods=None,
) -> Tile:
    """Apply an operation through each Tile position.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    scan_op : str
        Compile-time ``"+"`` (default) or ``"sum"`` for addition, ``"*"``,
        ``"min"``, ``"max"``,
        ``"&"``, ``"|"``, or ``"^"``. Bitwise operations require integer
        inputs. Python callbacks are unsupported.
    initial_value : None
        Must be omitted or ``None``. Use ``prefix_value`` to seed an
        inclusive scan.
    prefix_value : int, float, Tile
        Seed combined with the first input before scanning the rest.
        Accepts a constant, runtime scalar, or single-element Tile,
        converted to the input dtype. Omit it to start with the first input.
    algorithm : str
        Compile-time ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``.
        Defaults to ``"raking"``.
    methods : collections.abc.Mapping or tuple
        Compile-time options for ``scan_op``, ``prefix_value``, and
        ``algorithm``; ``initial_value`` may only be ``None``.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position ``i`` combines inputs
        ``[0, i]`` with the optional prefix. Accumulation does not widen
        the dtype or cross this Tile's boundary.

    Examples
    --------
    For a loaded numeric Tile ``values`` in a ``cuda.tile`` kernel:

    .. code-block:: python

        running_maximum = coop.inclusive_scan(values, scan_op="max")

    See Also
    --------
    inclusive_sum, exclusive_scan
    """

    return value


@stub(host=False, tile=True, simt=False)
def reduce(
    value: Tile,
    /,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    *,
    binary_op: str | None = None,
    algorithm=None,
    methods=None,
    broadcast: bool = False,
) -> Tile:
    """Reduce Tile axes with a built-in CUB operation.

    Parameters
    ----------
    value : Tile
        Numeric input Tile. Boolean addition accumulates in ``int32``.
    axis : int or tuple of int, optional
        Compile-time axes to reduce. ``None`` reduces all elements; negative
        axes count from the end. An empty tuple leaves the values unchanged.
    keepdims : bool
        Retain reduced axes with extent one. Defaults to ``False``.
    binary_op : str
        Compile-time ``"+"`` (default) or ``"sum"`` for addition, ``"*"``,
        ``"min"``, ``"max"``,
        ``"&"``, ``"|"``, or ``"^"``. Bitwise operations require integers;
        Python callbacks are unsupported.
    algorithm : str
        Compile-time ``"raking_commutative_only"``, ``"raking"``, or
        ``"warp_reductions"``. The first requires a commutative operation.
        Defaults to ``"warp_reductions"``.
    methods : collections.abc.Mapping or tuple
        Compile-time options for ``binary_op`` and ``algorithm``.
    broadcast : bool
        Broadcast aggregates back to the input shape when ``True``.
        Defaults to ``False``.

    Returns
    -------
    Tile
        The reduced shape, or the retained dimensions when ``keepdims=True``.
        An all-element reduction returns a rank-zero Tile, not shape ``(1,)``.
        Accumulation and output preserve the input dtype except boolean sums,
        which use ``int32``.

    Notes
    -----
    For valid Tile shapes, the number of elements reduced into each result
    must be 1, 2, 4, 8, 16 or a positive multiple of 32.
    Independent rows are batched through native provider invocations.
    ``warp_reductions`` uses CUB WarpReduce or WarpReduceBatched; explicit
    raking algorithms use CUB BlockReduce. Floating-point evaluation order
    can differ from ``cuda.tile.sum``; bitwise equality is not guaranteed.
    Rounding-mode and flush-to-zero options are unsupported.
    Min/max preserve CUB comparison behavior, including operand-order-sensitive
    NaN and signed-zero ties; they do not implement ``cuda.tile.min``/``max``
    NaN controls.

    Examples
    --------
    Shift values using their reduced minimum in a ``cuda.tile`` kernel:

    .. code-block:: python

        minimum = coop.reduce(values, binary_op="min")
        shifted = values - minimum

    See Also
    --------
    sum
    """

    return value


@stub(host=False, tile=True, simt=False)
def sum(
    value: Tile,
    /,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    *,
    algorithm=None,
    methods=None,
    broadcast: bool = False,
) -> Tile:
    """Sum Tile axes with native CUB reductions.

    Parameters
    ----------
    value : Tile
        Numeric input Tile. Boolean addition accumulates in ``int32``.
    axis : int or tuple of int, optional
        Compile-time axes to reduce. ``None`` reduces all elements; negative
        axes count from the end. An empty tuple leaves the values unchanged.
    keepdims : bool
        Retain reduced axes with extent one. Defaults to ``False``.
    algorithm : str
        Compile-time ``"raking_commutative_only"``, ``"raking"``, or
        ``"warp_reductions"``. Defaults to ``"warp_reductions"``.
    methods : collections.abc.Mapping or tuple
        Compile-time ``algorithm`` option.
    broadcast : bool
        Broadcast aggregates back to the input shape when ``True``.
        Defaults to ``False``.

    Returns
    -------
    Tile
        The reduced shape, or the retained dimensions when ``keepdims=True``.
        An all-element reduction returns a rank-zero Tile, not shape ``(1,)``.
        Accumulation and output preserve the input dtype except boolean sums,
        which use ``int32``.

    Notes
    -----
    For valid Tile shapes, the number of elements reduced into each result
    must be 1, 2, 4, 8, 16 or a positive multiple of 32.
    Independent rows are batched through native provider invocations.
    ``warp_reductions`` uses CUB WarpReduce or WarpReduceBatched; explicit
    raking algorithms use CUB BlockReduce. Floating-point evaluation order
    can differ from ``cuda.tile.sum``; bitwise equality is not guaranteed.
    Rounding-mode and flush-to-zero options are unsupported.

    Examples
    --------
    For a floating-point Tile ``values`` with a nonzero sum in a
    ``cuda.tile`` kernel:

    .. code-block:: python

        normalized = values / coop.sum(values)

    See Also
    --------
    reduce, inclusive_sum
    """

    return value


@stub(host=False, tile=True, simt=False)
def exchange_striped_to_blocked(
    value: Tile, /, *, block_threads: int, methods=None
) -> Tile:
    """Gather striped items into contiguous logical-thread segments.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
    block_threads : int
        Required compile-time width ``B``: a multiple of 32 in
        ``[32, 1024]`` that divides ``N``. The kernel must admit this width.
        Changing it changes the permutation.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. With ``I = N // B``,
        ``output[t * I + j] = value[j * B + t]`` for ``0 <= t < B`` and
        ``0 <= j < I``. The input is unchanged.

    Examples
    --------
    In a ``cuda.tile`` kernel admitting 128 workers, permute 256 values
    and restore them using the same width:

    .. code-block:: python

        blocked = coop.exchange_striped_to_blocked(values, block_threads=128)
        restored = coop.exchange_blocked_to_striped(blocked, block_threads=128)

    See Also
    --------
    exchange_blocked_to_striped
    """

    return value


@stub(host=False, tile=True, simt=False)
def exchange_blocked_to_striped(
    value: Tile, /, *, block_threads: int, methods=None
) -> Tile:
    """Interleave contiguous logical-thread segments into striped order.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
    block_threads : int
        Required compile-time width ``B``: a multiple of 32 in
        ``[32, 1024]`` that divides ``N``. The kernel must admit this width.
        Use the same width as the forward exchange to invert it.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. With ``I = N // B``,
        ``output[j * B + t] = value[t * I + j]`` for ``0 <= t < B`` and
        ``0 <= j < I``. The input is unchanged. With ``I == 1``, the
        permutation is the identity.

    Examples
    --------
    In a ``cuda.tile`` kernel admitting 128 workers, interleave a
    256-element Tile:

    .. code-block:: python

        striped = coop.exchange_blocked_to_striped(values, block_threads=128)

    See Also
    --------
    exchange_striped_to_blocked
    """

    return value


__all__ = [
    "exchange_blocked_to_striped",
    "exchange_striped_to_blocked",
    "exclusive_scan",
    "exclusive_sum",
    "inclusive_scan",
    "inclusive_sum",
    "reduce",
    "scan",
    "sum",
]
