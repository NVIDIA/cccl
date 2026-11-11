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
def merge_sort_keys(
    keys: Tile,
    /,
    *,
    descending: bool | None = None,
    valid_items=None,
    oob_default=None,
    methods=None,
) -> Tile:
    """Sort a Tile's keys by numeric comparison.

    Parameters
    ----------
    keys : Tile
        Numeric keys of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    descending : bool
        Compile-time sort direction. ``None`` or ``False`` selects ascending
        order; ``True`` selects descending order.
    valid_items : int or Tile
        Meaningful input prefix length in ``[0, N]``. Accepts a constant,
        runtime scalar, or single-element Tile. Supply with ``oob_default``.
    oob_default : int, float, Tile
        Key used to replace positions beyond ``valid_items`` before sorting.
        Accepts a constant, runtime scalar, or single-element Tile.
        Integer keys require an integer constant; floating-point keys accept
        numeric constants. Runtime values are cast to the key dtype.
        Supply with ``valid_items``.
    methods : collections.abc.Mapping or tuple
        Compile-time ``descending``, ``valid_items``, and ``oob_default``
        options.

    Returns
    -------
    Tile
        Sorted keys with the input shape and dtype. The input is unchanged.
        Equal keys have no stability guarantee; NaNs have no total order.

    Notes
    -----
    To place all valid keys first, choose a padding key strictly greater
    than them for ascending order or strictly smaller for descending order.
    Otherwise padding may interleave with valid keys. Output capacity stays
    ``N``, including when ``valid_items`` is zero.

    Examples
    --------
    In a ``cuda.tile`` kernel with 256 ``int32`` keys, of which the first
    193 are strictly below the padding key:

    .. code-block:: python

        ordered = coop.merge_sort_keys(
            keys, valid_items=193, oob_default=2147483647
        )

    See Also
    --------
    merge_sort_pairs, radix_sort_keys
    """

    return keys


@stub(host=False, tile=True, simt=False)
def merge_sort_pairs(
    keys: Tile,
    values: Tile,
    /,
    *,
    descending: bool | None = None,
    valid_items=None,
    oob_default=None,
    methods=None,
) -> tuple[Tile, Tile]:
    """Sort keys while carrying their associated payloads.

    Parameters
    ----------
    keys, values : Tile
        Numeric Tiles of matching shape ``(N,)``. Their dtypes may differ.
        ``N`` must be a positive multiple of 32.
    descending : bool
        Compile-time sort direction. ``None`` or ``False`` selects ascending
        order; ``True`` selects descending order.
    valid_items : int or Tile
        Meaningful input prefix length in ``[0, N]``. Accepts a constant,
        runtime scalar, or single-element Tile. Supply with ``oob_default``.
    oob_default : int, float, Tile
        Padding key for positions beyond ``valid_items``. Accepts a constant,
        runtime scalar, or single-element Tile. Integer keys require an
        integer constant; floating-point keys accept numeric constants.
        Runtime values are cast to the key dtype. Padded payloads are zeroed
        before sorting.
    methods : collections.abc.Mapping or tuple
        Compile-time ``descending``, ``valid_items``, and ``oob_default``
        options.

    Returns
    -------
    tuple[Tile, Tile]
        ``(sorted_keys, sorted_values)`` with the respective input dtypes
        and shape ``(N,)``. Each value follows its key; inputs are unchanged.

    Notes
    -----
    Equal keys have no stability guarantee, and NaNs have no total order.
    For partial sorting, use a strict high sentinel in ascending order or
    a strict low sentinel in descending order. This places the valid pairs
    first and the zeroed padding payloads last. A sentinel equal to a valid
    key can interleave padded and valid payloads.

    Examples
    --------
    For loaded key and payload Tiles in a ``cuda.tile`` kernel:

    .. code-block:: python

        ordered_keys, ordered_ids = coop.merge_sort_pairs(keys, ids)

    See Also
    --------
    merge_sort_keys, radix_sort_pairs
    """

    return keys, values


@stub(host=False, tile=True, simt=False)
def radix_sort_keys(
    keys: Tile,
    /,
    *,
    begin_bit=None,
    end_bit=None,
    descending: bool | None = None,
    radix_bits: int | None = None,
    memoize_outer_scan: bool | None = None,
    inner_scan_algorithm: str | None = None,
    smem_config: str | int | None = None,
    methods=None,
) -> Tile:
    """Stably sort keys by a selected interval of radix-ordering bits.

    Parameters
    ----------
    keys : Tile
        Numeric keys of shape ``(N,)``, with ``N <= 65535``.
        ``N`` must be a positive multiple of 32.
    begin_bit, end_bit : int or Tile
        Half-open interval, defaulting to zero and the key bit width.
        Accepts constants, runtime scalars, or single-element Tiles.
        Require ``0 <= begin_bit <= end_bit <= key_bits``.
    descending : bool
        Compile-time direction; ``None`` or ``False`` selects ascending.
    radix_bits : int
        Compile-time bits per pass in ``[1, 8]``, defaulting to 4.
        Scratch storage grows exponentially with this value. Configurations
        within this range can fail compilation depending on the worker count
        and target resources.
    memoize_outer_scan : bool
        Compile-time CUB outer-scan memoization, defaulting to ``True``.
    inner_scan_algorithm : str
        Compile-time ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``
        (default).
    smem_config : str or int
        Compile-time ``"four_byte"`` (default) or ``"eight_byte"`` bank
        configuration. Aliases are ``4``, ``8``, ``"4"``, ``"8"``, and
        ``"default"`` (four-byte banks).
    methods : collections.abc.Mapping or tuple
        Compile-time values for the seven options above.

    Returns
    -------
    Tile
        Sorted keys with the input shape and dtype. Equal selected digits
        retain input order. The input is unchanged.

    Notes
    -----
    Signed and floating-point keys undergo CUB's radix ordering transform
    before extracting the bit interval. Partial sorting therefore uses
    transformed bits. Use unsigned keys for direct bit-field sorting.
    An empty interval makes all keys equivalent. ``radix_bits`` limits one
    pass, not the total interval. NaNs follow their transformed bit patterns.

    Examples
    --------
    Sort an unsigned integer Tile by its low byte in a ``cuda.tile`` kernel:

    .. code-block:: python

        ordered = coop.radix_sort_keys(keys, begin_bit=0, end_bit=8)

    See Also
    --------
    radix_sort_pairs, radix_rank_keys
    """

    return keys


@stub(host=False, tile=True, simt=False)
def radix_sort_pairs(
    keys: Tile,
    values: Tile,
    /,
    *,
    begin_bit=None,
    end_bit=None,
    descending: bool | None = None,
    radix_bits: int | None = None,
    memoize_outer_scan: bool | None = None,
    inner_scan_algorithm: str | None = None,
    smem_config: str | int | None = None,
    methods=None,
) -> tuple[Tile, Tile]:
    """Stably sort key-value pairs by radix-ordering bits.

    Parameters
    ----------
    keys, values : Tile
        Numeric Tiles of matching shape ``(N,)``, with ``N <= 65535``.
        Their dtypes may differ.
        ``N`` must be a positive multiple of 32.
    begin_bit, end_bit : int or Tile
        Half-open key-bit interval, defaulting to zero and the key width.
        Accepts constants, runtime scalars, or single-element Tiles.
        Require ``0 <= begin_bit <= end_bit <= key_bits``.
    descending : bool
        Compile-time direction; ``None`` or ``False`` selects ascending.
    radix_bits : int
        Compile-time bits per pass in ``[1, 8]``, defaulting to 4.
        Scratch storage grows exponentially with this value. Configurations
        within this range can fail compilation depending on the worker count
        and target resources.
    memoize_outer_scan : bool
        Compile-time CUB outer-scan memoization, defaulting to ``True``.
    inner_scan_algorithm : str
        Compile-time ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``
        (default).
    smem_config : str or int
        Compile-time ``"four_byte"`` (default) or ``"eight_byte"`` bank
        configuration. Aliases are ``4``, ``8``, ``"4"``, ``"8"``, and
        ``"default"`` (four-byte banks).
    methods : collections.abc.Mapping or tuple
        Compile-time values for the seven options above.

    Returns
    -------
    tuple[Tile, Tile]
        ``(sorted_keys, sorted_values)`` with the respective input dtypes
        and shape ``(N,)``. Equal selected digits retain their input order
        and payload association. Inputs are unchanged.

    Notes
    -----
    Bit bounds select from CUB's transformed representation for signed and
    floating-point keys; unsigned keys expose their raw bits. An empty
    interval leaves pairs in input order. NaNs follow their transformed
    patterns. ``radix_bits`` controls each pass, not the total interval.

    Examples
    --------
    Sort a loaded key Tile and its payload in a ``cuda.tile`` kernel:

    .. code-block:: python

        ordered_keys, ordered_ids = coop.radix_sort_pairs(keys, ids)

    See Also
    --------
    radix_sort_keys, radix_rank_keys
    """

    return keys, values


@stub(host=False, tile=True, simt=False)
def topk_max_keys(
    keys: Tile,
    /,
    *,
    k,
    valid_items=None,
    begin_bit=None,
    end_bit=None,
    methods=None,
) -> Tile:
    """Select the ``k`` largest keys from a Tile.

    Parameters
    ----------
    keys : Tile
        Numeric keys of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    k : int or Tile
        Number requested, with ``0 <= k <= N``. Accepts a constant,
        runtime integer scalar, or rank-zero Tile. A shape-one Tile is
        unsupported.
    valid_items : int or Tile
        Input prefix length, defaulting to ``N``, in ``[0, N]``.
        Positions beyond it are excluded. Scalar restrictions match ``k``.
    begin_bit, end_bit : int or Tile
        Half-open interval in the radix ordering representation, defaulting
        to zero and the key width. Require
        ``0 <= begin_bit < end_bit <= key_bits``. Scalar restrictions match
        ``k``. Signed and floating-point keys use CUB's ordering transform.
    methods : collections.abc.Mapping or tuple
        Compile-time ``begin_bit`` and ``end_bit`` options only.

    Returns
    -------
    Tile
        Selected keys with the input dtype and capacity ``N``. Only the
        first ``min(k, valid_items)`` positions are defined. The input is
        unchanged.

    Notes
    -----
    Selected keys are not guaranteed to be sorted. Ties may select different
    input items, so tied payload identity is unspecified. The remaining
    positions are undefined. Zero counts produce no defined output items.
    A partial bit interval selects by that digit rather than the full value.

    Examples
    --------
    For 256 loaded items in a ``cuda.tile`` kernel, select from the first
    193 and consume only the first 16 output positions:

    .. code-block:: python

        best_keys = coop.topk_max_keys(keys, k=16, valid_items=193)

    See Also
    --------
    topk_min_keys, topk_max_pairs
    """

    return keys


@stub(host=False, tile=True, simt=False)
def topk_min_keys(
    keys: Tile,
    /,
    *,
    k,
    valid_items=None,
    begin_bit=None,
    end_bit=None,
    methods=None,
) -> Tile:
    """Select the ``k`` smallest keys from a Tile.

    Parameters
    ----------
    keys : Tile
        Numeric keys of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    k : int or Tile
        Number requested, with ``0 <= k <= N``. Accepts a constant,
        runtime integer scalar, or rank-zero Tile. A shape-one Tile is
        unsupported.
    valid_items : int or Tile
        Input prefix length, defaulting to ``N``, in ``[0, N]``.
        Positions beyond it are excluded. Scalar restrictions match ``k``.
    begin_bit, end_bit : int or Tile
        Half-open interval in the radix ordering representation, defaulting
        to zero and the key width. Require
        ``0 <= begin_bit < end_bit <= key_bits``. Scalar restrictions match
        ``k``. Signed and floating-point keys use CUB's ordering transform.
    methods : collections.abc.Mapping or tuple
        Compile-time ``begin_bit`` and ``end_bit`` options only.

    Returns
    -------
    Tile
        Selected keys with the input dtype and capacity ``N``. Only the
        first ``min(k, valid_items)`` positions are defined. The input is
        unchanged.

    Notes
    -----
    Selected keys are not guaranteed to be sorted. Ties may select different
    input items, so tied payload identity is unspecified. The remaining
    positions are undefined. Zero counts produce no defined output items.
    A partial bit interval selects by that digit rather than the full value.

    Examples
    --------
    For 256 loaded items in a ``cuda.tile`` kernel, select from the first
    193 and consume only the first 16 output positions:

    .. code-block:: python

        best_keys = coop.topk_min_keys(keys, k=16, valid_items=193)

    See Also
    --------
    topk_max_keys, topk_min_pairs
    """

    return keys


@stub(host=False, tile=True, simt=False)
def topk_max_pairs(
    keys: Tile,
    values: Tile,
    /,
    *,
    k,
    valid_items=None,
    begin_bit=None,
    end_bit=None,
    methods=None,
) -> tuple[Tile, Tile]:
    """Select the ``k`` largest keys and their payloads from a Tile.

    Parameters
    ----------
    keys, values : Tile
        Numeric Tiles of matching shape ``(N,)``. Their dtypes may differ.
        ``N`` must be a positive multiple of 32.
    k : int or Tile
        Number requested, with ``0 <= k <= N``. Accepts a constant,
        runtime integer scalar, or rank-zero Tile. A shape-one Tile is
        unsupported.
    valid_items : int or Tile
        Input prefix length, defaulting to ``N``, in ``[0, N]``.
        Positions beyond it are excluded. Scalar restrictions match ``k``.
    begin_bit, end_bit : int or Tile
        Half-open interval in the radix ordering representation, defaulting
        to zero and the key width. Require
        ``0 <= begin_bit < end_bit <= key_bits``. Scalar restrictions match
        ``k``. Signed and floating-point keys use CUB's ordering transform.
    methods : collections.abc.Mapping or tuple
        Compile-time ``begin_bit`` and ``end_bit`` options only.

    Returns
    -------
    tuple[Tile, Tile]
        ``(selected_keys, selected_values)`` with the respective input dtypes
        and capacity ``N``. Only the first ``min(k, valid_items)`` positions
        are defined.
        Values remain associated with selected keys; inputs are unchanged.

    Notes
    -----
    Selected keys are not guaranteed to be sorted. Ties may select different
    input items, so tied payload identity is unspecified. The remaining
    positions are undefined. Zero counts produce no defined output items.
    A partial bit interval selects by that digit rather than the full value.

    Examples
    --------
    For 256 loaded items in a ``cuda.tile`` kernel, select from the first
    193 and consume only the first 16 output positions:

    .. code-block:: python

        best_keys, best_ids = coop.topk_max_pairs(
            keys, ids, k=16, valid_items=193
        )

    See Also
    --------
    topk_min_pairs, topk_max_keys
    """

    return keys, values


@stub(host=False, tile=True, simt=False)
def topk_min_pairs(
    keys: Tile,
    values: Tile,
    /,
    *,
    k,
    valid_items=None,
    begin_bit=None,
    end_bit=None,
    methods=None,
) -> tuple[Tile, Tile]:
    """Select the ``k`` smallest keys and their payloads from a Tile.

    Parameters
    ----------
    keys, values : Tile
        Numeric Tiles of matching shape ``(N,)``. Their dtypes may differ.
        ``N`` must be a positive multiple of 32.
    k : int or Tile
        Number requested, with ``0 <= k <= N``. Accepts a constant,
        runtime integer scalar, or rank-zero Tile. A shape-one Tile is
        unsupported.
    valid_items : int or Tile
        Input prefix length, defaulting to ``N``, in ``[0, N]``.
        Positions beyond it are excluded. Scalar restrictions match ``k``.
    begin_bit, end_bit : int or Tile
        Half-open interval in the radix ordering representation, defaulting
        to zero and the key width. Require
        ``0 <= begin_bit < end_bit <= key_bits``. Scalar restrictions match
        ``k``. Signed and floating-point keys use CUB's ordering transform.
    methods : collections.abc.Mapping or tuple
        Compile-time ``begin_bit`` and ``end_bit`` options only.

    Returns
    -------
    tuple[Tile, Tile]
        ``(selected_keys, selected_values)`` with the respective input dtypes
        and capacity ``N``. Only the first ``min(k, valid_items)`` positions
        are defined.
        Values remain associated with selected keys; inputs are unchanged.

    Notes
    -----
    Selected keys are not guaranteed to be sorted. Ties may select different
    input items, so tied payload identity is unspecified. The remaining
    positions are undefined. Zero counts produce no defined output items.
    A partial bit interval selects by that digit rather than the full value.

    Examples
    --------
    For 256 loaded items in a ``cuda.tile`` kernel, select from the first
    193 and consume only the first 16 output positions:

    .. code-block:: python

        best_keys, best_ids = coop.topk_min_pairs(
            keys, ids, k=16, valid_items=193
        )

    See Also
    --------
    topk_max_pairs, topk_min_keys
    """

    return keys, values


@stub(host=False, tile=True, simt=False)
def histogram_256(samples: Tile, /, *, methods=None) -> Tile:
    """Count byte values in one Tile into all 256 bins.

    Parameters
    ----------
    samples : Tile
        ``uint8`` input of shape ``(N,)``. The implementation's worker width
        must divide both ``N`` and 256.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        ``int32`` counts of shape ``(256,)``. Bin ``b`` counts input values
        equal to ``b``. Unused bins are zero, and counts sum to ``N``.
        Each call starts a new histogram.

    Examples
    --------
    For a loaded byte Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        counts = coop.histogram_256(samples)

    See Also
    --------
    histogram
    """

    return samples


@stub(host=False, tile=True, simt=False)
def histogram(samples: Tile, /, *, bins: int = 256, methods=None) -> Tile:
    """Count byte values in one Tile into a fixed number of bins.

    Parameters
    ----------
    samples : Tile
        ``uint8`` input of shape ``(N,)``. Every sample must be below
        ``bins``. Out-of-range values cause undefined behavior.
        ``N`` must be a positive multiple of 32.
    bins : int
        Compile-time number of bins: 32, 64, 128, or 256 (default).
        The result extent must be a power of two. A supported worker width
        must divide both ``N`` and ``bins``. For 128 workers, use 128 or
        256 bins and an input extent divisible by 128.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty. There is no algorithm selector.

    Returns
    -------
    Tile
        ``int32`` counts of shape ``(bins,)``. Bin ``b`` counts input values
        equal to ``b``. Each call starts at zero; it does not accumulate into
        an existing histogram. Counts sum to ``N`` for valid input.

    Examples
    --------
    For a loaded byte Tile in a kernel admitting 128 workers:

    .. code-block:: python

        counts = coop.histogram(samples, bins=256)

    See Also
    --------
    histogram_256
    """

    return samples


@stub(host=False, tile=True, simt=False)
def run_length_decode(
    run_values: Tile,
    run_lengths: Tile,
    /,
    *,
    decoded_items: int,
    decoded_window_offset: int | Tile = 0,
    methods=None,
) -> Tile:
    """Expand runs into a fixed-capacity window of decoded values.

    Parameters
    ----------
    run_values : Tile
        Numeric values of shape ``(RunN,)``. Inputs are unchanged.
    run_lengths : Tile
        Matching ``uint32`` or ``uint64`` lengths. Positive runs may be
        followed by trailing zero-length runs. A zero-length run must not
        precede a positive run. The sum must fit the length dtype.
    decoded_items : int
        Required compile-time output extent, independent of ``RunN``.
        Both extents must be powers of two, at least 32, and admit a
        common worker count.
    decoded_window_offset : int or Tile
        Starting position in the expanded sequence, defaulting to zero.
        A constant, runtime integral scalar, or single-element integer Tile
        converts to the run-length dtype. The offset must be nonnegative
        and representable in that dtype.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(decoded_items,)`` with the run-value dtype. Positions past
        the expanded sequence are zero, including an entirely out-of-range
        window or an input with all zero lengths.

    Examples
    --------
    Decode a 256-element window from loaded run Tiles:

    .. code-block:: python

        values = coop.run_length_decode(
            run_values, run_lengths, decoded_items=256,
            decoded_window_offset=7,
        )

    See Also
    --------
    run_length_decode_with_offsets
    """

    return run_values


@stub(host=False, tile=True, simt=False)
def run_length_decode_with_offsets(
    run_values: Tile,
    run_lengths: Tile,
    /,
    *,
    decoded_items: int,
    decoded_window_offset: int | Tile = 0,
    methods=None,
) -> tuple[Tile, Tile]:
    """Expand runs and return values followed by offsets within each run.

    Parameters
    ----------
    run_values : Tile
        Numeric values of shape ``(RunN,)``. Inputs are unchanged.
    run_lengths : Tile
        Matching ``uint32`` or ``uint64`` lengths. Positive runs may be
        followed by trailing zero-length runs. A zero-length run must not
        precede a positive run. The sum must fit the length dtype.
    decoded_items : int
        Required compile-time output extent, independent of ``RunN``.
        Both extents must be powers of two, at least 32, and admit a
        common worker count.
    decoded_window_offset : int or Tile
        Starting position in the expanded sequence, defaulting to zero.
        A constant, runtime integral scalar, or single-element integer Tile
        converts to the run-length dtype. The offset must be nonnegative
        and representable in that dtype.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    tuple[Tile, Tile]
        ``(values, relative_offsets)``, both of shape ``(decoded_items,)``.
        Values retain the value dtype; offsets retain the length dtype and
        count from the start of the original run, including for a window
        starting inside a run. Padded values are zero and padded offsets
        are the maximum value of the length dtype.

    Examples
    --------
    Decode a 256-element window from loaded run Tiles:

    .. code-block:: python

        values, offsets = coop.run_length_decode_with_offsets(
            run_values, run_lengths, decoded_items=256,
            decoded_window_offset=7,
        )

    See Also
    --------
    run_length_decode
    """

    return run_values, run_lengths


@stub(host=False, tile=True, simt=False)
def discontinuity_flag_heads(value: Tile, /, *, methods=None) -> Tile:
    """Mark the first element of each equal-value run in a Tile.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        ``int32`` flags of shape ``(N,)``. Position zero is 1; every other
        flag is 1 when its value differs from the preceding value, else 0.
        A matching value in another Tile does not extend this Tile's run.

    Notes
    -----
    Comparison uses numeric inequality, so NaNs flag run boundaries.
    There is no custom predicate or predecessor argument.

    Examples
    --------
    In a ``cuda.tile`` kernel, turn head flags into one-based run IDs:

    .. code-block:: python

        heads = coop.discontinuity_flag_heads(values)
        run_ids = coop.inclusive_sum(heads)

    See Also
    --------
    discontinuity_flag_tails, discontinuity_flag_heads_and_tails
    """

    return value


@stub(host=False, tile=True, simt=False)
def discontinuity_flag_tails(value: Tile, /, *, methods=None) -> Tile:
    """Mark the last element of each equal-value run in a Tile.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        ``int32`` flags of shape ``(N,)``. The last position is 1; every
        other flag is 1 when its value differs from the next value, else 0.
        Runs end at this Tile's boundary.

    Notes
    -----
    Numeric inequality flags NaNs as boundaries. There is no custom
    predicate or successor argument.

    Examples
    --------
    For a loaded numeric Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        tails = coop.discontinuity_flag_tails(values)

    See Also
    --------
    discontinuity_flag_heads, discontinuity_flag_heads_and_tails
    """

    return value


@stub(host=False, tile=True, simt=False)
def discontinuity_flag_heads_and_tails(
    value: Tile, /, *, methods=None
) -> tuple[Tile, Tile]:
    """Mark both ends of each equal-value run in a Tile.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    tuple[Tile, Tile]
        ``(heads, tails)``, two ``int32`` 0/1 Tiles of shape ``(N,)``.
        Heads mark position zero and values unequal to their predecessor.
        Tails mark the last position and values unequal to their successor.
        Neighboring Tiles do not affect either boundary.

    Notes
    -----
    Comparisons use numeric inequality, including for NaNs. Custom
    predicates and external predecessor/successor values are unsupported.

    Examples
    --------
    For a loaded numeric Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        heads, tails = coop.discontinuity_flag_heads_and_tails(values)

    See Also
    --------
    discontinuity_flag_heads, discontinuity_flag_tails
    """

    return value, value


@stub(host=False, tile=True, simt=False)
def adjacent_difference_subtract_left(value: Tile, /, *, methods=None) -> Tile:
    """Subtract each element's left neighbor, preserving the first input.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position zero is ``value[0]``;
        position ``i > 0`` is ``value[i] - value[i - 1]``. The input is
        unchanged, and subtraction stays within this Tile.

    Notes
    -----
    Arithmetic uses the input dtype. Convert to a wider type first if the
    difference can exceed its range. The boundary value is preserved, not
    replaced with a zero difference.

    Examples
    --------
    For a loaded numeric Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        differences = coop.adjacent_difference_subtract_left(values)

    See Also
    --------
    adjacent_difference_subtract_right
    """

    return value


@stub(host=False, tile=True, simt=False)
def adjacent_difference_subtract_right(value: Tile, /, *, methods=None) -> Tile:
    """Subtract each element's right neighbor, preserving the last input.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position ``i < N - 1`` is
        ``value[i] - value[i + 1]``; the last result is ``value[N - 1]``.
        The input is unchanged, and subtraction stays within this Tile.

    Notes
    -----
    Arithmetic uses the input dtype. Convert to a wider type first if the
    difference can exceed its range. No successor value is accepted for
    the last position.

    Examples
    --------
    For a loaded numeric Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        differences = coop.adjacent_difference_subtract_right(values)

    See Also
    --------
    adjacent_difference_subtract_left
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


@stub(host=False, tile=True, simt=False)
def shuffle_up(value: Tile, /, *, methods=None) -> Tile:
    """Read each element's predecessor, preserving the first element.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``. Multiple items per worker are
        supported; the compiler selects a compatible implementation.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position zero is ``value[0]``;
        position ``i > 0`` is ``value[i - 1]``. The shift stays within this
        Tile and leaves the input unchanged.

    Examples
    --------
    For a loaded numeric Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        previous = coop.shuffle_up(values)

    See Also
    --------
    shuffle_down, shuffle_offset, shuffle_rotate
    """

    return value


@stub(host=False, tile=True, simt=False)
def shuffle_down(value: Tile, /, *, methods=None) -> Tile:
    """Read each element's successor, preserving the last element.

    Parameters
    ----------
    value : Tile
        Numeric input of shape ``(N,)``. Multiple items per worker are
        supported; the compiler selects a compatible implementation.
        ``N`` must be a positive multiple of 32.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position ``i < N - 1`` is
        ``value[i + 1]``; the last result is ``value[N - 1]``. The shift
        stays within this Tile and leaves the input unchanged.

    Examples
    --------
    For a loaded numeric Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        following = coop.shuffle_down(values)

    See Also
    --------
    shuffle_up, shuffle_offset, shuffle_rotate
    """

    return value


@stub(host=False, tile=True, simt=False)
def shuffle_offset(
    value: Tile, /, *, distance: int | Tile = 1, methods=None
) -> Tile:
    """Read a relative position, preserving values outside the valid range.

    Parameters
    ----------
    value : Tile
        Numeric Tile of shape ``(N,)``, where ``N`` is 32, 64, 128, 256,
        512, or 1024.
        This operation requires one worker per item; the kernel must admit
        that worker count.
    distance : int or Tile
        Source offset, defaulting to 1. Accepts a constant, runtime scalar,
        or single-element Tile, with ``-N < distance < N``. A positive
        offset reads a higher index and moves values toward lower positions.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position ``i`` reads
        ``value[i + distance]`` when in range, otherwise ``value[i]``.
        The input is unchanged; there is no wraparound.

    Examples
    --------
    For 128 loaded values in a ``cuda.tile`` kernel admitting 128 workers:

    .. code-block:: python

        following = coop.shuffle_offset(values, distance=3)

    See Also
    --------
    shuffle_rotate, shuffle_up, shuffle_down
    """

    return value


@stub(host=False, tile=True, simt=False)
def shuffle_rotate(
    value: Tile, /, *, distance: int | Tile = 1, methods=None
) -> Tile:
    """Read a relative position with wraparound within the Tile.

    Parameters
    ----------
    value : Tile
        Numeric Tile of shape ``(N,)``, where ``N`` is 32, 64, 128, 256,
        512, or 1024.
        This operation requires one worker per item; the kernel must admit
        that worker count.
    distance : int or Tile
        Source offset, defaulting to 1. Accepts a constant, runtime scalar,
        or single-element Tile, with ``0 <= distance < N``. Positive values
        move data toward lower output positions.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        Shape ``(N,)`` with the input dtype. Position ``i`` reads
        ``value[(i + distance) % N]``. The input is unchanged.

    Examples
    --------
    For 128 loaded values in a ``cuda.tile`` kernel admitting 128 workers:

    .. code-block:: python

        rotated = coop.shuffle_rotate(values, distance=3)

    See Also
    --------
    shuffle_offset
    """

    return value


@stub(host=False, tile=True, simt=False)
def radix_rank_keys(
    keys: Tile,
    /,
    *,
    begin_bit: int | Tile = 0,
    end_bit: int | Tile | None = None,
    radix_bits: int | None = None,
    descending: bool = False,
    memoize_outer_scan: bool | None = None,
    inner_scan_algorithm: str | None = None,
    smem_config: str | int | None = None,
    methods=None,
) -> Tile:
    """Return stable destination ranks for one ordered integer-key digit.

    Parameters
    ----------
    keys : Tile
        Signed or unsigned 8-, 16-, 32-, or 64-bit integer keys of shape
        ``(N,)``, with ``N <= 65535``.
        ``N`` must be a positive multiple of 32.
    begin_bit, end_bit : int or Tile
        Half-open interval in the ordered key representation. ``begin_bit``
        defaults to zero and
        ``end_bit`` to ``min(key_bits, begin_bit + radix_bits)``. Bounds may
        be constants, runtime scalars, or single-element Tiles. Require
        ``0 <= begin_bit < end_bit <= key_bits`` and an interval no wider
        than ``radix_bits``.
    radix_bits : int
        Compile-time digit capacity in ``[1, 16]``, defaulting to 4.
        Scratch storage grows exponentially with this value. Configurations
        within this range can fail compilation depending on the worker count
        and target resources.
    descending : bool
        Compile-time direction, defaulting to ``False``.
    memoize_outer_scan : bool
        Compile-time scan memoization, defaulting to ``True``.
    inner_scan_algorithm : str
        Compile-time ``"raking"``, ``"raking_memoize"``, or ``"warp_scans"``
        (the default).
    smem_config : str or int
        Compile-time shared-memory bank width: ``"four_byte"`` or ``4``
        (default), or ``"eight_byte"`` or ``8``.
    methods : collections.abc.Mapping or tuple
        Must be omitted or empty.

    Returns
    -------
    Tile
        ``int32`` ranks of shape ``(N,)``. ``rank[i]`` is the destination
        of input item ``i`` in the selected digit's ordering. Equal digits
        retain input order. Keys are neither reordered nor modified.

    Notes
    -----
    Signed keys have their sign bit flipped before digit extraction, matching
    radix sort's ordering of signed keys. Unsigned
    keys use their original bits. Use radix sort for intervals larger than
    one digit or complete-key sorting.

    Examples
    --------
    Rank the low nibble of a loaded integer Tile in a ``cuda.tile`` kernel:

    .. code-block:: python

        ranks = coop.radix_rank_keys(keys, begin_bit=0, end_bit=4, radix_bits=4)

    See Also
    --------
    radix_sort_keys
    """

    return keys


__all__ = [
    "adjacent_difference_subtract_left",
    "adjacent_difference_subtract_right",
    "discontinuity_flag_heads",
    "discontinuity_flag_heads_and_tails",
    "discontinuity_flag_tails",
    "exchange_blocked_to_striped",
    "exchange_striped_to_blocked",
    "exclusive_scan",
    "exclusive_sum",
    "histogram",
    "histogram_256",
    "inclusive_scan",
    "inclusive_sum",
    "merge_sort_keys",
    "merge_sort_pairs",
    "radix_rank_keys",
    "radix_sort_keys",
    "radix_sort_pairs",
    "reduce",
    "run_length_decode",
    "run_length_decode_with_offsets",
    "scan",
    "shuffle_down",
    "shuffle_offset",
    "shuffle_rotate",
    "shuffle_up",
    "sum",
    "topk_max_keys",
    "topk_max_pairs",
    "topk_min_keys",
    "topk_min_pairs",
]
