# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative operations on logical Tiles inside ``cuda.tile`` kernels.

Use ``import cuda.coop.tile as coop`` in kernel modules. Most operations
accept inputs of compile-time one-dimensional shape ``(N,)`` and numeric
element types:
signed or unsigned 8-, 16-, 32-, or 64-bit integers, ``float16``, ``bfloat16``,
``float32``, or ``float64``. Individual operations narrow these choices.
The compiler and native provider select physical participation and scratch
storage. Supported input extents depend on the operation. Operations other
than reductions require a width of 32, 64, 128, or 256 to divide ``N``.
Exchange's explicit width defines its permutation. Reductions also support
multidimensional inputs
and explicit axes, including 1, 2, 4, 8 or 16 elements per result.

Operations return new Tile values and leave inputs unchanged. Reductions
remove reduced axes by default; ``keepdims=True`` retains them with extent
one, and ``broadcast=True`` broadcasts aggregates to the input shape. TopK
preserves capacity while defining only a leading prefix. See each function
for its result and boundary contracts.
Examples in the function docstrings assume the ``coop`` import above and
already loaded Tiles inside a kernel.

The optional ``methods`` argument is a compile-time mapping or flat tuple
of alternating names and values, such as ``("algorithm", "raking")``.
Prefer named keyword arguments. A non-``None`` constant keyword overrides
the corresponding method value; a runtime keyword cannot also appear in
``methods``. Unknown method keys are errors. Numeric runtime controls
convert to the required dtype; integer controls use signed 32-bit values,
except Run Length Decode offsets, which use the unsigned length dtype.
For integer data, constant seeds and padding keys must be integers, not
floating-point values or booleans. Invalid constant controls fail during
compilation; runtime controls must satisfy each family's documented bounds.

Importing this package also exposes the C++ configuration helpers without
loading the Python compiler. Accessing a primitive loads cuTile and registers
its lowering once. Install ``cuda-coop-tile`` with compatible ``cuda-tile`` and
``cuda-tile-preview`` packages
for Python kernel compilation.
"""

from collections.abc import Mapping
from typing import TypeAlias

from cuda.tile import Tile as _Tile

from ._cpp import CppTileConfig as CppTileConfig
from ._cpp import get_cmake_dir as get_cmake_dir
from ._cpp import get_cpp_config as get_cpp_config
from ._cpp import get_cpp_include_path as get_cpp_include_path
from ._cpp import get_cpp_interface_path as get_cpp_interface_path
from ._cpp import get_cpp_tile_library_path as get_cpp_tile_library_path
from ._register import register as register

_Methods: TypeAlias = Mapping[str, object] | tuple[object, ...]
_RuntimeInt: TypeAlias = int | _Tile
_RuntimeNumber: TypeAlias = float | int | _Tile

def exclusive_sum(
    value: _Tile,
    /,
    *,
    prefix_value: _RuntimeNumber | None = None,
    algorithm: str | None = None,
    methods: _Methods | None = None,
) -> _Tile:
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

def inclusive_sum(
    value: _Tile,
    /,
    *,
    prefix_value: _RuntimeNumber | None = None,
    algorithm: str | None = None,
    methods: _Methods | None = None,
) -> _Tile:
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

def scan(
    value: _Tile,
    /,
    *,
    mode: str = "exclusive",
    scan_op: str | None = None,
    initial_value: _RuntimeNumber | None = None,
    prefix_value: _RuntimeNumber | None = None,
    algorithm: str | None = None,
    methods: _Methods | None = None,
) -> _Tile:
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

def exclusive_scan(
    value: _Tile,
    /,
    *,
    scan_op: str | None = None,
    initial_value: _RuntimeNumber | None = None,
    prefix_value: _RuntimeNumber | None = None,
    algorithm: str | None = None,
    methods: _Methods | None = None,
) -> _Tile:
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

def inclusive_scan(
    value: _Tile,
    /,
    *,
    scan_op: str | None = None,
    initial_value: None = None,
    prefix_value: _RuntimeNumber | None = None,
    algorithm: str | None = None,
    methods: _Methods | None = None,
) -> _Tile:
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

def reduce(
    value: _Tile,
    /,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    *,
    binary_op: str | None = None,
    algorithm: str | None = None,
    methods: _Methods | None = None,
    broadcast: bool = False,
) -> _Tile:
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

def sum(
    value: _Tile,
    /,
    axis: int | tuple[int, ...] | None = None,
    keepdims: bool = False,
    *,
    algorithm: str | None = None,
    methods: _Methods | None = None,
    broadcast: bool = False,
) -> _Tile:
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

def discontinuity_flag_heads(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> _Tile:
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

def discontinuity_flag_tails(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> _Tile:
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

def discontinuity_flag_heads_and_tails(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> tuple[_Tile, _Tile]:
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

def adjacent_difference_subtract_left(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> _Tile:
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

def adjacent_difference_subtract_right(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> _Tile:
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

def exchange_striped_to_blocked(
    value: _Tile,
    /,
    *,
    block_threads: int,
    methods: _Methods | None = None,
) -> _Tile:
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

def exchange_blocked_to_striped(
    value: _Tile,
    /,
    *,
    block_threads: int,
    methods: _Methods | None = None,
) -> _Tile:
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

def shuffle_up(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> _Tile:
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

def shuffle_down(
    value: _Tile,
    /,
    *,
    methods: _Methods | None = None,
) -> _Tile:
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

def shuffle_offset(
    value: _Tile,
    /,
    *,
    distance: _RuntimeInt = 1,
    methods: _Methods | None = None,
) -> _Tile:
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

def shuffle_rotate(
    value: _Tile,
    /,
    *,
    distance: _RuntimeInt = 1,
    methods: _Methods | None = None,
) -> _Tile:
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

__all__ = [
    "CppTileConfig",
    "adjacent_difference_subtract_left",
    "adjacent_difference_subtract_right",
    "discontinuity_flag_heads",
    "discontinuity_flag_heads_and_tails",
    "discontinuity_flag_tails",
    "exchange_blocked_to_striped",
    "exchange_striped_to_blocked",
    "exclusive_scan",
    "exclusive_sum",
    "get_cmake_dir",
    "get_cpp_config",
    "get_cpp_include_path",
    "get_cpp_interface_path",
    "get_cpp_tile_library_path",
    "inclusive_scan",
    "inclusive_sum",
    "reduce",
    "register",
    "scan",
    "shuffle_down",
    "shuffle_offset",
    "shuffle_rotate",
    "shuffle_up",
    "sum",
]
