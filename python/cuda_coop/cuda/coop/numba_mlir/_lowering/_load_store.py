# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Build the compiled implementations chosen for cooperative load/store calls.

A factory in this module is a host-side function that specializes a CUB
operation for a payload dtype, group size, per-thread extent, and algorithm.
The group planner selects the registered block or warp factory and describes
which optional controls are omitted, fixed at compile time, or supplied at
runtime. Calling the factory creates a device callable; data movement occurs
later when a kernel invokes that callable.

Separate factories represent algorithms that need shared scratch and those
that do not, so their argument lists agree with the planner's storage choice.
Both paths lower a common core specialization through ``NumbaMlirCoreAdapter``.
During specialization collection they return an ``Algorithm`` description
instead of compiling it immediately, allowing several operations to share an
NVRTC compilation.
"""

import operator
from collections.abc import Callable, Iterable
from typing import SupportsIndex, TypedDict, cast

import numba_cuda_mlir.numba_cuda.types as numba_types

from cuda.coop._core import ArgumentBinding, BindingKind, SynchronizationScope
from cuda.coop._core.block import (
    make_block_load_specialization,
    make_block_store_specialization,
)
from cuda.coop._core.warp import (
    make_warp_load_specialization,
    make_warp_store_specialization,
)

from .._compiler._load_store_algorithms import (
    _BLOCK_LOAD_STORE_ALGORITHMS,
    _STORAGE_FREE_ALGORITHMS,
    _WARP_LOAD_STORE_ALGORITHMS,
    _resolve_algorithm,
)
from .._compiler._operations import (
    StorageABI,
    factory_operation,
    register_factory,
)
from .._compiler._parameters import (
    _validate_common_numeric_dtype,
    _validate_static_oob_default,
    normalize_dim_param,
)
from .._types import (
    Algorithm,
    BoundedInteger,
    ExactValue,
    Invocable,
    Value,
    _validate_logical_warp_threads,
    make_invocable_from_specialization,
    numba_type_to_wrapper,
)
from ._core import NumbaMlirCoreAdapter, _optional_binding


class _GroupTopology(TypedDict, total=False):
    block_dim: Iterable[int]
    threads_in_warp: int


def _positive_int(value, *, name: str) -> int:
    if isinstance(value, bool):
        raise TypeError(f"{name} must be an integer")
    try:
        value = operator.index(value)
    except TypeError as exc:
        raise TypeError(f"{name} must be an integer") from exc
    if value <= 0:
        raise ValueError(f"{name} must be a positive integer")
    return value


def _registered_provider_metadata(factory, algorithm):
    registered = factory_operation(factory)
    if registered is None:
        raise RuntimeError(f"unregistered cuda.coop provider {factory!r}")
    expected_storage_abi = (
        StorageABI.NONE
        if algorithm in _STORAGE_FREE_ALGORITHMS
        else StorageABI.LEADING_POINTER
    )
    if registered.storage_abi is not expected_storage_abi:
        raise ValueError(
            f"{registered.operation} algorithm {algorithm!r} "
            f"requires the {expected_storage_abi.value!r} provider"
        )
    return registered


def _materialization_metadata(registered):
    return {
        "storage_abi": registered.storage_abi,
        "execution_scope": registered.execution_scope,
        "synchronization_scope": registered.synchronization_scope,
    }


def _load_store_value_abis(
    *,
    dtype: numba_types.Type,
    items_per_thread: int,
    valid_items: ArgumentBinding,
    oob_default: ArgumentBinding | None = None,
    block_dim: Iterable[int] | None = None,
    threads_in_warp: int | None = None,
) -> dict[str, Value]:
    """Describe runtime controls whose ABI differs from ordinary core scalars.

    A runtime valid-item count enters through a signed 64-bit value so bounds
    can be checked before narrowing to CUB's signed 32-bit count. Its inclusive
    limit is the exact group tile size. The load operation's ``oob_default``
    is the padding value for items beyond that count. When supplied at runtime,
    it uses ``ExactValue`` so typing requires the payload dtype without an
    implicit conversion. Omitted and static controls need no runtime ABI
    override.

    Parameters
    ----------
    dtype : numba_types.Type
        Normalized payload dtype, also required for a runtime default.
    items_per_thread : int
        Positive specialized payload extent per participating thread.
    valid_items : ArgumentBinding
        Binding kind for the optional valid-item count.
    oob_default : ArgumentBinding or None, optional
        Binding for the load operation's padding value; ``None`` supplies no
        override.
    block_dim : iterable of int, optional
        Block dimensions used to compute a block tile's capacity.
    threads_in_warp : int, optional
        Logical warp width used to compute a warp tile's capacity.

    Returns
    -------
    dict of str to Value
        Named backend overrides for ``num_valid_items`` and/or ``oob_default``.

    Raises
    ------
    ValueError
        A runtime count does not have exactly one block/warp topology source, or
        its bounds cannot fit the integer ABI.
    """

    value_abis: dict[str, Value] = {}
    if valid_items.kind is BindingKind.RUNTIME:
        tile_items = items_per_thread
        if (block_dim is None) == (threads_in_warp is None):
            raise ValueError(
                "exactly one of block_dim or threads_in_warp must be provided"
            )
        if block_dim is not None:
            for dimension in block_dim:
                tile_items *= dimension
        else:
            tile_items *= cast(int, threads_in_warp)
        value_abis["num_valid_items"] = BoundedInteger(
            numba_types.int32,
            minimum=0,
            maximum=tile_items,
        )
    if oob_default is not None and oob_default.kind is BindingKind.RUNTIME:
        value_abis["oob_default"] = ExactValue(dtype)
    return value_abis


def _warp_threads(value) -> int:
    value = _positive_int(value, name="threads_in_warp")
    return _validate_logical_warp_threads(value)


def _load(
    provider_factory: Callable[..., object],
    dtype: object,
    threads_per_block: SupportsIndex | tuple[SupportsIndex, ...] | None = None,
    items_per_thread: SupportsIndex = 1,
    algorithm: str = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    offset: object = None,
    threads_in_warp: SupportsIndex | None = None,
) -> Invocable | Algorithm:
    """Build the callable implementation for a planned load operation.

    Use the registered factory identity to select block/warp semantics and
    verify whether the chosen algorithm requires scratch. Lower the common load
    specialization through the Numba adapter, including checked runtime counts,
    exact-dtype runtime defaults, and embedded static controls. The factory
    describes a callable compiled operation; it does not load data here.

    Explicit ``ArgumentBinding`` objects distinguish omitted, static, and
    runtime controls. Legacy non-``None`` values indicate runtime presence, not
    a literal to embed. For a legacy count, retain the full-tile overload as
    well. Legacy offset arguments request offset overloads even when ``None``;
    explicit bindings let the planner select their precise form.

    Parameters
    ----------
    provider_factory : callable
        Registered load factory whose namespace, storage ABI, and
        synchronization metadata govern materialization.
    dtype : object
        Payload dtype accepted by the common numeric profile.
    threads_per_block : SupportsIndex or tuple of SupportsIndex
        Required enclosing block dimensions, including for warp providers.
    items_per_thread : SupportsIndex, optional
        Positive per-thread payload extent; defaults to one.
    algorithm : str, optional
        Specialized load algorithm, resolved within the factory's namespace.
    num_valid_items : ArgumentBinding or object, optional
        Valid-count binding or legacy runtime-presence marker.
    oob_default : ArgumentBinding or object, optional
        Binding or runtime-presence marker for the value used to pad items
        beyond the valid count. Requires a valid-count binding; static padding
        values are checked against the payload dtype.
    offset : ArgumentBinding or object, optional
        Pointer-offset binding or legacy overload-presence input.
    threads_in_warp : SupportsIndex, optional
        Required logical width for warp providers; invalid for block providers.

    Returns
    -------
    Invocable or Algorithm
        Compiled provider callable, or the specialization recorded when
        ``collect_specializations`` is active.

    Raises
    ------
    TypeError
        A dtype, integer extent, or static scalar is unsupported.
    ValueError
        Topology, algorithm/storage pairing, or optional bindings are invalid.
    RuntimeError
        The factory is unregistered or artifact construction fails.
    """

    valid_items_binding = _optional_binding(num_valid_items)
    oob_default_binding = _optional_binding(oob_default)
    offset_binding = _optional_binding(offset)
    if (
        oob_default_binding.kind is not BindingKind.OMITTED
        and valid_items_binding.kind is BindingKind.OMITTED
    ):
        raise ValueError("oob_default requires num_valid_items to be provided")
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="load")
    if oob_default_binding.kind is BindingKind.STATIC:
        oob_default_binding = ArgumentBinding.static(
            _validate_static_oob_default(oob_default_binding.value, dtype)
        )
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    registered = factory_operation(provider_factory)
    if registered is None:
        raise RuntimeError(
            f"unregistered cuda.coop provider {provider_factory!r}"
        )
    group_kwargs: _GroupTopology
    if registered.namespace == "block":
        if threads_in_warp is not None:
            raise ValueError("block Load does not accept threads_in_warp")
        algorithm = _resolve_algorithm(
            algorithm,
            _BLOCK_LOAD_STORE_ALGORITHMS,
            "block load",
        )
        group_kwargs = {"block_dim": block_dim}
    elif registered.namespace == "warp":
        threads_in_warp = _warp_threads(threads_in_warp)
        algorithm = _resolve_algorithm(
            algorithm,
            _WARP_LOAD_STORE_ALGORITHMS,
            "warp load",
        )
        group_kwargs = {"threads_in_warp": threads_in_warp}
    else:
        raise ValueError(
            f"unsupported cuda.coop Load provider namespace "
            f"{registered.namespace!r}"
        )
    registered = _registered_provider_metadata(provider_factory, algorithm)
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            items_per_thread=items_per_thread,
            valid_items=valid_items_binding,
            oob_default=oob_default_binding,
            block_dim=group_kwargs.get("block_dim"),
            threads_in_warp=group_kwargs.get("threads_in_warp"),
        )
    )
    specialization_factory = (
        make_block_load_specialization
        if registered.namespace == "block"
        else make_warp_load_specialization
    )
    core_specialization = specialization_factory(
        dtype=adapter.core_dtype(dtype),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        oob_default=oob_default_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
        **(
            {"block_dim": tuple(block_dim)}
            if registered.namespace == "block"
            else {"threads_in_warp": threads_in_warp}
        ),
    )
    specialization = adapter.materialize(
        core_specialization.specialization,
        **_materialization_metadata(registered),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    invocation_topology = (
        {
            "threads": threads_in_warp,
            "block_threads": block_dim,
        }
        if registered.namespace == "warp"
        else {}
    )
    return make_invocable_from_specialization(
        specialization,
        **invocation_topology,
    )


def load(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a block load callable that needs no shared scratch."""

    return _load(
        load,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def _load_with_storage(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a block transpose load callable with shared scratch."""

    return _load(
        _load_with_storage,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def warp_load(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a warp load callable that needs no shared scratch."""

    return _load(
        warp_load,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def _warp_load_with_storage(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a warp load callable with shared scratch."""

    return _load(
        _warp_load_with_storage,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def _store(
    provider_factory: Callable[..., object],
    dtype: object,
    threads_per_block: SupportsIndex | tuple[SupportsIndex, ...] | None = None,
    items_per_thread: SupportsIndex = 1,
    algorithm: str = "direct",
    num_valid_items: object = None,
    oob_default: object = None,
    offset: object = None,
    threads_in_warp: SupportsIndex | None = None,
) -> Invocable | Algorithm:
    """Build the callable implementation for a planned store operation.

    Resolve the registered block/warp factory and its storage contract, then
    materialize the common store specialization with Numba-specific runtime
    count
    bounds. A warp provider also retains the enclosing block dimensions for
    scratch layout. This constructs a compiled callable; it does not write
    device memory during host-side factory evaluation.

    As in ``_load``, explicit bindings control whether scalar arguments are
    omitted, embedded, or runtime. A legacy non-``None`` count means runtime
    presence and retains a full-tile overload. Legacy offset inputs request
    offset overloads regardless of their value; an explicit binding selects the
    offset form. Store operations do not accept ``oob_default``.

    Parameters
    ----------
    provider_factory : callable
        Registered store factory supplying namespace and ABI metadata.
    dtype : object
        Payload dtype accepted by the common numeric profile.
    threads_per_block : SupportsIndex or tuple of SupportsIndex
        Required enclosing block dimensions, also for warp providers.
    items_per_thread : SupportsIndex, optional
        Positive per-thread payload extent; defaults to one.
    algorithm : str, optional
        Specialized store algorithm within the factory's namespace.
    num_valid_items : ArgumentBinding or object, optional
        Valid-count binding or legacy runtime-presence marker.
    oob_default : None, optional
        Shared factory-interface slot; any non-``None`` value is rejected.
    offset : ArgumentBinding or object, optional
        Pointer-offset binding or legacy overload-presence input.
    threads_in_warp : SupportsIndex, optional
        Required logical width for warp providers; invalid for block providers.

    Returns
    -------
    Invocable or Algorithm
        Compiled provider callable, or the specialization recorded by an active
        ``collect_specializations`` context.

    Raises
    ------
    TypeError
        A dtype or integer extent is unsupported.
    ValueError
        Topology, algorithm/storage pairing, or optional bindings are invalid.
    RuntimeError
        The factory is unregistered or artifact construction fails.
    """

    if oob_default is not None:
        raise ValueError("oob_default is only valid for Load")
    valid_items_binding = _optional_binding(num_valid_items)
    offset_binding = _optional_binding(offset)
    if threads_per_block is None:
        raise ValueError("threads_per_block must be provided")
    block_dim = normalize_dim_param(threads_per_block)
    dtype = _validate_common_numeric_dtype(dtype, operation="store")
    items_per_thread = _positive_int(items_per_thread, name="items_per_thread")
    registered = factory_operation(provider_factory)
    if registered is None:
        raise RuntimeError(
            f"unregistered cuda.coop provider {provider_factory!r}"
        )
    group_kwargs: _GroupTopology
    if registered.namespace == "block":
        if threads_in_warp is not None:
            raise ValueError("block Store does not accept threads_in_warp")
        algorithm = _resolve_algorithm(
            algorithm,
            _BLOCK_LOAD_STORE_ALGORITHMS,
            "block store",
        )
        group_kwargs = {"block_dim": block_dim}
    elif registered.namespace == "warp":
        threads_in_warp = _warp_threads(threads_in_warp)
        algorithm = _resolve_algorithm(
            algorithm,
            _WARP_LOAD_STORE_ALGORITHMS,
            "warp store",
        )
        group_kwargs = {"threads_in_warp": threads_in_warp}
    else:
        raise ValueError(
            f"unsupported cuda.coop Store provider namespace "
            f"{registered.namespace!r}"
        )
    registered = _registered_provider_metadata(provider_factory, algorithm)
    adapter = NumbaMlirCoreAdapter(
        value_abis=_load_store_value_abis(
            dtype=dtype,
            items_per_thread=items_per_thread,
            valid_items=valid_items_binding,
            block_dim=group_kwargs.get("block_dim"),
            threads_in_warp=group_kwargs.get("threads_in_warp"),
        )
    )
    specialization_factory = (
        make_block_store_specialization
        if registered.namespace == "block"
        else make_warp_store_specialization
    )
    core_specialization = specialization_factory(
        dtype=adapter.core_dtype(dtype),
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        valid_items=valid_items_binding,
        include_full_tile=(
            not isinstance(num_valid_items, ArgumentBinding)
            and num_valid_items is not None
        ),
        include_pointer_offset=(
            offset_binding if isinstance(offset, ArgumentBinding) else True
        ),
        **(
            {"block_dim": tuple(block_dim)}
            if registered.namespace == "block"
            else {"threads_in_warp": threads_in_warp}
        ),
    )
    specialization = adapter.materialize(
        core_specialization.specialization,
        **_materialization_metadata(registered),
        extra_type_definitions=(numba_type_to_wrapper(dtype),),
    )
    invocation_topology = (
        {
            "threads": threads_in_warp,
            "block_threads": block_dim,
        }
        if registered.namespace == "warp"
        else {}
    )
    return make_invocable_from_specialization(
        specialization,
        **invocation_topology,
    )


def store(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a block store callable that needs no shared scratch."""

    return _store(
        store,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def _store_with_storage(
    dtype,
    threads_per_block=None,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a block transpose store callable with shared scratch."""

    return _store(
        _store_with_storage,
        dtype,
        threads_per_block,
        items_per_thread,
        algorithm,
        num_valid_items,
        oob_default,
        offset,
    )


def warp_store(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="direct",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a warp store callable that needs no shared scratch."""

    return _store(
        warp_store,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


def _warp_store_with_storage(
    dtype,
    threads_per_block=None,
    threads_in_warp=32,
    items_per_thread=1,
    algorithm="transpose",
    num_valid_items=None,
    oob_default=None,
    offset=None,
):
    """Build a warp store callable with shared scratch."""

    return _store(
        _warp_store_with_storage,
        dtype,
        threads_per_block=threads_per_block,
        threads_in_warp=threads_in_warp,
        items_per_thread=items_per_thread,
        algorithm=algorithm,
        num_valid_items=num_valid_items,
        oob_default=oob_default,
        offset=offset,
    )


for _factory, _operation in ((load, "load"), (store, "store")):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.NONE,
    )
for _factory, _operation in (
    (_load_with_storage, "load"),
    (_store_with_storage, "store"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="block",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.BLOCK,
        synchronization_scope=SynchronizationScope.BLOCK,
    )
for _factory, _operation in ((warp_load, "load"), (warp_store, "store")):
    register_factory(
        _factory,
        operation=_operation,
        namespace="warp",
        storage_abi=StorageABI.NONE,
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=SynchronizationScope.NONE,
    )
for _factory, _operation in (
    (_warp_load_with_storage, "load"),
    (_warp_store_with_storage, "store"),
):
    register_factory(
        _factory,
        operation=_operation,
        namespace="warp",
        storage_abi=StorageABI.LEADING_POINTER,
        execution_scope=SynchronizationScope.WARP,
        synchronization_scope=SynchronizationScope.WARP,
    )
del _factory, _operation
