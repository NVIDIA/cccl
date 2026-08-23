# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Private block-scope compatibility primitives for the CUTLASS backend."""

from .. import _root_load_store
from .._dsl import block as _backend
from .._scoped_api import (
    get_public_attr as _get_public_attr,
)
from .._scoped_api import (
    install_public_exports as _install_public_exports,
)

__all__ = list(_backend.__all__)
_SOURCE_SCOPE = "cuda.coop.cutlass._dsl.block"
_TARGET_SCOPE = __name__

_install_public_exports(
    globals(),
    backend=_backend,
    source=_SOURCE_SCOPE,
    target=_TARGET_SCOPE,
)

_tensor_load = globals()["load"]
_tensor_store = globals()["store"]
_tensor_make_load = globals()["make_load"]
_tensor_make_store = globals()["make_store"]


def load(source, /, *args, **kwargs):
    """Load a block tile into per-thread data.

    CUTLASS tensor sources use the default tensor path. Already-materialized
    Public ``cutlass.Array`` sources dispatch to the Prims array path. Pass
    ``payload=Payload.PRIMS`` to materialize per-thread register data from an
    otherwise memory-backed source. Prims-specific memory controls such as
    ``bounds_check=`` also select that path. ``dtype=`` alone remains on the
    tensor route, and ``offset=`` remains available to canonical CUB.
    """
    return _root_load_store.dispatch_load(
        source,
        *args,
        scope=_TARGET_SCOPE,
        tensor_load=_tensor_load,
        **kwargs,
    )


load.__module__ = __name__
load.__wrapped__ = _backend.load


def store(destination, value, /, *args, **kwargs):
    """Store per-thread data into a block tile.

    CUTLASS tensor destinations use the default tensor path.
    Public ``cutlass.Array`` destinations dispatch to the Prims array path. Pass
    ``payload=Payload.PRIMS`` to store through that path. Prims-specific
    bounds/memory controls such as ``bounds_check=`` also select it. ``dtype=``
    and ``ThreadData.dtype`` alone remain on the tensor route.
    """
    return _root_load_store.dispatch_store(
        destination,
        value,
        *args,
        scope=_TARGET_SCOPE,
        tensor_store=_tensor_store,
        **kwargs,
    )


store.__module__ = __name__
store.__wrapped__ = _backend.store


def make_load(*args, **kwargs):
    """Return a deferred block load callable for tensors or ``cutlass.Array``.

    A factory-bound ``payload=`` selector becomes the deferred-call default and
    can be overridden by a later ``payload=`` argument.  Factory-bound
    Prims-specific bounds/memory controls select the Prims array path and become
    deferred-call defaults. ``dtype=`` and ``offset=`` can stay on canonical CUB.
    """
    return _root_load_store.make_block_load_factory(
        _tensor_make_load,
        load,
        *args,
        scope=_TARGET_SCOPE,
        **kwargs,
    )


make_load.__module__ = __name__
make_load.__wrapped__ = _backend.make_load


def make_store(*args, **kwargs):
    """Return a deferred block store callable for tensors or ``cutlass.Array``.

    A factory-bound ``payload=`` selector becomes the deferred-call default and
    can be overridden by a later ``payload=`` argument.  Factory-bound
    Prims-specific bounds/memory controls select the Prims array path and become
    deferred-call defaults. ``dtype=`` and ``offset=`` can stay on canonical CUB.
    """
    return _root_load_store.make_block_store_factory(
        _tensor_make_store,
        store,
        *args,
        scope=_TARGET_SCOPE,
        **kwargs,
    )


make_store.__module__ = __name__
make_store.__wrapped__ = _backend.make_store


def __getattr__(name: str):
    return _get_public_attr(
        name,
        backend=_backend,
        source=_SOURCE_SCOPE,
        target=_TARGET_SCOPE,
        module_name=__name__,
        namespace=globals(),
    )


def __dir__() -> list[str]:
    return sorted(__all__)
