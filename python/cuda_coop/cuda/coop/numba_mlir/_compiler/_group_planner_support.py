# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Identify the public group calls understood by the Numba planner.

Common and backend-qualified constructors use different Python callables. For
example, ``cuda.coop.this_block`` and ``cuda.coop.numba_mlir.this_block`` both
describe a block, and the tables here map either callable to the backend's
descriptor constructor. The planner also records whether the kernel used the
common API, whose accepted group kinds and arguments must remain consistent
across backends.

That distinction still matters after the public call is replaced by a private
provider call. For example, common ``load`` requires a ``ThreadData`` output;
the Numba-CUDA-MLIR qualified API also accepts a CUDA local array. The planner
must retain the call's API origin to check the appropriate payload rule.
Operation lookup compares the actual registered Python objects, so a user
function named ``load`` is not mistaken for ``cuda.coop.load``.

The planner helpers also share an IR import, temporary-name counter, and base
exception here. Keeping these definitions separate lets group planning and its
operation modules use them without importing each other during initialization.
"""

from __future__ import annotations

from itertools import count
from typing import TYPE_CHECKING, Any

import cuda.coop._core.api as _common_api
import cuda.coop._core.api._dispatch as _common_dispatch

from .. import _thread_group as _thread_groups
from ._operations import group_operation_name

if TYPE_CHECKING:
    from numba_cuda_mlir.numba_cuda.core import (
        ir as ir,  # noqa: PLC0414 - Re-export for typing.
    )
else:
    from numba_cuda_mlir.numbair_transforms import (
        ir as ir,  # noqa: PLC0414 - Re-export for typing.
    )


_NAME_COUNTER = count()
_PAYLOAD_DTYPE_LIKE = "like"
_PAYLOAD_DTYPE_INT32 = "int32"
_GROUP_CONSTRUCTORS = {
    _thread_groups.this_thread: _thread_groups.this_thread,
    _thread_groups.this_warp: _thread_groups.this_warp,
    _thread_groups.this_block: _thread_groups.this_block,
    _thread_groups.this_cluster: _thread_groups.this_cluster,
    _thread_groups.this_grid: _thread_groups.this_grid,
    _common_api.this_thread: _thread_groups.this_thread,
    _common_api.this_warp: _thread_groups.this_warp,
    _common_api.this_block: _thread_groups.this_block,
    _common_api.this_cluster: _thread_groups.this_cluster,
    _common_api.this_grid: _thread_groups.this_grid,
}
_COMMON_GROUP_CONSTRUCTORS = frozenset(
    {
        _common_api.this_thread,
        _common_api.this_warp,
        _common_api.this_block,
        _common_api.this_cluster,
        _common_api.this_grid,
    }
)
_GROUP_METHODS = frozenset(
    {
        "rank",
        "count",
        "rank_as",
        "count_as",
        "sync",
        "sync_aligned",
        "group_by",
        "is_member",
    }
)


class GroupRewriteError(Exception):
    """A recognized group call cannot be lowered with the available facts.

    Report a planning failure to the compiler. ``ForceLiteralArg`` is the
    separate compiler signal for retrying with a specialized argument.
    """


def _group_operation_name(function: object) -> str | None:
    """Find the operation planner associated with a public Python callable.

    Marker detection and call rewriting pass the object recovered from the
    kernel's call target as ``function``. Return its registered name, such as
    ``"load"`` or ``"store"``, or ``None`` if it is not a recognized common
    or qualified operation. Comparing callable identity preserves aliases
    such as ``my_load = coop.load`` without accepting unrelated functions
    that happen to have the same name.
    """

    operation = group_operation_name(function)
    if operation is None:
        operation = _common_dispatch._common_group_operation_name(function)
    return operation


def _is_common_root_operation(function: object, operation: str) -> bool:
    """Tell call rewriting whether to apply the common operation's rules.

    ``_lower_root_operation`` uses this before replacing a public call with a
    backend provider. Since rewriting bypasses the common Python wrapper,
    the planner must check that API's supported group kinds, selector values,
    and payload types itself.

    ``function`` is the Python object recovered from the call target;
    ``operation`` is its expected registered name, such as ``"load"``.
    Return ``True`` only when that object is the common API marker for the
    named operation. Qualified operations and unrelated callables return
    ``False``.
    """

    return _common_dispatch._common_group_operation_name(function) == operation


def _typed_group_payload_like(
    _prototype: Any,
    _is_array: bool,
    _dtype_policy: str,
    _items_per_thread: int | None = None,
) -> Any:
    """Mark a fresh payload whose shape and dtype are resolved later.

    The planner emits this callable into IR while payload facts are still
    being inferred. ``_is_array`` selects inherited array extent or one scalar
    item; ``_items_per_thread`` can override that extent. ``_dtype_policy``
    either inherits the prototype dtype or fixes int32 for rank and
    discontinuity-flag output.

    The provider rewrite replaces the marker with a local-array allocation
    once dtype and extent are known. Calling it directly is an error; it must
    not survive into device compilation.

    Parameters
    ----------
    _prototype : object
        Scalar or array operand retained in the generated IR as the
        source of type and shape evidence.
    _is_array : bool
        Whether the default item count comes from the prototype
        array. False selects one item.
    _dtype_policy : str
        Compile-time rule used by the provider rewrite to select the
        result element dtype.
    _items_per_thread : int or None, optional
        Explicit per-thread element count, overriding the prototype-
        based count when supplied.
    """

    raise GroupRewriteError(
        "typed group payload markers must be lowered before device compilation"
    )


__all__ = [
    "_COMMON_GROUP_CONSTRUCTORS",
    "_GROUP_CONSTRUCTORS",
    "_GROUP_METHODS",
    "_NAME_COUNTER",
    "_PAYLOAD_DTYPE_INT32",
    "_PAYLOAD_DTYPE_LIKE",
    "GroupRewriteError",
    "_group_operation_name",
    "_is_common_root_operation",
    "_typed_group_payload_like",
    "ir",
]
