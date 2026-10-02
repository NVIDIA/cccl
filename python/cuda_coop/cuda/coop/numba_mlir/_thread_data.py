# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Expose per-thread payload construction to the Numba-CUDA-MLIR compiler.

``ThreadData`` is a kernel-language marker: the rewrite replaces a supported
call with a fixed-size local array, inferring its dtype from the surrounding
operations when possible. Calling the Python function directly raises a
compiler-context error. Its alignment helper reconciles the common API's
requested minimum with the compiler's pointer-alignment requirement.

``local`` and ``shared`` expose the active runtime's array namespaces. They
are loaded on first attribute access so importing this module does not itself
require those runtime namespaces to be initialized.
"""

from __future__ import annotations

import struct
from typing import SupportsIndex

from .._core.api._payload import _normalize_alignment
from .._core.thread_group import CoopCompilerContextRequiredError
from ._compiler._activation import _require_runtime

# Annotations keep the runtime namespaces lazy while documenting module
# ownership for introspection and static analysis.
local: object
shared: object


def _normalize_thread_data_alignment(
    alignment: SupportsIndex | None,
) -> int | None:
    """Convert a requested payload alignment to the compiler's minimum.

    The common API accepts a positive power-of-two byte alignment, while the
    Numba local-array representation also requires pointer alignment. Raise
    smaller explicit requests to the host pointer size used by this adapter;
    this still satisfies the caller's requested minimum. Leave unspecified
    alignment for the compiler to choose.

    Parameters
    ----------
    alignment : SupportsIndex or None
        Requested minimum in bytes. Integer-index values are normalized by
        the common helper; booleans are rejected. ``None`` means unspecified.

    Returns
    -------
    int or None
        At least ``struct.calcsize("P")`` for an explicit request, otherwise
        ``None``.

    Raises
    ------
    TypeError
        The request is a boolean or cannot be interpreted as an integer.
    ValueError
        The request is not a positive power of two.
    """

    alignment = _normalize_alignment(alignment)
    # The compiler requires pointer-aligned arrays. Stronger alignment also
    # satisfies smaller minimum-alignment requests from the common API.
    return None if alignment is None else max(struct.calcsize("P"), alignment)


def ThreadData(
    items_per_thread,
    dtype=None,
    *,
    alignment=None,
):
    """Create a fixed-size Numba local array for cooperative operations.

    The parameters, uninitialized contents, and example follow
    :func:`cuda.coop.ThreadData`. This qualified constructor returns the
    active Numba-CUDA-MLIR runtime's local-array representation. The compiler
    infers the element type from supported context, including Load's
    source, Store's destination, and typed indexed assignments. Leave the
    constructor type unspecified for those uses. See
    :ref:`element-type inference <coop-faq-thread-data-dtype>` for cases that
    need additional type information.

    Prefer a kernel parameter named ``items_per_thread`` and construct the
    payload with ``ThreadData(items_per_thread)``. The compiler specializes
    the kernel for each supplied count, which remains fixed during execution.

    ``alignment`` requests a minimum alignment in bytes. This backend also
    enforces the compiler's pointer-alignment requirement. See
    :ref:`per-thread payloads <coop-thread-data>` for indexing and lifetime.
    """

    raise CoopCompilerContextRequiredError(
        "cuda.coop.numba_mlir.ThreadData must "
        "be called from a supported GPU kernel."
    )


def __getattr__(name: str):
    if name in {"local", "shared"}:
        value = getattr(_require_runtime(), name)
        globals()[name] = value
        return value
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


def __dir__() -> list[str]:
    return sorted(__all__)


__all__ = ["ThreadData", "local", "shared"]
