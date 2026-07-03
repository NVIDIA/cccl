# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public, free-standing serialize/deserialize entry points.

The per-algorithm builder classes (``_Reduce``, ``_Scan``, ...) are private, so
their ``serialize``/``deserialize`` are exposed here as module-level functions.
``deserialize`` dispatches on the algorithm tag embedded in the blob, so a single
function reconstructs any built algorithm with no objects supplied.
"""

from __future__ import annotations

from typing import Any

from . import codec
from .serializable import Serializable


def serialize(algorithm: Any) -> bytes:
    """Serialize a built algorithm into a self-contained serialization blob.

    Args:
        algorithm: An object returned by a ``make_*`` factory (e.g.
            :func:`make_reduce_into`, :func:`make_exclusive_scan`).

    Returns:
        A versioned, self-describing byte blob. Reconstruct it with
        :func:`deserialize` — no objects required at load time.
    """
    # Gate on the method's presence rather than catching AttributeError from the
    # call, so a genuine AttributeError raised *inside* a valid serialize() isn't
    # masked as "not serializable".
    if not callable(getattr(type(algorithm), "serialize", None)):
        raise TypeError(
            f"{type(algorithm).__name__} is not a serializable algorithm "
            "(expected an object from a make_* factory)."
        )
    return algorithm.serialize()


def deserialize(blob: bytes):
    """Reconstruct a built algorithm from a blob produced by :func:`serialize`.

    Takes only the blob: the algorithm kind is read from the blob's header and
    its iterator/operator/value descriptors are rebuilt from the embedded
    sidecar, so no objects need to be supplied. The returned object is callable
    exactly like the one from the corresponding ``make_*`` factory.

    Raises:
        ValueError: if the blob is malformed or its algorithm tag is unknown.
    """
    tag = codec.peek_algo(blob)
    try:
        cls = Serializable._registry[tag]
    except KeyError:
        raise ValueError(f"serialization blob: unknown algorithm tag {tag}") from None
    return cls.deserialize(blob)
