# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public, free-standing serialize/deserialize entry points."""

from __future__ import annotations

from typing import Any

from . import codec
from .serializable import Serializable


def serialize(algorithm: Any) -> bytes:
    """Serialize a built algorithm into a blob of bytes.

    Args:
        algorithm: An object returned by a ``make_*`` factory (e.g.
            :func:`make_reduce_into`, :func:`make_exclusive_scan`).

    Returns:
        A versioned, self-describing byte blob. Reconstruct it with
        :func:`deserialize`; no objects required at load time.
    """
    if not callable(getattr(type(algorithm), "serialize", None)):
        raise TypeError(
            f"{type(algorithm).__name__} is not a serializable algorithm "
            "(expected an object from a make_* factory)."
        )
    return algorithm.serialize()


def deserialize(blob: bytes):
    """Reconstruct a built algorithm from a blob produced by :func:`serialize`.

    Warning:
        The returned object is **not safe to use from multiple threads
        concurrently**. Do not deserialize once and share the object across
        threads: every call writes its arguments (array pointers, sizes,
        operator and initial-value state) into the object before launching, so
        overlapping calls can launch kernels with another thread's arguments —
        silently wrong results or CUDA errors, with no exception raised at the
        point of misuse. Unlike the ``make_*`` factories, which hand each
        calling thread its own cached object, ``deserialize`` returns a fresh
        uncached object with no per-thread protection. For concurrent use,
        call :func:`deserialize` in each thread — reconstruction performs no
        recompilation, so per-thread deserialization from one shared blob is
        cheap. One thread at a time (for example, handing the object between
        threads with proper ordering) is fine.

    Raises:
        ValueError: if the blob is malformed or its algorithm tag is unknown.
    """
    tag = codec.peek_algo(blob)
    try:
        cls = Serializable._registry[tag]
    except KeyError:
        raise ValueError(f"serialization blob: unknown algorithm tag {tag!r}") from None
    return cls.deserialize(blob)
