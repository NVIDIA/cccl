# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ahead-of-time (serialization) serialization for cuda.compute algorithms.

This package's ``__init__`` is the front door: everything a consumer needs is
re-exported here, so algorithm modules import ``from .._serialization import ...`` and
never reach into a submodule. The submodules are internal implementation:

    codec        -- low-level blob codec (Writer/Reader, framing, descriptor I/O).
    serializable -- the ``Serializable`` schema base, its kinds, and ``AlgoTag``.
    dispatch     -- the public free-standing ``serialize`` / ``deserialize``.

Importing ``dispatch`` at package init is safe: it depends only on ``codec`` and
``serializable`` (not on the algorithm modules), so there is no import cycle —
each algorithm registers itself with the ``Serializable`` registry as it is
imported, and ``deserialize`` consults that registry at call time.
"""

from __future__ import annotations

from .dispatch import deserialize as deserialize
from .dispatch import serialize as serialize
from .serializable import BOOL as BOOL
from .serializable import BUILD_RESULT as BUILD_RESULT
from .serializable import CONDITIONAL as CONDITIONAL
from .serializable import ENUM as ENUM
from .serializable import ITER as ITER
from .serializable import NESTED as NESTED
from .serializable import OP as OP
from .serializable import U8 as U8
from .serializable import U32 as U32
from .serializable import U64 as U64
from .serializable import VALUE as VALUE
from .serializable import AlgoTag as AlgoTag
from .serializable import Serializable as Serializable

__all__ = [
    "serialize",
    "deserialize",
    "Serializable",
    "AlgoTag",
    "ITER",
    "OP",
    "VALUE",
    "U8",
    "U32",
    "U64",
    "BOOL",
    "ENUM",
    "CONDITIONAL",
    "BUILD_RESULT",
    "NESTED",
]
