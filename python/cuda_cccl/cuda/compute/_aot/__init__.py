# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Ahead-of-time (AoT) serialization for cuda.compute algorithms.

Submodules:
    serde     -- low-level blob codec (Writer/Reader, framing, descriptor I/O).
    dispatch  -- public free-standing ``serialize`` / ``deserialize`` that route
                 a blob to the right algorithm by its embedded tag.

IMPORTANT: keep this ``__init__`` import-light. The algorithm modules import
``serde`` (``from .._aot import serde``), and ``dispatch`` imports those
algorithm modules; eagerly importing ``dispatch`` here would create a circular
import at startup. The public names are re-exported from
``cuda.compute.algorithms`` instead.
"""

from __future__ import annotations
