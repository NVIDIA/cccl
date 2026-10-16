# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Diagnostics raised by the typed pipeline.

Every error is a Numba ``TypingError``, so it is reported with the kernel's
source location like any other typing failure, whether it is raised by a
typing template or by call resolution after typing.
"""

from __future__ import annotations

from . import _bridges as nb

MODE_HINT = (
    "Unset CUDA_COOP_NUMBA_MLIR_PIPELINE to compile this kernel with the "
    "default pipeline."
)

SUPPORTED_SURFACE = (
    "this_block(), ThreadData, and load and store on a block group"
)


class TypedPipelineError(nb.errors.TypingError):
    """A cooperative construct the typed pipeline rejects."""


def unsupported_primitive(qualified_name: str) -> TypedPipelineError:
    """The error for a cooperative primitive the typed pipeline lacks."""

    return TypedPipelineError(
        f"{qualified_name} is not supported on the typed pipeline "
        "(CUDA_COOP_NUMBA_MLIR_PIPELINE=typed). The typed pipeline "
        f"supports {SUPPORTED_SURFACE}. {MODE_HINT}"
    )


__all__ = [
    "MODE_HINT",
    "SUPPORTED_SURFACE",
    "TypedPipelineError",
    "unsupported_primitive",
]
