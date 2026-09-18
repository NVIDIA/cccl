# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Compiler integration for the Numba-CUDA-MLIR backend.

This package owns compiler registration, whole-function planning, and the
identity registries connecting public markers to lowering factories.  It does
not define the public ``cuda.coop`` API or primitive-specific CUB providers.
"""

__all__: tuple[str, ...] = ()
