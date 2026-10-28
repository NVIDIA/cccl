# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Cooperative CUDA primitives for the CUTLASS CuTe DSL."""

from ._compiler._runtime import validate_cutlass_runtime

validate_cutlass_runtime()
del validate_cutlass_runtime
