# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Every exported kernel primitive participates in native typing."""

from __future__ import annotations

from ._support import requires_typed_pipeline

pytestmark = requires_typed_pipeline


def test_every_kernel_callable_primitive_has_a_template():
    from numba_cuda_mlir.extending import typing_registry

    from cuda.coop.numba_mlir._pipeline._templates import kernel_primitives

    primitives = kernel_primitives()
    names = set(primitives.values())
    assert {"cuda.coop.load", "cuda.coop.numba_mlir.store"} <= names
    assert "cuda.coop.register" not in names
    registered = {key for key, _ in typing_registry.globals}
    missing = [n for obj, n in primitives.items() if obj not in registered]
    assert not missing
