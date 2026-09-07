# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Public-surface parity that requires the Numba-CUDA-MLIR runtime."""

import inspect
from types import SimpleNamespace

import cuda.coop.numba_mlir as numba_mlir_coop

from ....support.cases.api_contracts import (
    SCOPED_BLOCK_EXPORTS,
    SCOPED_ROOT_EXPORTS,
    SCOPED_WARP_EXPORTS,
)


def test_private_block_and_warp_retain_internal_compatibility_contract():
    assert SCOPED_ROOT_EXPORTS.issubset(numba_mlir_coop.__all__)
    for name in SCOPED_ROOT_EXPORTS:
        assert getattr(numba_mlir_coop, name) is not None
    for retired_name in ("block", "warp"):
        assert retired_name not in numba_mlir_coop.__all__
        assert not hasattr(numba_mlir_coop, retired_name)

    expected_by_scope = {
        "_block": SCOPED_BLOCK_EXPORTS,
        "_warp": SCOPED_WARP_EXPORTS,
    }
    for scope_name, expected_names in expected_by_scope.items():
        numba_mlir_scope = getattr(numba_mlir_coop, scope_name)
        assert inspect.getdoc(numba_mlir_scope).startswith("Private ")
        assert expected_names.issubset(set(numba_mlir_scope.__all__))

        for name in expected_names:
            assert getattr(numba_mlir_scope, name) is not None

        for name in expected_names:
            if name.startswith("make_"):
                assert getattr(numba_mlir_scope, name) is getattr(
                    numba_mlir_scope, name[5:]
                )


def test_internal_primitive_docstrings():
    for scope_name in ("_block", "_warp"):
        scope = getattr(numba_mlir_coop, scope_name)
        for name in scope.__all__:
            obj = getattr(scope, name)
            assert inspect.getdoc(obj), f"{scope.__name__}.{name}"


def test_single_phase_rewrite_recognizes_private_stateful_parent_factories():
    from cuda.coop.numba_mlir._block import _block_run_length_decode
    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    factories = (
        numba_mlir_coop._block.histogram,
        _block_run_length_decode._common_run_length,
        _block_run_length_decode._qualified_group_run_length,
    )

    assert all(rewrite._is_supported_parent_factory(factory) for factory in factories)


def test_single_phase_rewrite_rejects_retired_public_scope_tokens():
    from cuda.coop.numba_mlir._single_phase_rewrites import CoopSinglePhaseRewrite

    rewrite = object.__new__(CoopSinglePhaseRewrite)
    root = SimpleNamespace(__name__="cuda.coop.numba_mlir")

    assert rewrite._chain_can_be_coop(root, ["_block", "sum"])
    assert rewrite._chain_can_be_coop(root, ["_warp", "sum"])
    assert not rewrite._chain_can_be_coop(root, ["block", "sum"])
    assert not rewrite._chain_can_be_coop(root, ["warp", "sum"])
