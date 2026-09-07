# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import importlib.util
import re

import pytest

from ....support.paths import PACKAGE_ROOT

pytest.importorskip("numba_cuda_mlir")

from cuda.coop.numba_mlir import _caching, _single_phase_rewrites, _types  # noqa: E402


def test_algorithm_generated_symbols_use_numba_mlir_namespace():
    algo = _types.Algorithm("Algo", "Method", "algo", [], [], [[]])

    assert algo.c_name == "cuda_coop_numba_mlir_algo"
    assert all(
        name.startswith("cuda_coop_numba_mlir_")
        for name in algo._temp_storage_symbol_names()
    )


def test_rewrite_generated_globals_use_numba_mlir_namespace():
    name = _single_phase_rewrites._next_global_name("test")

    assert name.startswith("__cuda_coop_numba_mlir_test_")
    assert name.endswith("__")


def test_udf_symbols_use_numba_mlir_namespace():
    def binary_op(lhs, rhs):
        return lhs + rhs

    operator_symbol = _types._python_operator_symbol_name(
        binary_op,
        "int",
        ["int", "int"],
    )
    method_symbol = _types._python_type_method_symbol_name(
        "DummyType", "construct", binary_op
    )

    assert operator_symbol.startswith("cuda_coop_numba_mlir_F")
    assert method_symbol.startswith("cuda_coop_numba_mlir_type_DummyType_construct_")


def test_udf_symbols_sanitize_dtype_fragments_as_c_identifiers():
    def binary_op(lhs, rhs):
        return lhs + rhs

    symbol = _types._python_operator_symbol_name(
        binary_op,
        "Record(x[type=int32;offset=0])",
        ["array(int32, 1d, C)", "Tuple(int32, float32)"],
    )

    assert re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", symbol)


def test_symbol_namespace_rename_bumped_cache_version(monkeypatch):
    assert _caching._CACHE_SCHEMA_VERSION == 4
    current_hash = _caching.json_hash("same-input")

    monkeypatch.setattr(_caching, "_CACHE_SCHEMA_VERSION", 3)

    assert _caching.json_hash("same-input") != current_hash


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, False),
        ("", False),
        ("0", False),
        ("false", False),
        ("off", False),
        ("1", True),
        ("yes", True),
    ],
)
def test_cache_environment_flag_semantics(monkeypatch, value, expected):
    if value is None:
        monkeypatch.delenv("CUDA_COOP_ENABLE_CACHE", raising=False)
    else:
        monkeypatch.setenv("CUDA_COOP_ENABLE_CACHE", value)

    caching_path = PACKAGE_ROOT / "cuda" / "coop" / "numba_mlir" / "_caching.py"
    spec = importlib.util.spec_from_file_location(
        f"_cuda_coop_numba_mlir_caching_{value!r}", caching_path
    )
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)

    assert module._ENABLE_CACHE is expected
