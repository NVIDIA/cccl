# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import hashlib
import itertools
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest

pytest.importorskip("cutlass")

from cuda.coop.cutlass._compiler import _bundle as provider_bundle
from cuda.coop.cutlass._compiler import _bundle_contract as provider_contract
from cuda.coop.cutlass._compiler import _rendering as provider_rendering
from cuda.coop.cutlass._compiler import _types as provider_types


@dataclass(frozen=True)
class _Request:
    symbol_name: str
    body: str
    kind: str = "test"
    items_per_thread: int = 1


def _render(requests, include_lines):
    return provider_rendering.render_bundle_source(
        requests,
        scope="cuda.coop.cutlass",
        include_lines=include_lines,
        render_local_request=lambda request: [
            f"void {request.symbol_name}() {{ {request.body}; }}"
        ],
    )


def test_bundle_source_is_independent_of_request_and_preamble_permutations():
    requests = (
        _Request("cuda_coop_z", "int z = 0"),
        _Request("cuda_coop_a", "int a = 0"),
    )
    preamble = (
        "#include <z.cuh>",
        "#define FEATURE_Z",
        "#include <a.cuh>",
        "#define FEATURE_A",
    )
    expected = _render(requests, preamble)
    assert _render((requests[0], requests[0], requests[1]), preamble) == expected

    for request_order in itertools.permutations(requests):
        for preamble_order in itertools.permutations(preamble):
            assert _render(request_order, preamble_order) == expected

    assert expected.index("#define FEATURE_A") < expected.index("#define FEATURE_Z")
    assert expected.index("#define FEATURE_Z") < expected.index("#include <a.cuh>")
    assert expected.index("#include <a.cuh>") < expected.index("#include <z.cuh>")
    assert expected.index("void cuda_coop_a") < expected.index("void cuda_coop_z")


def test_bundle_source_rejects_conflicting_symbols_and_features():
    with pytest.raises(ValueError, match="conflicting bundle requests"):
        _render(
            (
                _Request("cuda_coop_conflict", "int first = 0"),
                _Request("cuda_coop_conflict", "int second = 0"),
            ),
            (),
        )

    with pytest.raises(ValueError, match="conflicting definitions"):
        _render(
            (_Request("cuda_coop_ok", "int value = 0"),),
            ("#define FEATURE 1", "#define FEATURE 2"),
        )


def test_registered_headers_are_canonical_and_reject_conflicts(monkeypatch):
    def render(_request):
        return []

    renderer_a = provider_types.BundleRenderer(
        include_lines=(),
        cccl_headers=(
            ("#include <z.cuh>", "z.cuh"),
            ("#include <a.cuh>", "a.cuh"),
        ),
        render=render,
    )
    renderer_z = provider_types.BundleRenderer(
        include_lines=(),
        cccl_headers=(("#include <m.cuh>", "m.cuh"),),
        render=render,
    )
    monkeypatch.setattr(
        provider_rendering,
        "_BUNDLE_RENDERERS",
        {"z-kind": renderer_z, "a-kind": renderer_a},
    )
    assert list(provider_rendering.registered_bundle_headers()) == [
        "#include <a.cuh>",
        "#include <m.cuh>",
        "#include <z.cuh>",
    ]

    conflicting = provider_types.BundleRenderer(
        include_lines=(),
        cccl_headers=(("#include <a.cuh>", "different/a.cuh"),),
        render=render,
    )
    monkeypatch.setitem(
        provider_rendering._BUNDLE_RENDERERS,
        "conflict",
        conflicting,
    )
    with pytest.raises(ValueError, match="conflicting CCCL headers"):
        provider_rendering.registered_bundle_headers()


def test_required_headers_are_sorted_and_deduplicated():
    headers = provider_bundle.required_cccl_headers(
        "#include <z.cuh>\n#include <alias.cuh>\n#include <a.cuh>\n",
        registered_headers=lambda: {
            "#include <z.cuh>": "z.cuh",
            "#include <alias.cuh>": "a.cuh",
            "#include <a.cuh>": "a.cuh",
        },
    )
    assert headers == ("a.cuh", "z.cuh")


def test_bundle_source_is_independent_of_python_hash_seed():
    script = r"""
import hashlib
import os
from dataclasses import dataclass
import cuda.coop

cuda.coop.__path__.insert(0, os.environ["CUDA_COOP_SOURCE_PATH"])
from cuda.coop.cutlass._compiler import _rendering as provider_rendering

@dataclass(frozen=True)
class Request:
    symbol_name: str
    body: str
    kind: str = "hash_seed_test"
    items_per_thread: int = 1

requests = list({
    Request("cuda_coop_z", "int z = 0"),
    Request("cuda_coop_a", "int a = 0"),
})
lines = list({
    "#include <z.cuh>",
    "#include <a.cuh>",
    "#define FEATURE_Z",
    "#define FEATURE_A",
})
source = provider_rendering.render_bundle_source(
    requests,
    scope="cuda.coop.cutlass",
    include_lines=lines,
    render_local_request=lambda request: [
        f"void {request.symbol_name}() {{ {request.body}; }}"
    ],
)
print(hashlib.sha256(source.encode()).hexdigest())
"""
    digests = []
    for seed in ("1", "7", "100"):
        env = os.environ.copy()
        env["CUDA_COOP_SOURCE_PATH"] = str(
            Path(provider_rendering.__file__).resolve().parents[2]
        )
        env["PYTHONHASHSEED"] = seed
        result = subprocess.run(
            [sys.executable, "-B", "-c", script],
            check=True,
            capture_output=True,
            env=env,
            text=True,
        )
        digests.append(result.stdout.strip())
    assert len(set(digests)) == 1


def test_bundle_cache_identity_is_schema_and_compiler_versioned():
    source_hash = hashlib.sha256(b"source").hexdigest()
    compiler_options = provider_contract.bundle_compiler_options(
        "ltoir",
        "compute_100a",
    )
    identity = provider_contract.make_bundle_identity(
        source_hash=source_hash,
        bundle_format="ltoir",
        bundle_arch="compute_100a",
        bundle_sm_arch="sm_100a",
        compiler_options=compiler_options,
        layout_expressions=("layout-expression",),
    )
    cache_identity = provider_contract.make_bundle_cache_identity(
        identity,
        include_key="0123456789abcdef",
        producer_compiler_version="13.0",
    )

    assert (
        identity.provider_abi_version
        == provider_contract.PROVIDER_BUNDLE_ABI_VERSION
        == 1
    )
    assert identity.compiler_options == compiler_options
    assert identity.layout_expressions == ("layout-expression",)
    assert (
        cache_identity.schema_version
        == provider_contract.BUNDLE_CACHE_SCHEMA_VERSION
        == 2
    )
    assert cache_identity.cache_key == (
        f"v2:{source_hash}:ltoir:compute_100a:sm_100a:13.0:0123456789abcdef"
    )
    assert cache_identity.artifact_stem == (
        f"bundle_v2_{source_hash}_compute_100a_sm_100a_13_0_0123456789abcdef"
    )
