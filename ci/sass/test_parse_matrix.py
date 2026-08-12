#!/usr/bin/env python3
"""Tests for ci/sass/parse_matrix.py.

Run with: python3 -m pytest ci/sass/test_parse_matrix.py

The resolution of `ctk` and `cxx` is checked against the real ci/matrix.yaml,
because the point of both is to agree with what every other job in that file
means by the same value.
"""

import sys
from pathlib import Path

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))

from parse_matrix import (  # noqa: E402
    Workflow,
    parse_matrix,
    resolve_ctk,
    resolve_cxx,
)

MATRIX_PATH = Path(__file__).resolve().parent.parent / "matrix.yaml"


@pytest.fixture(scope="module")
def matrix() -> dict:
    with MATRIX_PATH.open() as f:
        return yaml.safe_load(f)


def test_a_ctk_alias_resolves_to_a_real_version(matrix) -> None:
    """`13.X` follows the newest CTK, so it must not resolve to itself."""
    resolved = resolve_ctk(matrix, "13.X")
    # YAML reads the version keys as numbers, so compare as strings.
    assert resolved in {str(key) for key in matrix["ctk_versions"]}
    assert resolved != "13.X"


def test_an_explicit_ctk_version_is_kept(matrix) -> None:
    assert resolve_ctk(matrix, "12.0") == "12.0"


def test_an_unknown_ctk_is_rejected(matrix) -> None:
    with pytest.raises(KeyError):
        resolve_ctk(matrix, "99.X")


def test_clang_resolves_to_the_llvm_container_tag(matrix) -> None:
    """The image name is not the compiler name, so the tag must be looked up."""
    assert resolve_cxx(matrix, "clang19") == "llvm19"


def test_gcc_keeps_its_name_and_version(matrix) -> None:
    assert resolve_cxx(matrix, "gcc13") == "gcc13"


def test_a_cxx_without_a_version_is_allowed(matrix) -> None:
    """`cxx: 'gcc'` means the newest, and `launch.sh` accepts the bare tag."""
    assert resolve_cxx(matrix, "gcc") == "gcc"


def test_an_unknown_cxx_is_rejected(matrix) -> None:
    with pytest.raises(KeyError):
        resolve_cxx(matrix, "icc19")


def test_the_entry_carries_the_resolved_launch_args() -> None:
    """`launch.sh` is called with these, so the spelling must be exact."""
    row = parse_matrix(MATRIX_PATH, Workflow.PULL_REQUEST)["include"][0]
    cuda, host = row["launch_args"].split()[1::2]
    assert row["launch_args"].strip() == f"--cuda {cuda} --host {host}"
    assert cuda[0].isdigit()


def test_the_filters_reach_the_workflow_as_json() -> None:
    """A matrix value cannot hold a list, so the filters are serialized."""
    import json

    row = parse_matrix(MATRIX_PATH, Workflow.PULL_REQUEST)["include"][0]
    assert isinstance(json.loads(row["target_filters_json"]), list)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
