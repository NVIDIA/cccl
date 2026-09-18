# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
# ruff: noqa: E402

"""Compiler-owned launch facts and shared group resolution."""

from types import SimpleNamespace

import pytest

pytest.importorskip("cutlass")

from cuda.coop._core import LaunchFacts
from cuda.coop.cutlass._compiler import _launch
from cuda.coop.cutlass._thread_group import (
    ThreadGroup,
    _resolve_primitive_group_from_launch,
    this_block,
)

pytestmark = [pytest.mark.backend_cutlass, pytest.mark.unit]


def test_exact_dimensions_and_flags_keep_verified_provenance():
    facts = _launch.launch_facts_from_cutlass_api(
        SimpleNamespace(
            exact_block_dim=(8, 4, 2),
            exact_grid_dim=(12, 2, 1),
            exact_cluster_dim=(2, 1, 1),
            cooperative_launch=False,
            cluster_launch=True,
        )
    )
    assert facts.exact_block_dim == (8, 4, 2)
    assert facts.exact_block_threads == 64
    assert facts.exact_grid_dim == (12, 2, 1)
    assert facts.exact_cluster_dim == (2, 1, 1)
    assert facts.cooperative_launch is False
    assert facts.cluster_launch is True
    assert all(facts.is_verified(origin.fact) for origin in facts.provenance)


def test_upper_bound_never_becomes_an_exact_fact():
    facts = _launch.launch_facts_from_cutlass_api(
        SimpleNamespace(max_block_dim=(128, 1, 1))
    )
    assert facts == LaunchFacts()
    with pytest.raises(NotImplementedError, match="exact block dimensions"):
        _resolve_primitive_group_from_launch(this_block(), facts, feature="load")


@pytest.mark.parametrize(
    "dimension", [(8, 4), (8, 4, 0), (8, 4, -1), (8, True, 1), (8, 4.0, 1), 32]
)
def test_malformed_exact_dimensions_are_rejected(dimension):
    with pytest.raises(ValueError, match="exact_block_dim"):
        _launch.launch_facts_from_cutlass_api(
            SimpleNamespace(exact_block_dim=dimension)
        )


@pytest.mark.parametrize("flag", ["cooperative_launch", "cluster_launch"])
@pytest.mark.parametrize("value", [0, 1, "true"])
def test_launch_mode_requires_boolean(flag, value):
    with pytest.raises(ValueError, match=flag):
        _launch.launch_facts_from_cutlass_api(SimpleNamespace(**{flag: value}))


def test_unknown_optional_launch_modes_stay_unknown():
    facts = _launch.launch_facts_from_cutlass_api(
        SimpleNamespace(exact_block_dim=(32, 1, 1))
    )
    assert facts.exact_cluster_dim is None
    assert facts.cluster_launch is None
    assert facts.cooperative_launch is None
    assert not facts.is_verified("cluster_launch")


def test_current_facts_use_only_the_compiler_hook(monkeypatch):
    runtime = SimpleNamespace(
        cute=SimpleNamespace(
            _get_launch_facts=lambda: SimpleNamespace(exact_block_dim=(8, 4, 2))
        )
    )
    monkeypatch.setattr(_launch, "validate_cutlass_runtime", lambda: runtime)
    assert _launch.current_kernel_block_dim() == (8, 4, 2)


def test_unavailable_compiler_facts_preserve_the_error(monkeypatch):
    error = RuntimeError("launch facts are unavailable")

    def unavailable():
        raise error

    runtime = SimpleNamespace(cute=SimpleNamespace(_get_launch_facts=unavailable))
    monkeypatch.setattr(_launch, "validate_cutlass_runtime", lambda: runtime)
    with pytest.raises(RuntimeError, match="launch facts are unavailable") as caught:
        _launch.current_kernel_launch_facts()
    assert caught.value is error


def test_block_descriptor_resolves_without_mutating_the_symbolic_group():
    symbolic = this_block()
    resolved = _resolve_primitive_group_from_launch(
        symbolic, LaunchFacts(exact_block_dim=(8, 4, 2)), feature="load"
    )
    assert isinstance(resolved, ThreadGroup)
    assert resolved.block_dim == (8, 4, 2)
    assert resolved.group_thread_count == 64
    assert symbolic.block_dim is None
    with pytest.raises(ValueError, match="do not match"):
        _resolve_primitive_group_from_launch(
            resolved, LaunchFacts(exact_block_dim=(64, 1, 1)), feature="store"
        )
