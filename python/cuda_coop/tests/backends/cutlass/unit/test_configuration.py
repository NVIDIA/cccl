# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import textwrap

from ....support.toolchains.cutlass import (
    candidate_ptxas_paths,
    find_cuda_tool,
)
from ..support.source import run_python_with_source


def test_cutlass_provider_block_dim_tokens_preserve_existing_abi():
    script = textwrap.dedent(
        """
        from cuda.coop.cutlass._dsl._symbols import block_dim_token

        assert block_dim_token((64, 1, 1)) == "b64"
        assert block_dim_token((8, 4, 1)) == "b8x4x1"
        assert block_dim_token((8, 4, 2)) == "b8x4x2"
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_collective_group_resolution_reuses_frontend_launch_facts():
    script = textwrap.dedent(
        """
        from cuda.coop._core import LaunchFacts
        from cuda.coop.cutlass._thread_group import (
            _resolve_collective_group_from_launch,
            this_block,
        )

        launch = LaunchFacts(exact_block_dim=(8, 4, 2))
        inferred = _resolve_collective_group_from_launch(
            this_block(),
            launch,
            feature="test_collective",
        )
        assert inferred.hierarchy is not None
        assert inferred.hierarchy.block_dim == (8, 4, 2)
        assert inferred.source == "inferred_launch"
        """
    )

    result = run_python_with_source(script)

    assert result.returncode == 0, result.stderr


def test_cutlass_runtime_toolchain_prefers_ptxas_override(monkeypatch, tmp_path):
    override_ptxas = tmp_path / "override" / "ptxas"
    override_ptxas.parent.mkdir()
    override_ptxas.write_text("", encoding="utf-8")
    override_ptxas.chmod(0o755)

    path_ptxas = tmp_path / "path" / "ptxas"
    path_ptxas.parent.mkdir()
    path_ptxas.write_text("", encoding="utf-8")
    path_ptxas.chmod(0o755)

    monkeypatch.setenv("CUDA_COOP_CUTLASS_PTXAS", str(override_ptxas))
    monkeypatch.setenv("PATH", str(path_ptxas.parent))
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)

    candidates = candidate_ptxas_paths()

    assert candidates[0] == override_ptxas.resolve()
    assert path_ptxas.resolve() in candidates


def test_cutlass_runtime_toolchain_discovers_inspection_tool_override(
    monkeypatch, tmp_path
):
    override_nvdisasm = tmp_path / "override" / "nvdisasm"
    override_nvdisasm.parent.mkdir()
    override_nvdisasm.write_text("", encoding="utf-8")
    override_nvdisasm.chmod(0o755)

    monkeypatch.setenv(
        "CUDA_COOP_CUTLASS_NVDISASM",
        str(override_nvdisasm),
    )
    monkeypatch.setenv("PATH", "")
    monkeypatch.delenv("CUDA_PATH", raising=False)
    monkeypatch.delenv("CUDA_HOME", raising=False)

    assert find_cuda_tool("nvdisasm") == override_nvdisasm.resolve()
