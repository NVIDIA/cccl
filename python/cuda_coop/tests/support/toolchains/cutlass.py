# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path

CUDA_TOOL_OVERRIDE_ENV_VARS = {
    "cuobjdump": "CUDA_COOP_CUTLASS_CUOBJDUMP",
    "nvdisasm": "CUDA_COOP_CUTLASS_NVDISASM",
    "ptxas": "CUDA_COOP_CUTLASS_PTXAS",
}
PTXAS_OVERRIDE_ENV_VAR = CUDA_TOOL_OVERRIDE_ENV_VARS["ptxas"]


def candidate_cuda_tool_paths(tool_name: str) -> tuple[Path, ...]:
    if not tool_name or Path(tool_name).name != tool_name:
        raise ValueError("tool_name must be a non-empty executable basename")

    candidates: list[Path] = []

    def add(path: str | Path | None) -> None:
        if path is None:
            return
        tool = Path(path).expanduser()
        if tool.is_file():
            candidates.append(tool.resolve())

    override_env_var = CUDA_TOOL_OVERRIDE_ENV_VARS.get(tool_name)
    if override_env_var is not None:
        add(os.environ.get(override_env_var))
    add(shutil.which(tool_name))
    for env_var in ("CUDA_PATH", "CUDA_HOME"):
        cuda_root = os.environ.get(env_var)
        if cuda_root:
            add(Path(cuda_root) / "bin" / tool_name)
    for tool in sorted(
        Path("/usr/local").glob(f"cuda-*/bin/{tool_name}"), reverse=True
    ):
        add(tool)
    add(Path("/usr/local/cuda/bin") / tool_name)

    return tuple(dict.fromkeys(candidates))


def candidate_ptxas_paths() -> tuple[Path, ...]:
    return candidate_cuda_tool_paths("ptxas")


def find_cuda_tool(tool_name: str) -> Path | None:
    candidates = candidate_cuda_tool_paths(tool_name)
    return candidates[0] if candidates else None


def ptxas_supports_ptx_93(ptxas: Path) -> bool:
    if not ptxas.is_file():
        return False

    ptx_src = (
        ".version 9.3\n"
        ".target sm_100\n"
        ".address_size 64\n\n"
        ".visible .entry _probe() {\n"
        "  ret;\n"
        "}\n"
    )

    with tempfile.TemporaryDirectory(prefix="cuda-coop-cutlass-ptxas-") as tmpdir:
        ptx_path = Path(tmpdir) / "probe.ptx"
        cubin_path = Path(tmpdir) / "probe.cubin"
        ptx_path.write_text(ptx_src, encoding="utf-8")
        result = subprocess.run(
            [str(ptxas), str(ptx_path), "-arch=sm_100", "-o", str(cubin_path)],
            check=False,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
        )
    return result.returncode == 0


def find_ptxas_with_ptx_93() -> tuple[Path | None, tuple[Path, ...]]:
    candidates = candidate_ptxas_paths()
    for ptxas in candidates:
        if ptxas_supports_ptx_93(ptxas):
            return ptxas, candidates
    return None, candidates


def ptxas_probe_skip_reason() -> str:
    _, candidates = find_ptxas_with_ptx_93()
    probed = ", ".join(str(candidate) for candidate in candidates) or "none"
    return f"requires ptxas support for PTX .version 9.3; probed {probed}"
