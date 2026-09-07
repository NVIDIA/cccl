# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import re
import subprocess

from ...support.paths import PACKAGE_ROOT

CUDA_COOP_ROOT = PACKAGE_ROOT
REPO_ROOT = CUDA_COOP_ROOT.parents[1]
_TEXT_SUFFIXES = {
    ".cpp",
    ".css",
    ".cu",
    ".cuh",
    ".h",
    ".hpp",
    ".md",
    ".mdx",
    ".py",
    ".rst",
    ".toml",
    ".ts",
    ".tsx",
    ".txt",
}


def _owned_text_files():
    try:
        tracked = subprocess.run(
            ["git", "ls-files", "-z", "--", "python/cuda_coop"],
            cwd=REPO_ROOT,
            check=True,
            stdout=subprocess.PIPE,
        ).stdout
        candidates = sorted(
            REPO_ROOT / raw.decode("utf-8", errors="surrogateescape")
            for raw in tracked.split(b"\0")
            if raw
        )
    except (OSError, subprocess.CalledProcessError):
        candidates = sorted(CUDA_COOP_ROOT.rglob("*"))

    for path in candidates:
        if not path.is_file() or path.suffix not in _TEXT_SUFFIXES:
            continue
        yield path


def test_cuda_coop_owned_names_do_not_use_legacy_prefixes():
    forbidden = (
        "CUTE" + "_COOP_",
        "CUDA_CCCL" + "_COOP_",
        "cutlass_cute" + "_coop_",
        "cutlass" + "_coop_",
        "CCCL" + "_ENABLE_CACHE",
    )
    violations = []
    for path in _owned_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for prefix in forbidden:
            if prefix in text:
                violations.append(f"{path.relative_to(CUDA_COOP_ROOT)}: {prefix}")

    assert violations == []


def test_cuda_coop_owned_generic_environment_names_are_gone():
    forbidden = (
        "CUDA_COOP_" + "DEBUG",
        "CUDA_COOP_" + "INJECT_PRINTFS",
        "CUDA_COOP_" + "NVRTC_",
        "CUDA_COOP_" + "BUNDLE_LTOIR",
    )
    violations = []
    for path in _owned_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        for name in forbidden:
            if name in text:
                violations.append(f"{path.relative_to(CUDA_COOP_ROOT)}: {name}")

    assert violations == []


def test_cuda_coop_owned_cub_request_identifiers_use_dsl_namespace():
    bare_cub_request = re.compile(r"""["']cub_(?:block|warp)_""")
    violations = []
    for path in _owned_text_files():
        text = path.read_text(encoding="utf-8", errors="replace")
        if bare_cub_request.search(text):
            violations.append(str(path.relative_to(CUDA_COOP_ROOT)))

    assert violations == []
