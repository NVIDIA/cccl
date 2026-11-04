#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select PR compile-time benchmarks from their inputs before allocating runners."""

from __future__ import annotations

from typing import Sequence

import inspect_changes

# The benchmark builds public headers, not Python packages or C++ library tests.
# Reuse their ownership from the matrix classifier.
PUBLIC_PROJECTS = ("libcudacxx_public", "cub_public", "thrust_public")
BENCHMARK_INPUTS = (
    r"ci/compile_time/",
    r"ci/build_compile_time_bench\.sh$",
    r"\.github/workflows/compile-time-bench\.yml$",
)


def compile_time_changed(
    config: inspect_changes.Config,
    dirty_files: Sequence[str],
    ignored_files: Sequence[str],
) -> bool:
    # Include benchmark inputs ignored by the main matrix, while continuing to
    # ignore documentation-only edits.
    all_files = [
        path for path in (*dirty_files, *ignored_files) if not path.endswith(".md")
    ]
    inputs = inspect_changes.compile_patterns(BENCHMARK_INPUTS)
    benchmark_changed = any(
        inspect_changes.matches_any(inputs, path) for path in all_files
    )
    project_files = inspect_changes.build_project_dirty_map(config, dirty_files)

    # Preserve the classifier's conservative fallback for unknown infrastructure.
    infrastructure_changed = bool(project_files[inspect_changes.CORE_PROJECT_KEY])
    public_changed = any(project_files[key] for key in PUBLIC_PROJECTS)
    # Package config, install rules, and header-test generation live beside the
    # public headers. Their changes also affect compile-time data.
    cmake_inputs = inspect_changes.compile_patterns(
        (
            r"(libcudacxx|cub|thrust|cudax)/(cmake/|CMakeLists\.txt$)",
            r"cccl-version\.json$",
        )
    )
    cmake_changed = any(
        inspect_changes.matches_any(cmake_inputs, path) for path in all_files
    )
    return (
        infrastructure_changed or public_changed or cmake_changed or benchmark_changed
    )


def main(argv: Sequence[str] | None = None) -> int:
    args = inspect_changes.parse_args(argv)
    config = inspect_changes.load_config(inspect_changes.CONFIG_PATH)
    dirty_files, ignored_files = inspect_changes.collect_dirty_files(args, config)
    enabled = compile_time_changed(config, dirty_files, ignored_files)
    inspect_changes.write_output("compile_time_changed", str(enabled).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
