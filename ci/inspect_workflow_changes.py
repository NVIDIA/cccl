#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
"""Select PR integration jobs from their inputs before allocating runners."""

from __future__ import annotations

from typing import Sequence

import inspect_changes

# These workflows consume the installed C++ headers, not the Python packages or
# the C++ libraries' tests. Reuse their ownership from the matrix classifier.
PUBLIC_PROJECTS = ("libcudacxx_public", "cub_public", "thrust_public", "cudax_public")
JOB_INPUTS = {
    "rapids": (r"ci/rapids/", r"\.github/workflows/build-rapids\.yml$"),
    "matx": (r"ci/matx/", r"\.github/workflows/build-matx\.yml$"),
    "pytorch": (r"ci/pytorch/", r"\.github/workflows/build-pytorch\.yml$"),
    "compile_time": (
        r"ci/compile_time/",
        r"ci/build_compile_time_bench\.sh$",
        r"\.github/workflows/compile-time-bench\.yml$",
    ),
}


def select_workflows(
    config: inspect_changes.Config,
    dirty_files: Sequence[str],
    ignored_files: Sequence[str],
) -> dict[str, bool]:
    # Integration workflow inputs are intentionally ignored by the main matrix.
    # Include them here, while continuing to ignore documentation-only edits.
    all_files = [
        path for path in (*dirty_files, *ignored_files) if not path.endswith(".md")
    ]
    inputs = {
        job: inspect_changes.compile_patterns(patterns)
        for job, patterns in JOB_INPUTS.items()
    }
    job_files = {
        job: [path for path in all_files if inspect_changes.matches_any(patterns, path)]
        for job, patterns in inputs.items()
    }
    owned_files = {path for paths in job_files.values() for path in paths}
    project_files = inspect_changes.build_project_dirty_map(
        config, [path for path in dirty_files if path not in owned_files]
    )

    # Preserve the classifier's conservative fallback for unknown infrastructure.
    infrastructure_changed = bool(project_files[inspect_changes.CORE_PROJECT_KEY])
    public_changed = {key for key in PUBLIC_PROJECTS if project_files[key]}
    # Package config, install rules, and header-test generation live beside the
    # public headers. Their changes also affect consumers and compile-time data.
    cmake_inputs = inspect_changes.compile_patterns(
        (
            r"(libcudacxx|cub|thrust|cudax)/(cmake/|CMakeLists\.txt$)",
            r"cccl-version\.json$",
        )
    )
    cmake_changed = any(
        inspect_changes.matches_any(cmake_inputs, path) for path in all_files
    )
    shared_changed = infrastructure_changed or cmake_changed

    return {
        "rapids": shared_changed or bool(public_changed) or bool(job_files["rapids"]),
        "matx": shared_changed or bool(public_changed) or bool(job_files["matx"]),
        "pytorch": shared_changed or bool(public_changed) or bool(job_files["pytorch"]),
        "compile_time": shared_changed
        or bool(public_changed - {"cudax_public"})
        or bool(job_files["compile_time"]),
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = inspect_changes.parse_args(argv)
    config = inspect_changes.load_config(inspect_changes.CONFIG_PATH)
    dirty_files, ignored_files = inspect_changes.collect_dirty_files(args, config)
    for job, enabled in select_workflows(config, dirty_files, ignored_files).items():
        inspect_changes.write_output(f"{job}_changed", str(enabled).lower())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
