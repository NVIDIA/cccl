#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine independently generated workflows and rebuild their summaries."""

import argparse
import json
import runpy
from pathlib import Path


def merge_workflows(input_dirs, output_dir):
    # Reuse the parser's finalization and output helpers without loading a matrix.
    builder = runpy.run_path(str(Path(__file__).with_name("build-workflow.py")))
    workflow = {}
    overrides = []
    for input_dir in input_dirs:
        with (input_dir / "workflow.json").open() as stream:
            groups = json.load(stream)
        # Finalized groups omit empty arrays; restore them before combining.
        for group in groups.values():
            group.setdefault("standalone", [])
            group.setdefault("two_stage", [])
        builder["merge_dispatch_groups"](workflow, groups)

        override_file = input_dir / "override.json"
        if override_file.is_file():
            with override_file.open() as stream:
                overrides.extend(json.load(stream))

    # Deduplicate shared builds and tests, then assign IDs across the whole run.
    workflow = builder["finalize_workflow_dispatch_groups"](workflow)
    builder["write_outputs"](workflow, output_dir)
    if overrides:
        builder["write_override_matrix"](overrides, output_dir)
    else:
        (output_dir / "override.json").unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("input_dirs", type=Path, nargs="+")
    parser.add_argument("--output-dir", type=Path, default=Path("workflow"))
    args = parser.parse_args()
    merge_workflows(args.input_dirs, args.output_dir)


if __name__ == "__main__":
    main()
