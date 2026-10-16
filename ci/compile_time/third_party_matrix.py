#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import argparse
import json
from pathlib import Path

from ci.compile_time.parse_matrix import (
    die,
    load_matrix,
    require_id,
    require_mapping,
    require_string,
    validate_slice,
)


def parse_matrix(
    path: Path, workflow: str, *, opt_in: bool = False, skips: tuple[str, ...] = ()
) -> dict:
    data = load_matrix(path)
    section = require_mapping(data.get("compile_time", {}), "compile_time")
    override = section.get("third_party_override") or []
    configs = section.get("third_party", [])
    if workflow == "pull_request":
        configs = override or (configs if opt_in else [])
    if not isinstance(configs, list):
        die("compile_time.third_party must be a list")
    seen = set()
    include = []
    for index, raw in enumerate(configs):
        where = f"compile_time.third_party[{index}]"
        config = require_mapping(raw, where)
        config_id = require_id(config.get("id"), f"{where}.id")
        if config_id in seen:
            die(f"duplicate config id {config_id}")
        seen.add(config_id)
        project = config.get("project")
        if project not in ("matx", "pytorch", "rapids"):
            die(f"{where}.project must be matx, pytorch, or rapids")
        if project in skips or "all" in skips:
            continue
        targets = config.get("targets", [])
        if not isinstance(targets, list) or any(
            not isinstance(t, str) or not t for t in targets
        ):
            die(f"{where}.targets must be a list of non-empty strings")
        if targets and project != "rapids":
            die(f"{where}.targets are only supported for RAPIDS")
        slices = config.get("slices")
        if not isinstance(slices, list) or not slices:
            die(f"{where}.slices must be a non-empty list")
        slice_ids = set()
        slices = [
            validate_slice(s, where=f"{where}.slices[{i}]", seen_ids=slice_ids)
            for i, s in enumerate(slices)
        ]
        jobs = config.get("summary_jobs", 8)
        if isinstance(jobs, bool) or not isinstance(jobs, int) or jobs < 1:
            die(f"{where}.summary_jobs must be positive")
        include.append(
            {
                "id": config_id,
                "project": project,
                "name": require_string(config.get("name"), f"{where}.name"),
                "runner": require_string(config.get("runner"), f"{where}.runner"),
                "launch_args": require_string(
                    config.get("launch_args"), f"{where}.launch_args"
                ),
                "targets_json": json.dumps(targets, separators=(",", ":")),
                "slices_json": json.dumps({"slices": slices}, separators=(",", ":")),
                "summary_jobs": jobs,
            }
        )
    return {"include": include}


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Select weekly third-party trace builds and PR opt-ins."
    )
    parser.add_argument("matrix", type=Path)
    parser.add_argument("--workflow", choices=("weekly", "pull_request"), required=True)
    parser.add_argument("--opt-in", action="store_true")
    parser.add_argument("--skip", action="append", default=[])
    args = parser.parse_args()
    print(
        json.dumps(
            parse_matrix(
                args.matrix, args.workflow, opt_in=args.opt_in, skips=tuple(args.skip)
            )
        )
    )


if __name__ == "__main__":
    main()
