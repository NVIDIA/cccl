#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Combine JSON dispatch matrices saved by the compile-time and SASS parsers."""

import argparse
import json
import sys
from pathlib import Path


def merge_matrix_files(matrix_files):
    deduped_jobs = {}
    prefix_fields = ("id", "comment_header")

    for i, matrix in enumerate(matrix_files):
        matrix_label, _, matrix_path = (
            matrix.partition(":") if ":" in matrix else (str(i), "", matrix)
        )
        with Path(matrix_path).open() as stream:
            matrix_jobs = json.load(stream).get("include", [])

        # A single job file doesn't need deduping or prefixing
        if len(matrix_files) == 1:
            return {"include": matrix_jobs}

        for job in matrix_jobs:
            # Dedupe jobs based on content excluding id and comment header
            job_key = frozenset(
                (k, v) for k, v in job.items() if k not in prefix_fields
            )
            if job_key in deduped_jobs:
                continue

            # Prefix the job id and comment_header
            for field in prefix_fields:
                if job_field := job.get(field):
                    job[field] = f"{matrix_label}.{job_field}"
            deduped_jobs[job_key] = job

    return {"include": list(deduped_jobs.values())}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("matrix_files", nargs="+")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()
    matrix = merge_matrix_files(args.matrix_files)

    with args.output.open("w") if args.output else sys.stdout as stream:
        json.dump(matrix, stream, separators=(",", ":"))
        stream.write("\n")


if __name__ == "__main__":
    main()
