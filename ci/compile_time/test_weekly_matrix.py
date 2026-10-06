#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import unittest

from ci.compile_time.weekly_matrix import (
    expand_matrix,
    previous_weekly_run,
)

CURRENT = {"id": 30, "created_at": "2026-09-29T12:00:00Z"}
PREVIOUS_SHA = "a" * 40


class WeeklyMatrixTest(unittest.TestCase):
    def test_previous_run(self) -> None:
        runs = [
            {
                "id": 30,
                "created_at": "2026-09-29T12:00:00Z",
                "event": "schedule",
                "head_branch": "main",
                "head_sha": "c" * 40,
            },
            {
                "id": 29,
                "created_at": "2026-09-27T08:17:15Z",
                "event": "schedule",
                "head_branch": "main",
                "head_sha": PREVIOUS_SHA,
                "conclusion": "failure",
            },
            {
                "id": 28,
                "created_at": "2026-09-28T12:00:00Z",
                "event": "workflow_dispatch",
                "head_branch": "main",
                "head_sha": "d" * 40,
            },
            {
                "id": 27,
                "created_at": "2026-09-20T08:16:36Z",
                "event": "schedule",
                "head_branch": "main",
                "head_sha": "e" * 40,
            },
        ]
        self.assertEqual(previous_weekly_run(runs, CURRENT, "main")["id"], 29)
        with self.assertRaisesRegex(ValueError, "no earlier scheduled weekly run"):
            previous_weekly_run(runs, CURRENT, "release/3.5")

    def test_previous_baseline(self) -> None:
        matrix = {
            "include": [
                {"id": "cccl", "name": "CCCL", "project": "cccl"},
                {"id": "matx", "name": "MatX", "project": "matx"},
            ]
        }
        expanded = expand_matrix(matrix, PREVIOUS_SHA, include_cccl=True)["include"]
        self.assertEqual(
            [entry["id"] for entry in expanded],
            ["cccl", "matx-previous-weekly"],
        )
        self.assertEqual(
            [entry["baseline_ref"] for entry in expanded[1:]],
            [PREVIOUS_SHA],
        )
        weekly = expand_matrix(matrix, PREVIOUS_SHA, include_cccl=False)["include"]
        self.assertEqual([entry["id"] for entry in weekly], ["matx-previous-weekly"])


if __name__ == "__main__":
    unittest.main()
