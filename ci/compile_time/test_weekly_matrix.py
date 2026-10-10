#!/usr/bin/env python3

# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import unittest
from unittest.mock import patch
from urllib.parse import parse_qs, urlparse

from ci.compile_time.weekly_matrix import (
    expand_matrix,
    main,
    previous_weekly_run,
)

CURRENT = {"id": 30, "created_at": "2026-09-29T12:00:00Z"}
PREVIOUS_SHA = "a" * 40


class WeeklyMatrixTest(unittest.TestCase):
    def test_cli_filters_and_paginates_workflow_history(self) -> None:
        matrix = {"include": [{"id": "matx", "name": "MatX", "project": "matx"}]}
        previous = {
            "id": 29,
            "created_at": "2026-09-27T08:17:15Z",
            "event": "schedule",
            "head_branch": "main",
            "head_sha": PREVIOUS_SHA,
        }
        manual = {
            **previous,
            "id": 31,
            "event": "workflow_dispatch",
            "created_at": "2026-09-28T12:00:00Z",
            "head_sha": "b" * 40,
        }
        other_branch = {
            **manual,
            "id": 32,
            "event": "schedule",
            "head_branch": "release/3.5",
        }
        future = {
            **previous,
            "id": 33,
            "created_at": "2026-10-04T12:00:00Z",
            "head_sha": "c" * 40,
        }

        for paginate in (False, True):
            with self.subTest(paginate=paginate):
                queries = []

                def response(request, **kwargs):
                    url = urlparse(request.full_url)
                    if url.path.endswith("/runs/30"):
                        return io.StringIO(json.dumps(CURRENT))
                    query = parse_qs(url.query)
                    queries.append(query)
                    self.assertTrue({"event", "branch", "created"}.isdisjoint(query))
                    if paginate and query["page"] == ["1"]:
                        runs = [manual] * 100
                    else:
                        runs = [future, manual, other_branch, previous]
                    return io.StringIO(json.dumps({"workflow_runs": runs}))

                output = io.StringIO()
                diagnostics = io.StringIO()
                with (
                    patch.dict("os.environ", {"GH_TOKEN": "test-token"}),
                    patch(
                        "sys.argv",
                        [
                            "weekly_matrix.py",
                            "--repository",
                            "NVIDIA/cccl",
                            "--run-id",
                            "30",
                            "--branch",
                            "main",
                        ],
                    ),
                    patch("sys.stdin", io.StringIO(json.dumps(matrix))),
                    patch("sys.stdout", output),
                    patch("sys.stderr", diagnostics),
                    patch(
                        "ci.compile_time.weekly_matrix.urlopen", side_effect=response
                    ) as api,
                ):
                    main()

                self.assertEqual(api.call_count, 3 if paginate else 2)
                self.assertEqual(queries[-1]["page"], ["2" if paginate else "1"])
                self.assertEqual(
                    json.loads(output.getvalue())["include"][0]["baseline_ref"],
                    PREVIOUS_SHA,
                )
                self.assertIn(
                    f"run 29 ({previous['created_at']}), commit {PREVIOUS_SHA}",
                    diagnostics.getvalue(),
                )

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
