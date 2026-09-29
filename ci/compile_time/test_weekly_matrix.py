#!/usr/bin/env python3

import unittest

from ci.compile_time.weekly_matrix import (
    expand_matrix,
    latest_cccl_release,
    previous_weekly_run,
    tag_commit_sha,
)

CURRENT = {"id": 30, "created_at": "2026-09-29T12:00:00Z"}
PREVIOUS_SHA = "a" * 40
RELEASE_SHA = "b" * 40


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

    def test_release_filter(self) -> None:
        releases = [
            {
                "tag_name": "python-1.2.1",
                "published_at": "2026-09-28T20:40:20Z",
            },
            {
                "tag_name": "v3.5.0-rc2",
                "published_at": "2026-09-10T00:00:00Z",
                "prerelease": True,
            },
            {
                "tag_name": "v3.5.0",
                "published_at": "2026-10-01T00:00:00Z",
            },
            {
                "tag_name": "v3.4.2",
                "published_at": "2026-08-05T18:54:34Z",
            },
            {
                "tag_name": "v3.4.1",
                "published_at": "2026-08-05T18:54:15Z",
            },
        ]
        self.assertEqual(latest_cccl_release(releases, CURRENT)["tag_name"], "v3.4.2")
        with self.assertRaisesRegex(ValueError, "no published final CCCL release"):
            latest_cccl_release(releases[:3], CURRENT)

    def test_annotated_tag_chain(self) -> None:
        outer = "1" * 40
        inner = "2" * 40
        objects = {
            "repos/NVIDIA/cccl/git/ref/tags/v3.4.2": {
                "object": {"type": "tag", "sha": outer}
            },
            f"repos/NVIDIA/cccl/git/tags/{outer}": {
                "object": {"type": "tag", "sha": inner}
            },
            f"repos/NVIDIA/cccl/git/tags/{inner}": {
                "object": {"type": "commit", "sha": RELEASE_SHA}
            },
        }
        self.assertEqual(
            tag_commit_sha(objects.__getitem__, "NVIDIA/cccl", "v3.4.2"),
            RELEASE_SHA,
        )

    def test_dual_baselines(self) -> None:
        matrix = {
            "include": [
                {"id": "cccl", "name": "CCCL", "project": "cccl"},
                {"id": "matx", "name": "MatX", "project": "matx"},
            ]
        }
        expanded = expand_matrix(
            matrix, PREVIOUS_SHA, RELEASE_SHA, "v3.4.2", include_cccl=True
        )["include"]
        self.assertEqual(
            [entry["id"] for entry in expanded],
            ["cccl", "matx-previous-weekly", "matx-latest-release"],
        )
        self.assertEqual(
            [entry["baseline_ref"] for entry in expanded[1:]],
            [PREVIOUS_SHA, RELEASE_SHA],
        )


if __name__ == "__main__":
    unittest.main()
