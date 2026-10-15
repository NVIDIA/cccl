#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ci.compile_time import render_pr_comment
from ci.compile_time.test_summarize_events import (
    REPO_ROOT,
    SUMMARY_SCRIPT,
    TraceBuilder,
    csv_rows,
)


class ReportStabilityTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.builder = TraceBuilder(REPO_ROOT)

    def pair(
        self,
        index,
        *,
        baseline=100_000_000,
        current=100_000_000,
        headers=(),
        root_tu=None,
    ):
        for side, total, offset in (("baseline", baseline, 0), ("current", current, 1)):
            events = [self.builder.event("Driver", "", 0, total)]
            position = 10
            for name, before, after, occurrences in headers:
                duration = (before, after)[offset]
                occurrences = (
                    occurrences[offset]
                    if isinstance(occurrences, tuple)
                    else occurrences
                )
                for occurrence in range(occurrences):
                    # Partition integer duration without changing its TU total.
                    cost = duration // occurrences + (
                        occurrence < duration % occurrences
                    )
                    events.append(
                        self.builder.event(
                            "Processing Header File",
                            self.builder.project_detail(name),
                            position,
                            cost,
                        )
                    )
                    position += cost + 1
            self.builder.write_trace(
                self.root / side / f"{index}.json", events, root_tu or str(index)
            )

    def summarize(self, *, jobs=1, threshold=0.2, extra_slices=()):
        output = self.root / f"reports-{jobs}"
        slices = self.root / "slices.json"
        slices.write_text(
            json.dumps(
                {
                    "slices": [
                        {
                            "id": "files",
                            "title": "Files",
                            "filter": "file-processing",
                            "timing": "exclusive",
                            "sort": "total",
                            "top": 15,
                            "threshold": threshold,
                        }
                    ]
                    + list(extra_slices)
                }
            )
        )
        subprocess.run(
            [
                sys.executable,
                str(SUMMARY_SCRIPT),
                str(self.root / "current"),
                "--baseline-dir",
                str(self.root / "baseline"),
                "--slices",
                str(slices),
                "--stability-filter",
                "--jobs",
                str(jobs),
                "-o",
                str(output),
            ],
            check=True,
            capture_output=True,
        )
        return json.loads((output / "summary.json").read_text())

    def test_drift_is_removed_from_diagnostics_and_kept_in_headline(self):
        for i in range(48):
            headers = [
                ("libcudacxx/include/cuda/std/common.h", 5_000_000, 5_500_000, 50)
            ]
            if i < 24:
                headers.append(("cub/cub/specific.cuh", 2_000_000, 2_300_000, 1))
            self.pair(i, current=110_000_000, headers=headers)
        summary = self.summarize()
        overall = summary["overall"]
        self.assertEqual(overall["delta_s"], "480.000000")
        self.assertEqual(overall["matched_tu_count"], 48)
        comparison = summary["slices"][0]["comparison"]
        self.assertAlmostEqual(comparison["drift_factor"], 1.1)
        rows = comparison["worse"]["rows"]
        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["event_key"], "cub/cub/specific.cuh")
        self.assertEqual(rows[0]["adjusted_delta_s"], "2.400000")
        self.assertEqual(rows[0]["impact_magnitude_s"], "7.200000")
        self.assertEqual(rows[0]["stability"], "consistent")
        complete = csv_rows(Path(comparison["all_csv"]))
        self.assertEqual(len(complete), 2)
        common = next(row for row in complete if "common.h" in row["event_key"])
        self.assertEqual(common["impact_delta_s"], "24.000000")
        self.assertEqual(common["adjusted_delta_s"], "0.000000")
        self.assertEqual(common["matched_tu_count"], "48")
        self.assertEqual(common["baseline_event_count"], "2400")
        rendered = render_pr_comment.render_comment(
            summary, {"id": "example"}, artifacts_url="https://example.com/artifacts"
        )
        self.assertIn("+480.000 s", rendered)
        self.assertIn("specific.cuh", rendered)
        self.assertNotIn("common.h", rendered)
        self.assertIn("independent TU directions", rendered)
        self.assertNotIn("confirmation", summary)
        self.assertEqual(summary["consistency_test"]["hypothesis_count"], 4)
        self.assertLessEqual(rows[0]["consistency_adjusted_pvalue"], 0.05)

    def test_inconsistent_small_tu_movements_do_not_accumulate_into_alert(self):
        deltas = [
            -150000,
            -140000,
            -130000,
            -120000,
            -110000,
            20000,
            30000,
            140000,
            150000,
            160000,
            170000,
            190000,
        ]
        deltas *= 2
        self.assertGreater(sum(deltas), 200000)
        for i, delta in enumerate(deltas):
            self.pair(
                i, headers=[("cub/cub/noisy.cuh", 1_000_000, 1_000_000 + delta, 100)]
            )
        comparison = self.summarize()["slices"][0]["comparison"]
        self.assertEqual(comparison["worse"]["rows"], [])
        self.assertEqual(comparison["worse"]["common_headers"], [])
        row = csv_rows(Path(comparison["all_csv"]))[0]
        self.assertEqual(row["stability"], "inconsistent")
        self.assertEqual(row["matched_tu_count"], "24")

    def test_localized_and_sparse_changes_are_visible_without_consistency_claim(self):
        for i in range(24):
            self.pair(
                i,
                headers=[
                    (
                        "cub/cub/local.cuh",
                        1_000_000,
                        1_000_000 + (300000 if i == 0 else 0),
                        1,
                    )
                ]
                + ([("cub/cub/sparse.cuh", 1_000_000, 1_300_000, 1)] if i == 0 else []),
            )
        comparison = self.summarize()["slices"][0]["comparison"]
        unconfirmed = {
            row["event_key"]: row for row in comparison["worse"]["unconfirmed"]
        }
        self.assertEqual(
            unconfirmed["cub/cub/sparse.cuh"]["stability"], "insufficient-tus"
        )
        self.assertEqual(unconfirmed["cub/cub/local.cuh"]["stability"], "localized")
        self.assertEqual(comparison["worse"]["rows"], [])
        self.assertEqual(comparison["worse"]["common_headers"], [])

    def test_exposure_cannot_manufacture_consistency(self):
        # Raw TU changes have median 60 ms and MAD zero, solely because
        # occurrence counts differ. Per-occurrence changes have median
        # 250 us and MAD 1250 us, and must not become a diagnostic alert.
        costs = [
            -1500,
            -1400,
            -1300,
            -1200,
            -1100,
            200,
            300,
            1500,
            1500,
            1500,
            1500,
            1500,
        ]
        counts = [20, 20, 20, 20, 20, 300, 200, 40, 40, 40, 40, 40]
        costs *= 2
        counts *= 2
        for i, (cost, count) in enumerate(zip(costs, counts)):
            self.pair(
                i,
                headers=[
                    (
                        "cub/cub/exposed.cuh",
                        10000 * count,
                        (10000 + cost) * count,
                        count,
                    )
                ],
            )
        comparison = self.summarize()["slices"][0]["comparison"]
        self.assertEqual(comparison["worse"]["rows"], [])
        self.assertEqual(comparison["worse"]["common_headers"], [])
        row = csv_rows(Path(comparison["all_csv"]))[0]
        self.assertEqual(row["adjusted_delta_s"], "0.580000")
        self.assertEqual(row["median_tu_delta_s"], "0.060000")
        self.assertEqual(row["mad_tu_delta_s"], "0.000000")
        self.assertEqual(row["median_change_per_baseline_event_s"], "0.000250000")
        self.assertEqual(row["mad_change_per_baseline_event_s"], "0.001250000")
        self.assertEqual(row["stability"], "inconsistent")

    def test_increased_occurrence_count_is_not_hidden_by_average_cost(self):
        for i in range(48):
            self.pair(
                i,
                headers=[("cub/cub/extra.cuh", 100000, 200000, (1, 2))]
                if i < 24
                else [],
            )
        comparison = self.summarize()["slices"][0]["comparison"]
        row = comparison["worse"]["rows"][0]
        self.assertEqual(row["adjusted_delta_s"], "2.400000")
        self.assertEqual(row["baseline_avg_per_event_s"], "0.100000000")
        self.assertEqual(row["current_avg_per_event_s"], "0.100000000")
        self.assertEqual(row["baseline_event_count"], 24)
        self.assertEqual(row["current_event_count"], 48)
        self.assertEqual(row["stability"], "consistent")

    def test_four_context_change_remains_visible_as_inspection_candidate(self):
        for i in range(12):
            self.pair(
                i,
                headers=[("cub/cub/small-panel.cuh", 1000000, 1080000, 1)]
                if i < 4
                else [],
            )
        comparison = self.summarize()["slices"][0]["comparison"]
        self.assertEqual(comparison["worse"]["rows"], [])
        row = comparison["worse"]["unconfirmed"][0]
        self.assertEqual(row["impact_delta_s"], "0.320000")
        self.assertEqual(row["matched_tu_count"], 4)
        self.assertEqual(row["stability"], "insufficient-tus")
        self.assertGreater(row["consistency_adjusted_pvalue"], 0.05)

    def test_trace_fragments_of_one_tu_are_one_observation(self):
        for i in range(4):
            self.pair(
                i,
                headers=[("cub/cub/shared.cuh", 1_000_000, 1_300_000, 100)],
                root_tu="same",
            )
        summary = self.summarize()
        self.assertEqual(summary["overall"]["matched_tu_count"], 1)
        self.assertEqual(summary["overall"]["matched_trace_count"], 4)
        row = summary["slices"][0]["comparison"]["worse"]["unconfirmed"][0]
        self.assertEqual(row["matched_tu_count"], 1)
        self.assertEqual(row["adjusted_delta_s"], "1.200000")
        self.assertEqual(row["stability"], "insufficient-tus")
        self.assertTrue(summary["slices"][0]["warnings"])

    def test_threshold_selection_does_not_reduce_hypothesis_family(self):
        for i in range(24):
            self.pair(i, headers=[("cub/cub/changed.cuh", 100000, 120000, 1)])
        first = self.summarize()
        row = first["slices"][0]["comparison"]["worse"]["common_headers"][0]
        self.assertLessEqual(row["consistency_adjusted_pvalue"], 0.05)
        for i in range(24):
            self.pair(
                i,
                headers=[("cub/cub/changed.cuh", 100000, 120000, 1)]
                + [(f"cub/cub/unchanged-{j}.cuh", 1000, 1000, 1) for j in range(100)],
            )
        second = self.summarize()
        self.assertEqual(second["consistency_test"]["hypothesis_count"], 202)
        comparison = second["slices"][0]["comparison"]
        self.assertEqual(comparison["worse"]["rows"], [])
        self.assertEqual(comparison["worse"]["common_headers"], [])
        row = next(
            r
            for r in csv_rows(Path(comparison["all_csv"]))
            if "changed.cuh" in r["event_key"]
        )
        self.assertGreater(float(row["consistency_adjusted_pvalue"]), 0.05)
        self.assertEqual(float(row["consistency_pvalue"]), 0.75**24)

    def test_drift_direction_reversal_does_not_create_statistical_claim(self):
        for i in range(24):
            self.pair(
                i,
                current=110000000,
                headers=[("cub/cub/reversed.cuh", 1000000, 1050000, 1)],
            )
        comparison = self.summarize()["slices"][0]["comparison"]
        self.assertEqual(comparison["better"]["rows"], [])
        self.assertEqual(comparison["worse"]["rows"], [])
        row = csv_rows(Path(comparison["all_csv"]))[0]
        self.assertEqual(row["positive_tus"], "24")
        self.assertEqual(row["impact_delta_s"], "1.200000")
        self.assertEqual(row["adjusted_delta_s"], "-1.200000")
        self.assertEqual(float(row["consistency_adjusted_pvalue"]), 1)

    def test_hypothesis_family_spans_nested_and_thresholded_slices(self):
        for i in range(24):
            self.pair(i, headers=[("cub/cub/change.cuh", 100000, 120000, 1)])
        ordinary = self.summarize()
        child = {
            "id": "nested",
            "title": "Nested",
            "filter": "file-processing",
            "timing": "exclusive",
            "sort": "total",
            "top": 1,
            "threshold": 100,
        }
        extra = dict(child, id="extra", children=[child])
        combined = self.summarize(extra_slices=[extra])
        self.assertEqual(combined["consistency_test"]["hypothesis_count"], 6)
        first = ordinary["slices"][0]["comparison"]["worse"]["common_headers"][0]
        second = combined["slices"][0]["comparison"]["worse"]["common_headers"][0]
        self.assertEqual(second["consistency_pvalue"], first["consistency_pvalue"])
        self.assertEqual(
            second["consistency_adjusted_pvalue"],
            3 * first["consistency_adjusted_pvalue"],
        )
        self.assertEqual(combined["slices"][1]["comparison"]["worse"]["rows"], [])

    def test_parallel_analysis_is_deterministic(self):
        for i in range(8):
            self.pair(i, headers=[("cub/cub/common.cuh", 1_000_000, 1_300_000, 2)])
        serial, parallel = self.summarize(jobs=1), self.summarize(jobs=2)
        self.assertEqual(serial["overall"], parallel["overall"])
        for direction in ("worse", "better"):
            for key in ("rows", "common_headers"):
                self.assertEqual(
                    serial["slices"][0]["comparison"][direction][key],
                    parallel["slices"][0]["comparison"][direction][key],
                )

    def test_comment_has_global_budgets_and_preserves_warnings(self):
        row = {
            "event_name": "Function",
            "event_key": "cuda::f",
            "impact_delta_s": "1.0",
            "adjusted_delta_s": "0.9",
            "matched_tu_count": 4,
            "stability": "consistent",
        }
        summary = {
            "overall": {
                "baseline_s": "10",
                "current_s": "11",
                "delta_s": "1",
                "relative_delta_pct": 10,
                "matched_tu_count": 4,
                "matched_trace_count": 4,
                "stability_filter": True,
            },
            "slices": [
                {
                    "filter": "functions",
                    "warnings": ["missing traces"],
                    "comparison": {
                        "worse": {
                            "rows": [
                                dict(row, event_key=f"cuda::f{i}") for i in range(30)
                            ]
                        },
                        "better": {
                            "rows": [
                                dict(
                                    row,
                                    event_key=f"cuda::b{i}",
                                    adjusted_delta_s="-0.5",
                                )
                                for i in range(30)
                            ]
                        },
                    },
                }
            ],
        }
        comment = render_pr_comment.render_comment(
            summary, {"id": "test"}, artifacts_url="https://example.com"
        )
        self.assertEqual(comment.count("Reliable across TU contexts |"), 8)
        self.assertIn("missing traces", comment)
        self.assertIn("<summary>Largest improvement candidates</summary>", comment)

    def test_invalid_stability_mode_requires_baseline(self):
        completed = subprocess.run(
            [sys.executable, str(SUMMARY_SCRIPT), str(self.root), "--stability-filter"],
            capture_output=True,
            text=True,
        )
        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("requires --baseline-dir", completed.stderr)

    def test_zero_deltas_are_retained_in_full_csv(self):
        self.pair(0, headers=[("cub/cub/same.cuh", 1_000_000, 1_000_000, 1)])
        comparison = self.summarize()["slices"][0]["comparison"]
        self.assertEqual(comparison["all_row_count"], 1)
        self.assertEqual(
            csv_rows(Path(comparison["all_csv"]))[0]["impact_delta_s"], "0.000000"
        )


if __name__ == "__main__":
    unittest.main()
