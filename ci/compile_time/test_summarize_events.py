#!/usr/bin/env python3

import csv
import json
import os
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ci.compile_time import summarize_tus

REPO_ROOT = Path(__file__).resolve().parents[2]
SUMMARY_SCRIPT = REPO_ROOT / "ci" / "compile_time" / "summarize_events.py"
PREPARE_SCRIPT = REPO_ROOT / "ci" / "compile_time" / "prepare_traces.py"
PARSE_MATRIX_SCRIPT = REPO_ROOT / "ci" / "compile_time" / "parse_matrix.py"
RENDER_COMMENT_SCRIPT = REPO_ROOT / "ci" / "compile_time" / "render_pr_comment.py"
WRAPPER_SCRIPT = REPO_ROOT / "ci" / "build_compile_time_bench.sh"


def csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


class TraceBuilder:
    def __init__(self, root: Path):
        self.root = root

    def project_detail(self, rel: str) -> str:
        return (self.root / rel).as_posix()

    def event(
        self,
        name: str,
        detail: str,
        ts: int,
        dur: int,
        *,
        pid: int = 1,
        tid: int = 1,
    ) -> dict:
        return {
            "ph": "X",
            "pid": pid,
            "tid": tid,
            "name": name,
            "ts": ts,
            "dur": dur,
            "args": {"detail": detail},
        }

    def write_trace(self, path: Path, events: list[dict], input_name: str) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(
                {
                    "traceEvents": events,
                    "otherData": {
                        "inputFiles": [f"/generated/headers/target/{input_name}.cu"]
                    },
                }
            ),
            encoding="utf-8",
        )


class SummarizeEventsBaselineCompareTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.work = Path(self.tempdir.name)
        self.traces = TraceBuilder(REPO_ROOT)
        self.wrapper_infix = f"compile-time-test-{self.work.name}"
        self.wrapper_preset = "all-dev"
        self.wrapper_build_dir = (
            REPO_ROOT / "build" / self.wrapper_infix / self.wrapper_preset
        )

    def tearDown(self) -> None:
        shutil.rmtree(REPO_ROOT / "build" / self.wrapper_infix, ignore_errors=True)
        self.tempdir.cleanup()

    def comparison_csv(self, output_dir: Path, name: str) -> Path:
        return output_dir / "comparison" / name

    def wrapper_trace_dir(self) -> Path:
        return self.wrapper_build_dir / "compile_time" / "raw_traces"

    def wrapper_output_dir(self) -> Path:
        return self.wrapper_build_dir / "compile_time" / "event_reports"

    def assert_empty_csv(self, path: Path) -> None:
        self.assertTrue(path.exists())
        self.assertEqual(csv_rows(path), [])

    def run_summary(
        self,
        current: Path,
        baseline: Path,
        output_dir: Path,
        *args: str,
        baseline_repo_root: Path | None = None,
    ) -> subprocess.CompletedProcess[str]:
        command = [
            sys.executable,
            SUMMARY_SCRIPT.as_posix(),
            current.as_posix(),
            "--baseline-dir",
            baseline.as_posix(),
            "-o",
            output_dir.as_posix(),
        ]
        if baseline_repo_root is not None:
            command.extend(["--baseline-repo-root", baseline_repo_root.as_posix()])
        command.extend(args)

        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def run_wrapper(
        self,
        *args: str,
        no_tu_csv: bool = True,
        common_args: tuple[str, ...] = (),
    ) -> subprocess.CompletedProcess[str]:
        command = [
            "bash",
            WRAPPER_SCRIPT.as_posix(),
            "-skip-configure",
            "-skip-build",
            "-no-prepare-perfetto",
            *common_args,
        ]
        if no_tu_csv:
            command.append("-no-tu-csv")
        command.extend(["--", *args])
        env = os.environ.copy()
        env["CCCL_BUILD_INFIX"] = self.wrapper_infix
        return subprocess.run(
            command,
            cwd=REPO_ROOT,
            env=env,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

    def test_comparison_outputs_and_unmatched_children(self) -> None:
        baseline = self.work / "baseline"
        current = self.work / "current"
        output = self.work / "reports"

        parent = self.traces.project_detail("libcudacxx/include/cuda/std/parent.h")
        matched_child = self.traces.project_detail(
            "libcudacxx/include/cuda/std/matched_child.h"
        )
        better = self.traces.project_detail("libcudacxx/include/cuda/std/better.h")
        baseline_only_child = self.traces.project_detail(
            "libcudacxx/include/cuda/std/baseline_only_child.h"
        )
        baseline_only = self.traces.project_detail(
            "libcudacxx/include/cuda/std/only_baseline.h"
        )
        current_only = self.traces.project_detail(
            "libcudacxx/include/cuda/std/only_current.h"
        )

        self.traces.write_trace(
            baseline / "target" / "match.json",
            [
                self.traces.event("Parent", parent, 0, 100),
                self.traces.event("Child", matched_child, 10, 30),
                self.traces.event("Child", baseline_only_child, 50, 20),
                self.traces.event("Better", better, 200, 70),
                self.traces.event("OnlyBaseline", baseline_only, 300, 20),
            ],
            "match",
        )
        self.traces.write_trace(
            current / "target" / "match.json",
            [
                self.traces.event("Parent", parent, 0, 150),
                self.traces.event("Child", matched_child, 10, 40),
                self.traces.event("Better", better, 200, 40),
                self.traces.event("OnlyCurrent", current_only, 300, 90),
            ],
            "match",
        )
        self.traces.write_trace(
            baseline / "target" / "baseline_only_trace.json",
            [self.traces.event("Parent", parent, 0, 999)],
            "baseline_only_trace",
        )
        self.traces.write_trace(
            current / "target" / "current_only_trace.json",
            [self.traces.event("Parent", parent, 0, 999)],
            "current_only_trace",
        )

        self.run_summary(
            current,
            baseline,
            output,
            "-f",
            "all",
            "-e",
            "--sort",
            "total",
            "-n",
            "10",
            "--tag",
            "synthetic",
        )

        worse = csv_rows(
            self.comparison_csv(
                output, "top-10-all-exclusive-all-by-total-worse-synthetic.csv"
            )
        )
        better_rows = csv_rows(
            self.comparison_csv(
                output, "top-10-all-exclusive-all-by-total-better-synthetic.csv"
            )
        )

        parent_row = next(row for row in worse if row["event_name"] == "Parent")
        self.assertEqual(parent_row["baseline_selected_s"], "0.000070")
        self.assertEqual(parent_row["current_selected_s"], "0.000110")
        self.assertEqual(parent_row["impact_magnitude_s"], "0.000040")

        better_row = next(row for row in better_rows if row["event_name"] == "Better")
        self.assertEqual(better_row["baseline_selected_s"], "0.000070")
        self.assertEqual(better_row["current_selected_s"], "0.000040")
        self.assertEqual(better_row["impact_magnitude_s"], "0.000030")

        all_comparison_keys = {row["event_key"] for row in worse + better_rows}
        self.assertFalse(any("only_baseline" in key for key in all_comparison_keys))
        self.assertFalse(any("only_current" in key for key in all_comparison_keys))

    def test_comparison_threshold_filters_small_changes(self) -> None:
        baseline = self.work / "baseline"
        current = self.work / "current"
        output = self.work / "reports"

        small = self.traces.project_detail("libcudacxx/include/cuda/std/small.h")
        large = self.traces.project_detail("libcudacxx/include/cuda/std/large.h")
        self.traces.write_trace(
            baseline / "target" / "match.json",
            [
                self.traces.event("Same", small, 0, 10),
                self.traces.event("Same", large, 100, 10),
            ],
            "match",
        )
        self.traces.write_trace(
            current / "target" / "match.json",
            [
                self.traces.event("Same", small, 0, 12),
                self.traces.event("Same", large, 100, 20),
            ],
            "match",
        )

        self.run_summary(
            current,
            baseline,
            output,
            "-f",
            "all",
            "-i",
            "--sort",
            "total",
            "-n",
            "10",
            "--threshold",
            "0.000005",
            "--tag",
            "threshold",
        )

        worse = csv_rows(
            self.comparison_csv(
                output, "top-10-all-inclusive-by-total-worse-threshold.csv"
            )
        )
        self.assertEqual(len(worse), 1)
        self.assertTrue(worse[0]["event_key"].endswith("large.h"))
        self.assertEqual(worse[0]["impact_magnitude_s"], "0.000010")

    def test_comparison_ranks_by_total_impact_not_selected_metric(self) -> None:
        baseline = self.work / "baseline"
        current = self.work / "current"
        output = self.work / "reports"

        repeated = self.traces.project_detail("libcudacxx/include/cuda/std/repeated.h")
        one_trace = self.traces.project_detail(
            "libcudacxx/include/cuda/std/one_trace.h"
        )
        self.traces.write_trace(
            baseline / "target" / "first.json",
            [
                self.traces.event("Same", repeated, 0, 10),
                self.traces.event("Same", one_trace, 100, 10),
            ],
            "first",
        )
        self.traces.write_trace(
            current / "target" / "first.json",
            [
                self.traces.event("Same", repeated, 0, 16),
                self.traces.event("Same", one_trace, 100, 20),
            ],
            "first",
        )
        self.traces.write_trace(
            baseline / "target" / "second.json",
            [self.traces.event("Same", repeated, 0, 10)],
            "second",
        )
        self.traces.write_trace(
            current / "target" / "second.json",
            [self.traces.event("Same", repeated, 0, 16)],
            "second",
        )

        self.run_summary(
            current,
            baseline,
            output,
            "-f",
            "all",
            "-i",
            "--sort",
            "max",
            "-n",
            "10",
            "--tag",
            "impact",
        )

        worse = csv_rows(
            self.comparison_csv(output, "top-10-all-inclusive-by-max-worse-impact.csv")
        )
        self.assertTrue(worse[0]["event_key"].endswith("repeated.h"))
        self.assertEqual(worse[0]["impact_magnitude_s"], "0.000012")
        self.assertEqual(worse[0]["selected_magnitude_s"], "0.000006")

    def test_threshold_requires_comparison_mode(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"
        same = self.traces.project_detail("libcudacxx/include/cuda/std/same.h")
        self.traces.write_trace(
            traces / "target" / "same.json",
            [self.traces.event("Same", same, 0, 10)],
            "same",
        )

        completed = subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "--threshold",
                "0.000001",
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn(
            "--threshold can only be used together with --baseline-dir",
            completed.stderr,
        )

    def test_file_processing_same_filter_keeps_unmatched_child_in_parent_cost(
        self,
    ) -> None:
        baseline = self.work / "baseline"
        current = self.work / "current"
        output = self.work / "reports"

        parent = self.traces.project_detail("libcudacxx/include/cuda/std/parent.h")
        matched_child = self.traces.project_detail(
            "libcudacxx/include/cuda/std/matched_child.h"
        )
        unmatched_child = self.traces.project_detail(
            "libcudacxx/include/cuda/std/baseline_only_child.h"
        )

        self.traces.write_trace(
            baseline / "target" / "same.json",
            [
                self.traces.event("Processing Header File", parent, 0, 100),
                self.traces.event("Processing Header File", matched_child, 10, 30),
                self.traces.event("Processing Header File", unmatched_child, 50, 20),
            ],
            "same",
        )
        self.traces.write_trace(
            current / "target" / "same.json",
            [
                self.traces.event("Processing Header File", parent, 0, 130),
                self.traces.event("Processing Header File", matched_child, 10, 40),
            ],
            "same",
        )

        self.run_summary(
            current,
            baseline,
            output,
            "-f",
            "file-processing",
            "-e",
            "--sort",
            "total",
            "-n",
            "5",
            "--tag",
            "file-processing",
        )
        worse = csv_rows(
            self.comparison_csv(
                output,
                "top-5-file-processing-exclusive-same-filter-by-total-worse-file-processing.csv",
            )
        )
        parent_row = next(row for row in worse if row["event_key"].endswith("parent.h"))
        self.assertEqual(parent_row["baseline_selected_s"], "0.000070")
        self.assertEqual(parent_row["current_selected_s"], "0.000090")

    def test_same_filter_exclusive_uses_direct_children_only(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        parent = self.traces.project_detail("libcudacxx/include/cuda/std/parent.h")
        nested = self.traces.project_detail("libcudacxx/include/cuda/std/nested.h")

        self.traces.write_trace(
            traces / "target" / "direct.json",
            [
                self.traces.event("Processing Header File", parent, 0, 100),
                self.traces.event("Scanning Function Body", "not-a-header", 10, 80),
                self.traces.event("Processing Header File", nested, 20, 10),
            ],
            "direct",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "file-processing",
                "-e",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "direct",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(
            output / "top-5-file-processing-exclusive-same-filter-by-total-direct.csv"
        )
        parent_row = next(row for row in rows if row["event_key"].endswith("parent.h"))
        self.assertEqual(parent_row["selected_total_s"], "0.000100")

    def test_empty_comparisons_still_write_csvs(self) -> None:
        baseline = self.work / "baseline"
        current = self.work / "current"
        output = self.work / "reports"

        foo = self.traces.project_detail("libcudacxx/include/cuda/std/foo.h")
        bar = self.traces.project_detail("libcudacxx/include/cuda/std/bar.h")
        self.traces.write_trace(
            baseline / "target" / "same.json",
            [self.traces.event("Foo", foo, 0, 10)],
            "same",
        )
        self.traces.write_trace(
            current / "target" / "same.json",
            [self.traces.event("Bar", bar, 0, 20)],
            "same",
        )
        self.run_summary(
            current,
            baseline,
            output,
            "-f",
            "all",
            "-i",
            "--sort",
            "total",
            "-n",
            "5",
            "--tag",
            "empty",
        )

        self.assert_empty_csv(
            self.comparison_csv(output, "top-5-all-inclusive-by-total-worse-empty.csv")
        )
        self.assert_empty_csv(
            self.comparison_csv(output, "top-5-all-inclusive-by-total-better-empty.csv")
        )
        with (output / "summary.json").open(encoding="utf-8") as f:
            manifest = json.load(f)
        self.assertIn(
            "no comparable event keys",
            " ".join(manifest["slices"][0]["warnings"]),
        )

    def test_wrapper_forwarding_writes_current_report(self) -> None:
        current = self.wrapper_trace_dir()
        output = self.wrapper_output_dir()

        same = self.traces.project_detail("libcudacxx/include/cuda/std/same.h")
        self.traces.write_trace(
            current / "target" / "current_name.json",
            [self.traces.event("Same", same, 0, 100)],
            "current_name",
        )

        self.run_wrapper(
            "-f",
            "all",
            "-i",
            "--sort",
            "total",
            "-n",
            "5",
            "--tag",
            "wrapper",
            common_args=("-arch", "70"),
        )

        self.assertTrue((output / "top-5-all-inclusive-by-total-wrapper.csv").exists())

    def test_comparison_matches_paths_from_distinct_repo_roots(self) -> None:
        baseline_repo = self.work / "baseline-src"
        baseline = self.work / "baseline"
        current = self.work / "current"
        output = self.work / "reports"

        baseline_traces = TraceBuilder(baseline_repo)
        baseline_same = baseline_traces.project_detail(
            "libcudacxx/include/cuda/std/same.h"
        )
        current_same = self.traces.project_detail("libcudacxx/include/cuda/std/same.h")
        self.traces.write_trace(
            current / "target" / "same.json",
            [self.traces.event("Same", current_same, 0, 12)],
            "same",
        )
        baseline_traces.write_trace(
            baseline / "target" / "same.json",
            [baseline_traces.event("Same", baseline_same, 0, 10)],
            "same",
        )

        self.run_summary(
            current,
            baseline,
            output,
            "-f",
            "all",
            "-i",
            "--sort",
            "total",
            "-n",
            "5",
            baseline_repo_root=baseline_repo,
        )

        worse = csv_rows(
            self.comparison_csv(output, "top-5-all-inclusive-by-total-worse.csv")
        )
        self.assertEqual(worse[0]["event_key"], "libcudacxx/include/cuda/std/same.h")
        self.assertEqual(worse[0]["impact_magnitude_s"], "0.000002")

    def test_wrapper_uses_computed_event_output_dir(self) -> None:
        current = self.wrapper_trace_dir()
        wrapper_default = self.wrapper_output_dir()
        explicit_output = self.work / "explicit-output"

        same = self.traces.project_detail("libcudacxx/include/cuda/std/same.h")
        self.traces.write_trace(
            current / "target" / "same.json",
            [self.traces.event("Same", same, 0, 12)],
            "same",
        )

        self.run_wrapper(
            f"--output-dir={explicit_output.as_posix()}",
            "-f",
            "all",
            "-i",
            "--sort",
            "total",
            "-n",
            "5",
            "--tag",
            "equals-output",
        )

        self.assertTrue(
            (
                wrapper_default / "top-5-all-inclusive-by-total-equals-output.csv"
            ).exists()
        )
        self.assertFalse(
            (
                explicit_output / "top-5-all-inclusive-by-total-equals-output.csv"
            ).exists()
        )

    def test_sort_by_average_per_root_tu(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        one_tu = self.traces.project_detail("libcudacxx/include/cuda/std/one_tu.h")
        two_tus = self.traces.project_detail("libcudacxx/include/cuda/std/two_tus.h")
        self.traces.write_trace(
            traces / "target" / "first.json",
            [
                self.traces.event("Same", one_tu, 0, 80),
                self.traces.event("Same", two_tus, 100, 60),
            ],
            "first",
        )
        self.traces.write_trace(
            traces / "target" / "second.json",
            [self.traces.event("Same", two_tus, 0, 40)],
            "second",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "all",
                "-i",
                "--sort",
                "avg-root-tu",
                "-n",
                "2",
                "--tag",
                "avg-root-tu",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(output / "top-2-all-inclusive-by-avg-root-tu-avg-root-tu.csv")
        self.assertTrue(rows[0]["event_key"].endswith("one_tu.h"))
        self.assertEqual(rows[0]["selected_avg_per_root_tu_s"], "0.000080")
        self.assertTrue(rows[1]["event_key"].endswith("two_tus.h"))
        self.assertEqual(rows[1]["selected_avg_per_root_tu_s"], "0.000050")

    def test_host_compiler_filter_matches_host_phase_events(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        self.traces.write_trace(
            traces / "target" / "host.json",
            [
                self.traces.event(
                    "g++ (preprocessing 1)", "g++ (preprocessing 1)", 0, 10
                ),
                self.traces.event("gcc (compiling)", "gcc (compiling)", 20, 30),
                self.traces.event("CUDA C++ Front-End", "frontend", 60, 40),
            ],
            "host",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "host-compiler",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "host",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(output / "top-5-host-compiler-inclusive-by-total-host.csv")
        self.assertEqual(
            {row["event_name"] for row in rows},
            {"g++ (preprocessing 1)", "gcc (compiling)"},
        )

    def test_scope_filter_defaults_to_cccl_demangled_symbols(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        self.traces.write_trace(
            traces / "target" / "symbols.json",
            [
                self.traces.event(
                    "Scanning Function Body",
                    "std::basic_string_view::size() const noexcept",
                    0,
                    100,
                ),
                self.traces.event(
                    "Scanning Function Body",
                    "cuda::std::__4::basic_string_view::size() const noexcept",
                    200,
                    80,
                ),
                self.traces.event(
                    "Scanning Function Body",
                    'cuda::std::literals::operator ""sv(const char *, unsigned long)',
                    400,
                    60,
                ),
            ],
            "symbols",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "scanning-function-body",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "scope-default",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(
            output / "top-5-scanning-function-body-inclusive-by-total-scope-default.csv"
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("cuda::std", rows[0]["event_key"])
        self.assertIn("operator", rows[1]["event_key"])

    def test_empty_scope_filter_disables_symbol_scope_filtering(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        self.traces.write_trace(
            traces / "target" / "symbols.json",
            [
                self.traces.event(
                    "Scanning Function Body",
                    "std::basic_string_view::size() const noexcept",
                    0,
                    100,
                ),
                self.traces.event(
                    "Scanning Function Body",
                    "cuda::std::__4::basic_string_view::size() const noexcept",
                    200,
                    80,
                ),
            ],
            "symbols",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "scanning-function-body",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--scope-filter",
                "",
                "--tag",
                "scope-disabled",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(
            output
            / "top-5-scanning-function-body-inclusive-by-total-scope-disabled.csv"
        )
        self.assertEqual(len(rows), 2)
        self.assertIn("std::basic_string_view", rows[0]["event_key"])
        self.assertIn("cuda::std", rows[1]["event_key"])

    def test_scope_filter_uses_reported_template_symbol_scope(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        self.traces.write_trace(
            traces / "target" / "templates.json",
            [
                self.traces.event(
                    "Instantiating Template Function",
                    (
                        "nvtx3::v1::domain::get "
                        "[nvtx3::v1::domain::get<cuda::__4::__nvtx_cccl_domain>()]"
                    ),
                    0,
                    100,
                ),
                self.traces.event(
                    "Instantiating Template Function",
                    "cub::detail::load [cub::detail::load<int>()]",
                    200,
                    80,
                ),
            ],
            "templates",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "template-instantiation",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "template-scope",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(
            output
            / "top-5-template-instantiation-inclusive-by-total-template-scope.csv"
        )
        self.assertEqual(len(rows), 1)
        self.assertIn("cub::detail::load", rows[0]["event_key"])

    def test_scope_filter_matches_mangled_cccl_namespaces(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        self.traces.write_trace(
            traces / "target" / "mangled.json",
            [
                self.traces.event(
                    "Generating Function IR",
                    "_ZN4cuda3std3__43fooEv",
                    0,
                    100,
                ),
                self.traces.event(
                    "Generating Function IR",
                    "_ZN6thrust6detail3barEv",
                    200,
                    80,
                ),
                self.traces.event(
                    "Generating Function IR",
                    "_ZNSt6vectorIiE4sizeEv",
                    400,
                    200,
                ),
            ],
            "mangled",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "code-generation",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "mangled-scope",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rows = csv_rows(
            output / "top-5-code-generation-inclusive-by-total-mangled-scope.csv"
        )
        self.assertEqual(
            {row["event_key"] for row in rows},
            {
                "_ZN4cuda3std3__43fooEv",
                "_ZN6thrust6detail3barEv",
            },
        )

    def test_total_compilation_filter_uses_trace_wall_span(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"

        self.traces.write_trace(
            traces / "target" / "total.json",
            [
                self.traces.event(
                    "g++ (preprocessing 1)", "g++ (preprocessing 1)", 10, 20
                ),
                self.traces.event("CUDA C++ Front-End", "frontend", 50, 40),
            ],
            "total",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "total-compilation",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "total",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        total_rows = csv_rows(
            output / "top-5-total-compilation-inclusive-by-total-total.csv"
        )
        self.assertEqual(len(total_rows), 1)
        self.assertEqual(total_rows[0]["event_name"], "Total Compilation Time")
        self.assertEqual(total_rows[0]["selected_total_s"], "0.000080")

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "-f",
                "all",
                "-i",
                "--sort",
                "total",
                "-n",
                "5",
                "--tag",
                "all-no-total",
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        all_rows = csv_rows(output / "top-5-all-inclusive-by-total-all-no-total.csv")
        self.assertFalse(
            any(row["event_name"] == "Total Compilation Time" for row in all_rows)
        )

    def test_same_dir_comparison_is_empty(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"
        same = self.traces.project_detail("libcudacxx/include/cuda/std/same.h")
        self.traces.write_trace(
            traces / "target" / "same.json",
            [self.traces.event("Same", same, 0, 10)],
            "same",
        )

        self.run_summary(
            traces,
            traces,
            output,
            "-f",
            "all",
            "-i",
            "--sort",
            "max",
            "-n",
            "5",
            "--tag",
            "same",
        )
        self.assert_empty_csv(
            self.comparison_csv(output, "top-5-all-inclusive-by-max-worse-same.csv")
        )
        self.assert_empty_csv(
            self.comparison_csv(output, "top-5-all-inclusive-by-max-better-same.csv")
        )

    def test_multi_slice_writes_manifest_and_allows_empty_slice(self) -> None:
        traces = self.work / "traces"
        output = self.work / "reports"
        slices = self.work / "slices.json"
        same = self.traces.project_detail("libcudacxx/include/cuda/std/same.h")
        self.traces.write_trace(
            traces / "target" / "same.json",
            [self.traces.event("Same", same, 0, 10)],
            "same",
        )
        slices.write_text(
            json.dumps(
                {
                    "slices": [
                        {
                            "id": "all-events",
                            "title": "All events",
                            "filter": "all",
                            "timing": "inclusive",
                            "sort": "total",
                            "top": 5,
                            "threshold": 0,
                        },
                        {
                            "id": "empty-events",
                            "title": "Empty events",
                            "filter": "does-not-match",
                            "timing": "inclusive",
                            "sort": "total",
                            "top": 5,
                            "threshold": 0,
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                SUMMARY_SCRIPT.as_posix(),
                traces.as_posix(),
                "-o",
                output.as_posix(),
                "--slices",
                slices.as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        with (output / "summary.json").open(encoding="utf-8") as f:
            manifest = json.load(f)

        self.assertEqual(
            [item["id"] for item in manifest["slices"]], ["all-events", "empty-events"]
        )
        self.assertEqual(manifest["slices"][0]["reports"]["current"]["row_count"], 1)
        self.assertEqual(manifest["slices"][1]["reports"]["current"]["row_count"], 0)
        self.assertTrue(
            (
                output
                / "empty-events"
                / "top-5-regex-does-not-match-inclusive-by-total.csv"
            ).exists()
        )


class CompileTimeMatrixAndCommentTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.work = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_parse_matrix_disabled_when_section_missing(self) -> None:
        matrix = self.work / "matrix.yaml"
        matrix.write_text("workflows: {}\n", encoding="utf-8")

        completed = subprocess.run(
            [
                sys.executable,
                PARSE_MATRIX_SCRIPT.as_posix(),
                matrix.as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertEqual(json.loads(completed.stdout), {"include": []})

    def test_parse_matrix_valid_config(self) -> None:
        matrix = self.work / "matrix.yaml"
        matrix.write_text(
            """
compile_time:
  pull_request:
    - id: public-headers
      name: Public headers
      gpu: rtx2080
      launch_args: "--cuda 13.3 --host gcc13"
      baseline_ref: origin/main
      preset: all-dev
      targets: [cub.headers.base]
      args: "-arch native"
      slices:
        - id: total-compilation
          title: TU total compilation
          filter: total-compilation
          timing: inclusive
          sort: total
          top: 15
          threshold: 0.001
""",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                PARSE_MATRIX_SCRIPT.as_posix(),
                matrix.as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        include = json.loads(completed.stdout)["include"]
        self.assertEqual(len(include), 1)
        self.assertEqual(
            include[0]["comment_header"], "compile-time-bench-public-headers"
        )
        self.assertEqual(json.loads(include[0]["targets_json"]), ["cub.headers.base"])
        self.assertEqual(
            json.loads(include[0]["slices_json"])["slices"][0]["id"],
            "total-compilation",
        )

    def test_parse_matrix_rejects_duplicate_slice_ids(self) -> None:
        matrix = self.work / "matrix.yaml"
        matrix.write_text(
            """
compile_time:
  pull_request:
    - id: public-headers
      name: Public headers
      gpu: rtx2080
      launch_args: "--cuda 13.3 --host gcc13"
      baseline_ref: origin/main
      preset: all-dev
      targets: [cub.headers.base]
      slices:
        - id: repeated
          title: First
          filter: all
          timing: inclusive
          sort: total
          top: 15
          threshold: 0
        - id: repeated
          title: Second
          filter: all
          timing: inclusive
          sort: total
          top: 15
          threshold: 0
""",
            encoding="utf-8",
        )

        completed = subprocess.run(
            [
                sys.executable,
                PARSE_MATRIX_SCRIPT.as_posix(),
                matrix.as_posix(),
            ],
            cwd=REPO_ROOT,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        self.assertNotEqual(completed.returncode, 0)
        self.assertIn("duplicate slice id", completed.stderr)

    def test_render_comment_omits_empty_sections_and_splits_directions(self) -> None:
        summary = self.work / "summary.json"
        config = self.work / "config.json"
        output = self.work / "comment.md"
        summary.write_text(
            json.dumps(
                {
                    "slices": [
                        {
                            "id": "nonempty",
                            "title": "Nonempty",
                            "filter": "all",
                            "timing": "inclusive",
                            "sort": "total",
                            "comparison": {
                                "worse": {
                                    "rows": [
                                        {
                                            "rank": 1,
                                            "event_name": "Same",
                                            "event_key": "cuda/std/same",
                                            "baseline_selected_s": "0.000001",
                                            "current_selected_s": "0.000003",
                                            "impact_magnitude_s": "0.000010",
                                            "selected_delta_s": "0.000002",
                                            "matched_trace_count": 1,
                                        }
                                    ]
                                },
                                "better": {"rows": []},
                            },
                            "children": [
                                {
                                    "id": "empty-child",
                                    "title": "Empty child",
                                    "filter": "all",
                                    "timing": "inclusive",
                                    "sort": "total",
                                    "comparison": {
                                        "worse": {"rows": []},
                                        "better": {"rows": []},
                                    },
                                    "children": [],
                                }
                            ],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        config.write_text(
            json.dumps(
                {
                    "id": "public-headers",
                    "name": "Public headers",
                    "baseline_ref": "origin/main",
                    "preset": "all-dev",
                    "targets": ["cub.headers.base"],
                    "gpu": "rtx2080",
                    "launch_args": "--cuda 13.3 --host gcc13",
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                RENDER_COMMENT_SCRIPT.as_posix(),
                "--summary",
                summary.as_posix(),
                "--config",
                config.as_posix(),
                "--artifacts-url",
                "https://example.test/artifacts",
                "-o",
                output.as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rendered = output.read_text(encoding="utf-8")
        self.assertIn("<!-- cccl-compile-time-bench: public-headers -->", rendered)
        self.assertIn("Regressions", rendered)
        self.assertIn("Regression impact", rendered)
        self.assertIn("0.000010", rendered)
        self.assertNotIn("Improvements</strong>", rendered)
        self.assertNotIn("Empty child", rendered)
        self.assertIn("https://example.test/artifacts", rendered)

    def test_render_comment_reports_empty_slice_warnings(self) -> None:
        summary = self.work / "summary.json"
        config = self.work / "config.json"
        output = self.work / "comment.md"
        summary.write_text(
            json.dumps(
                {
                    "slices": [
                        {
                            "id": "empty",
                            "title": "Empty slice",
                            "filter": "typo-filter",
                            "timing": "inclusive",
                            "sort": "total",
                            "warnings": [
                                "baseline report matched no events for this slice"
                            ],
                            "comparison": {
                                "worse": {"rows": []},
                                "better": {"rows": []},
                            },
                            "children": [],
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        config.write_text(
            json.dumps(
                {
                    "id": "public-headers",
                    "name": "Public headers",
                    "baseline_ref": "origin/main",
                    "preset": "all-dev",
                    "targets": ["cub.headers.base"],
                    "gpu": "rtx2080",
                    "launch_args": "--cuda 13.3 --host gcc13",
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                RENDER_COMMENT_SCRIPT.as_posix(),
                "--summary",
                summary.as_posix(),
                "--config",
                config.as_posix(),
                "--artifacts-url",
                "https://example.test/artifacts",
                "-o",
                output.as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        rendered = output.read_text(encoding="utf-8")
        self.assertIn("1 warning(s)", rendered)
        self.assertIn("Empty slice — Warnings", rendered)
        self.assertIn("baseline report matched no events", rendered)
        self.assertNotIn(
            "No compile-time benchmark changes exceeded the configured thresholds.",
            rendered,
        )


class PrepareTracesTest(unittest.TestCase):
    def setUp(self) -> None:
        self.tempdir = tempfile.TemporaryDirectory()
        self.work = Path(self.tempdir.name)

    def tearDown(self) -> None:
        self.tempdir.cleanup()

    def test_promotes_project_details_into_perfetto_event_names(self) -> None:
        input_dir = self.work / "raw"
        output_dir = self.work / "perfetto"
        trace_path = input_dir / "target" / "trace.json"
        trace_path.parent.mkdir(parents=True, exist_ok=True)
        trace_path.write_text(
            json.dumps(
                {
                    "traceEvents": [
                        {
                            "ph": "X",
                            "name": "Processing Header File",
                            "args": {
                                "detail": (
                                    REPO_ROOT
                                    / "libcudacxx/include/cuda/std/string_view"
                                ).as_posix()
                            },
                        },
                        {
                            "ph": "X",
                            "name": "Other Event",
                            "args": {"detail": "not promoted"},
                        },
                    ]
                }
            ),
            encoding="utf-8",
        )

        subprocess.run(
            [
                sys.executable,
                PREPARE_SCRIPT.as_posix(),
                "--input",
                input_dir.as_posix(),
                "--output",
                output_dir.as_posix(),
                "--repo-root",
                REPO_ROOT.as_posix(),
            ],
            cwd=REPO_ROOT,
            check=True,
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )

        output_trace = output_dir / "target" / "trace.perfetto.json"
        with output_trace.open(encoding="utf-8") as f:
            events = json.load(f)["traceEvents"]

        self.assertEqual(
            events[0]["name"], "Processing Header File: cuda/std/string_view"
        )
        self.assertEqual(events[0]["args"]["original_name"], "Processing Header File")
        self.assertEqual(events[1]["name"], "Other Event")


class SummarizeTusTest(unittest.TestCase):
    def test_generated_tu_input(self) -> None:
        build_dir = Path("/tmp/build")
        pp_path = (
            build_dir
            / "libcudacxx/test"
            / "headers"
            / "libcudacxx.test.public_headers"
            / "cuda/std/string_view.cpp4.ii"
        )

        self.assertEqual(
            summarize_tus.generated_tu_input(pp_path.with_name("string_view.cpp")),
            "cuda/std/string_view",
        )


if __name__ == "__main__":
    unittest.main()
