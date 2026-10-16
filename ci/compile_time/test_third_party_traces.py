#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import hashlib
import json
import os
import re
import shlex
import shutil
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

from ci.compile_time.render_trace_report import (
    COMMENT_LIMIT,
    render_reports,
    render_row,
)
from ci.compile_time.summarize_events import normalize_cccl_file
from ci.compile_time.test_summarize_events import TraceBuilder
from ci.compile_time.third_party_matrix import parse_matrix
from ci.compile_time.write_trace_metadata import build_configuration

ROOT = Path(__file__).resolve().parents[2]
COLLECT = ROOT / "ci/compile_time/collect_traces.py"
SUMMARIZE = ROOT / "ci/compile_time/summarize_events.py"


class ThirdPartyTracesTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.work = Path(self.temp.name)

    def tearDown(self):
        self.temp.cleanup()

    def run_python(self, script, *args, check=True):
        return subprocess.run(
            [sys.executable, str(script), *map(str, args)],
            capture_output=True,
            text=True,
            check=check,
        )

    def test_report_escapes_bitwise_operator_names(self):
        row = render_row(
            {
                "rank": 1,
                "event_name": "Scanning Function Body",
                "event_key": "cuda::std::operator|(cuda::std::byte, cuda::std::byte)",
                "selected_total_s": "1.000000",
                "selected_avg_per_event_s": "0.100000",
                "event_count": 10,
                "root_tu_count": 2,
            },
            "https://example.com/reports",
        )
        self.assertIn(r"operator\|", row)
        self.assertEqual(len(re.findall(r"(?<!\\)\|", row)), 7)

    def test_weekly_and_explicit_pr_selection(self):
        matrix = ROOT / "ci/matrix.yaml"
        weekly = parse_matrix(matrix, "weekly")["include"]
        self.assertEqual(len(weekly), 6)
        self.assertEqual(parse_matrix(matrix, "pull_request")["include"], [])
        self.assertEqual(
            parse_matrix(matrix, "pull_request", opt_in=True)["include"], weekly
        )
        self.assertEqual(
            len(
                parse_matrix(matrix, "pull_request", opt_in=True, skips=("rapids",))[
                    "include"
                ]
            ),
            2,
        )
        self.assertEqual(
            parse_matrix(matrix, "pull_request", opt_in=True, skips=("all",))[
                "include"
            ],
            [],
        )
        for config in weekly:
            self.assertNotIn("baseline_ref", config)
            self.assertEqual(config["runner"], "linux-amd64-cpu32")
            slices = json.loads(config["slices_json"])["slices"]
            self.assertEqual(len(slices), 5)
            self.assertTrue(all(s["top"] == 50 for s in slices))
        pytorch = next(c for c in weekly if c["project"] == "pytorch")
        self.assertEqual(pytorch["summary_jobs"], 1)

    def test_pr_override_selects_only_matx(self):
        config = parse_matrix(ROOT / "ci/matrix.yaml", "weekly")["include"][0]
        raw = {k: config[k] for k in ("id", "project", "name", "runner", "launch_args")}
        raw["slices"] = json.loads(config["slices_json"])["slices"]
        path = self.work / "matrix.json"
        path.write_text(json.dumps({"compile_time": {"third_party_override": [raw]}}))
        self.assertEqual(
            [c["project"] for c in parse_matrix(path, "pull_request")["include"]],
            ["matx"],
        )

    def test_configurations_reference_existing_devcontainers(self):
        for config in parse_matrix(ROOT / "ci/matrix.yaml", "weekly")["include"]:
            with self.subTest(config=config["id"]):
                args = shlex.split(config["launch_args"])
                cuda = args[args.index("--cuda") + 1]
                host = args[args.index("--host") + 1]
                suffix = "ext" if "--cuda-ext" in args else ""
                path = (
                    ROOT
                    / ".devcontainer"
                    / f"cuda{cuda}{suffix}-{host}"
                    / "devcontainer.json"
                )
                self.assertTrue(
                    path.is_file(), f"Missing launcher configuration: {path}"
                )
                self.assertIn("image", json.loads(path.read_text()))

    def test_metadata_records_build_flags_without_unrelated_cache_values(self):
        build = self.work / "build"
        build.mkdir()
        (build / "CMakeCache.txt").write_text(
            "CMAKE_CUDA_ARCHITECTURES:STRING=80\n"
            "CMAKE_CUDA_FLAGS_RELEASE:STRING=-O3\n"
            "CMAKE_BUILD_TYPE:STRING=Release\n"
            "UNRELATED_VALUE:STRING=not-part-of-the-dataset\n"
        )
        self.assertEqual(
            build_configuration([("matx", build)]),
            {
                "matx/CMakeCache.txt": {
                    "CMAKE_CUDA_ARCHITECTURES": "80",
                    "CMAKE_CUDA_FLAGS_RELEASE": "-O3",
                    "CMAKE_BUILD_TYPE": "Release",
                }
            },
        )

    def test_collection_preserves_bytes_and_build_phases(self):
        source = self.work / "source"
        source.mkdir()
        trace = source / "kernel.cu.o.json"
        contents = b'{ "traceEvents": [], "note": "original bytes" }\n'
        trace.write_bytes(contents)
        (source / "unrelated.json").write_text("not a compiler trace")
        (source / "main.cu.o.json").symlink_to(trace.name)
        state = self.work / "state.json"
        first = self.work / "raw/no-tests"
        self.run_python(
            COLLECT,
            "--input",
            f"consumer={source}",
            "--output",
            first,
            "--state",
            state,
        )
        collected = list(first.rglob("*.json"))
        self.assertEqual(len(collected), 1)
        self.assertEqual(collected[0].read_bytes(), contents)
        second = self.work / "raw/with-tests"
        self.run_python(
            COLLECT,
            "--input",
            f"consumer={source}",
            "--output",
            second,
            "--state",
            state,
        )
        self.assertEqual(list(second.rglob("*.json")), [])
        rewritten = contents + b"\n"
        trace.write_bytes(rewritten)
        self.run_python(
            COLLECT,
            "--input",
            f"consumer={source}",
            "--output",
            second,
            "--state",
            state,
        )
        self.assertEqual(next(second.rglob("*.json")).read_bytes(), rewritten)
        self.assertEqual(collected[0].read_bytes(), contents)

    def test_header_scope_includes_each_third_party_layout(self):
        root = self.work / "project"
        cases = {
            "build/_deps/cccl-src/cub/cub/config.cuh": "cub/cub/config.cuh",
            "cpp/build/_deps/raft-src/build/_deps/cccl-src/thrust/thrust/tuple.h": "thrust/thrust/tuple.h",
            "cuda/targets/x86_64-linux/include/cccl/cuda/std/tuple": "libcudacxx/include/cuda/std/tuple",
            "cuda/include/cccl/cuda/experimental/stream.cuh": "cudax/include/cuda/experimental/stream.cuh",
            "cuda/include/cccl/cub/block/block_reduce.cuh": "cub/cub/block/block_reduce.cuh",
            "build/_deps/pybind11-src/include/pybind11/pybind11.h": None,
            "build/_deps/cccl-src/build/generated.h": None,
            "include/consumer.h": None,
        }
        for relative, expected in cases.items():
            with self.subTest(relative=relative):
                self.assertEqual(
                    normalize_cccl_file(str(root / relative), root), expected
                )
        self.assertIsNone(
            normalize_cccl_file("/usr/local/cuda/include/cccl/cub/cub.cuh", root)
        )

    def test_collector_failure_preserves_previous_dataset(self):
        source = self.work / "source"
        source.mkdir()
        output = self.work / "dataset"
        output.mkdir()
        sentinel = output / "keep.json"
        sentinel.write_text("already collected")
        outside = self.work / "outside.cu.o.json"
        outside.write_text("{}")
        (source / "escaped.cu.o.json").symlink_to(outside)
        result = self.run_python(
            COLLECT, "--input", f"x={source}", "--output", output, check=False
        )
        self.assertNotEqual(result.returncode, 0)
        self.assertEqual(sentinel.read_text(), "already collected")

    def fixture_checkout(self, exit_code):
        checkout = self.work / "checkout"
        for relative in (
            "ci/build_third_party_compile_time.sh",
            "ci/compile_time/collect_traces.py",
            "ci/compile_time/write_trace_metadata.py",
        ):
            destination = checkout / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(ROOT / relative, destination)
        driver = checkout / "ci/matx/build_matx.sh"
        driver.parent.mkdir()
        driver.write_text(
            """#!/usr/bin/env bash
set -euo pipefail
[[ "${CCCL_COMPILE_TIME_BENCH}" == 1 && "${SCCACHE_DISABLE}" == 1 ]]
directory="build/${CCCL_BUILD_INFIX}/matx/build"
mkdir -p "${directory}"
printf '{ "traceEvents": [] }\\n' > "${directory}/source.cu.o.json"
exit """
            + str(exit_code)
            + "\n"
        )
        driver.chmod(0o755)
        subprocess.run(["git", "init", "-q", str(checkout)], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(checkout),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "-c",
                "commit.gpgsign=false",
                "-c",
                "core.hooksPath=/dev/null",
                "commit",
                "--allow-empty",
                "-qm",
                "fixture",
            ],
            check=True,
        )
        return checkout

    def test_failed_build_still_collects_original_traces(self):
        checkout = self.fixture_checkout(7)
        env = dict(os.environ, CCCL_BUILD_INFIX="test-traces")
        result = subprocess.run(
            ["bash", "ci/build_third_party_compile_time.sh", "-project", "matx"],
            cwd=checkout,
            env=env,
            capture_output=True,
            text=True,
        )
        self.assertEqual(result.returncode, 7, result.stderr)
        dataset = checkout / "build/test-traces/matx/compile_time"
        metadata = json.loads((dataset / "dataset.json").read_text())
        self.assertEqual(metadata["status"], "partial")
        self.assertEqual(metadata["build_exit_code"], 7)
        self.assertEqual(metadata["trace_count"], 1)
        self.assertIsNotNone(metadata["cccl_sha"])
        self.assertEqual(
            next((dataset / "raw_traces").rglob("*.json")).read_bytes(),
            b'{ "traceEvents": [] }\n',
        )
        self.assertFalse((dataset / "event_reports").exists())

    def test_successful_build_cannot_overwrite_an_existing_dataset(self):
        checkout = self.fixture_checkout(0)
        env = dict(os.environ, CCCL_BUILD_INFIX="test-traces")
        command = ["bash", "ci/build_third_party_compile_time.sh", "-project", "matx"]
        result = subprocess.run(
            command, cwd=checkout, env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 0, result.stderr)
        path = checkout / "build/test-traces/matx/compile_time/dataset.json"
        before = path.read_bytes()
        self.assertEqual(json.loads(before)["status"], "complete")
        result = subprocess.run(
            command, cwd=checkout, env=env, capture_output=True, text=True
        )
        self.assertEqual(result.returncode, 2)
        self.assertEqual(path.read_bytes(), before)

    def test_single_dataset_top_50_scoping_grouping_and_parallelism(self):
        traces = self.work / "dataset/raw_traces"
        repo = self.work / "consumer"
        builder = TraceBuilder(repo)
        config = parse_matrix(ROOT / "ci/matrix.yaml", "weekly")["include"][0]
        specs = json.loads(config["slices_json"])
        specs_file = self.work / "slices.json"
        specs_file.write_text(json.dumps(specs))
        events = [
            builder.event(
                "Processing Header File",
                str(repo / "include/consumer.hpp"),
                0,
                1_000_000,
            )
        ]
        for i in range(75):
            events.extend(
                [
                    builder.event(
                        "Processing Header File",
                        str(repo / f"build/_deps/cccl-src/cub/cub/header_{i:02}.cuh"),
                        i * 100,
                        100 - i,
                    ),
                    builder.event(
                        "Instantiating Template Function",
                        f"cub::kernel_{i:02} [cub::kernel_{i:02}<int>()]",
                        10_000 + i * 100,
                        100 - i,
                    ),
                    builder.event(
                        "Scanning Function Body",
                        f"cuda::function_{i:02}()",
                        20_000 + i * 100,
                        100 - i,
                    ),
                ]
            )
        events.extend(
            [
                builder.event(
                    "Instantiating Template Function",
                    "cub::kernel_00 [cub::kernel_00<double>()]",
                    30_000,
                    100,
                ),
                builder.event(
                    "Scanning Function Body",
                    "consumer::function(cuda::std::tuple<int>)",
                    40_000,
                    1_000_000,
                ),
            ]
        )
        builder.write_trace(traces / "one.cu.o.json", events, "one")
        builder.write_trace(traces / "two.cu.o.json", events, "two")
        before = {
            p: hashlib.sha256(p.read_bytes()).digest() for p in traces.rglob("*.json")
        }
        results = []
        for jobs in (1, 2):
            output = self.work / f"report-{jobs}"
            self.run_python(
                SUMMARIZE,
                traces,
                "--repo-root",
                repo,
                "--cccl-only",
                "--slices",
                specs_file,
                "--jobs",
                jobs,
                "-o",
                output,
            )
            summary = json.loads((output / "summary.json").read_text())
            self.assertEqual(summary["mode"], "single")
            self.assertEqual(summary["header_scope"], "cccl")
            slices = {s["id"]: s for s in summary["slices"]}
            self.assertTrue(all("comparison" not in s for s in slices.values()))
            for sid in (
                "file-processing",
                "function-bodies",
                "template-instantiation",
                "primary-template-instantiation",
            ):
                self.assertEqual(slices[sid]["reports"]["current"]["row_count"], 50)
            headers = slices["file-processing"]["reports"]["current"]["rows"]
            self.assertTrue(
                all(row["event_key"].startswith("cub/cub/") for row in headers)
            )
            functions = slices["function-bodies"]["reports"]["current"]["rows"]
            self.assertTrue(
                all(row["event_key"].startswith("cuda::") for row in functions)
            )
            primary = slices["primary-template-instantiation"]["reports"]["current"][
                "rows"
            ][0]
            self.assertEqual(primary["event_key"], "cub::kernel_00")
            self.assertEqual(primary["event_count"], 4)
            results.append(
                {sid: s["reports"]["current"]["rows"] for sid, s in slices.items()}
            )
        self.assertEqual(results[0], results[1])
        self.assertEqual(
            before,
            {
                p: hashlib.sha256(p.read_bytes()).digest()
                for p in traces.rglob("*.json")
            },
        )

    def test_paginated_comments_keep_all_rows_and_link_oversized_names(self):
        dataset = self.work / "dataset"
        (dataset / "event_reports").mkdir(parents=True)
        spec = {
            "id": "templates",
            "title": "Template instantiation",
            "top": 50,
            "timing": "inclusive",
        }
        (dataset / "config.json").write_text(
            json.dumps({"id": "matx", "name": "MatX", "slices": {"slices": [spec]}})
        )
        (dataset / "dataset.json").write_text(
            json.dumps({"cccl_sha": "a" * 40, "status": "complete", "trace_count": 2})
        )
        rows = [
            {
                "rank": i,
                "event_name": "Instantiating Template Function",
                "event_key": f"cub::item_{i} [cub::item_{i}<"
                + "x" * (100_000 if i == 50 else 2000)
                + ">()]",
                "selected_total_s": "1.000000",
                "selected_avg_per_event_s": "0.100000",
                "event_count": 10,
                "root_tu_count": 2,
            }
            for i in range(1, 51)
        ]
        summary_path = dataset / "event_reports/summary.json"
        summary_path.write_text(
            json.dumps(
                {
                    "mode": "single",
                    "slices": [
                        {"id": "templates", "reports": {"current": {"rows": rows}}}
                    ],
                }
            )
        )
        original = summary_path.read_bytes()
        output = self.work / "comments"
        index = render_reports(
            dataset, output, "https://example.com/traces", "https://example.com/reports"
        )
        bodies = [(output / item["file"]).read_text() for item in index["include"]]
        self.assertTrue(all(len(body.encode()) <= COMMENT_LIMIT for body in bodies))
        joined = "\n".join(bodies)
        for i in range(1, 51):
            self.assertEqual(joined.count(f"| {i} | `1.000000`"), 1)
        self.assertIn("complete event name in report CSV", joined)
        self.assertNotIn("Regression", joined)
        self.assertEqual(summary_path.read_bytes(), original)


if __name__ == "__main__":
    unittest.main()
