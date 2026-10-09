#!/usr/bin/env python3
# Copyright (c) 2026, NVIDIA CORPORATION.

import json
import os
import shutil
import subprocess
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

from ci.compile_time import warmup

REPO_ROOT = Path(__file__).resolve().parents[2]


class BuildPairTest(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)

    def test_warmup_cleans_traces_without_deleting_configured_directories(self):
        build = self.root / "build"
        trace_dir = build / "compile_time" / "raw_traces" / "target"
        trace_dir.mkdir(parents=True)
        (trace_dir / "warmup.json").write_text("{}")
        (build / "compile_commands.json").write_text(
            json.dumps(
                [
                    {
                        "directory": str(build),
                        "command": "nvcc --fdevice-time-trace=trace -c header.cu -o header.o",
                    }
                ]
            )
        )
        with patch.object(warmup.subprocess, "run") as run:
            warmup.warmup(build)
        self.assertEqual(
            run.call_args_list[0].args[0],
            ["cmake", "--build", str(build), "--target", "header.o", "-j1"],
        )
        self.assertEqual(run.call_args_list[1].args[0][-1], "clean")
        self.assertTrue(trace_dir.is_dir())
        self.assertEqual(list(trace_dir.iterdir()), [])

    def test_missing_instrumentation_fails_instead_of_skipping_warmup(self):
        (self.root / "compile_commands.json").write_text("[]")
        with self.assertRaisesRegex(SystemExit, "no instrumented compile command"):
            warmup.warmup(self.root)

    def test_warmup_selects_the_same_shared_object_when_command_order_differs(self):
        before, after = self.root / "baseline", self.root / "current"
        for build, names in (
            (before, ["a.o", "shared.o"]),
            (after, ["z.o", "shared.o"]),
        ):
            build.mkdir()
            (build / "compile_commands.json").write_text(
                json.dumps(
                    [
                        {
                            "directory": str(build),
                            "arguments": ["nvcc", "--fdevice-time-trace=t", "-o", name],
                        }
                        for name in names
                    ]
                )
            )
        with patch.object(warmup.subprocess, "run") as run:
            warmup.warmup(before, after)
            warmup.warmup(after, before)
        self.assertEqual(run.call_args_list[0].args[0][-2], "shared.o")
        self.assertEqual(run.call_args_list[2].args[0][-2], "shared.o")

    def test_wrapper_orders_clean_builds_preserves_current_changes_and_cleans_worktrees(
        self,
    ):
        for order in ("current-first", "baseline-first"):
            with self.subTest(order=order):
                self.run_pair(order)

    def test_local_auto_order_is_deterministic(self):
        self.run_pair("auto")

    def test_failed_measurement_preserves_partial_traces_and_removes_stale_summary(
        self,
    ):
        self.run_pair("current-first", fail=True)

    def run_pair(self, order, *, fail=False):
        repo = self.root / order
        (repo / "ci" / "util").mkdir(parents=True)
        (repo / "cmake").mkdir()
        (repo / "CMakePresets.json").write_text("{}")
        (repo / "cmake" / "CCCLGenerateHeaderTests.cmake").write_text("")
        (repo / "header.h").write_text("baseline")
        shutil.copy(REPO_ROOT / "ci" / "build_compile_time_bench.sh", repo / "ci")
        shutil.copytree(
            REPO_ROOT / "ci" / "compile_time",
            repo / "ci" / "compile_time",
            ignore=shutil.ignore_patterns("__pycache__"),
        )
        shutil.copy(
            REPO_ROOT / "ci" / "util" / "extract_switches.sh", repo / "ci" / "util"
        )
        (repo / "ci" / "build_common.sh").write_text("""
BUILD_ROOT="${repo_root}/build"
BUILD_DIR="${BUILD_ROOT}/${CCCL_BUILD_INFIX}"
mkdir -p "${BUILD_DIR}"
CONFIGURE_ONLY=false
configure_preset() { (cd .. && cmake configure "${BUILD_DIR}/$2"); }
build_preset() { (cd .. && cmake measure "${BUILD_DIR}/$2"); }
""")
        subprocess.run(["git", "init", "-q", str(repo)], check=True)
        subprocess.run(["git", "-C", str(repo), "add", "."], check=True)
        subprocess.run(
            [
                "git",
                "-C",
                str(repo),
                "-c",
                "user.name=Test",
                "-c",
                "user.email=test@example.com",
                "commit",
                "--no-gpg-sign",
                "-qm",
                "base",
            ],
            check=True,
        )
        (repo / "header.h").write_text("current")
        (repo / "new.h").write_text("new current header")
        report_root = repo / "build" / "pair-test" / "all-dev" / "compile_time"
        (report_root / "event_reports").mkdir(parents=True)
        (report_root / "event_reports" / "summary.json").write_text('{"stale":true}')
        tools = self.root / f"tools-{order}"
        tools.mkdir()
        fake = tools / "cmake"
        fake.write_text("""#!/usr/bin/env python3
import json, os, sys
from pathlib import Path
args = sys.argv[1:]
build = Path(args[1])
side = {"source-a": "baseline", "source-b": "current"}.get(Path.cwd().name)
if args[0] in ("configure", "measure"):
    assert not os.environ.get("CMAKE_CXX_COMPILER_LAUNCHER")
    assert not os.environ.get("CMAKE_CUDA_COMPILER_LAUNCHER")
    assert (Path.cwd() / "header.h").read_text() == ("current" if side == "current" else "baseline")
    assert (Path.cwd() / "new.h").exists() == (side == "current")
    build.mkdir(parents=True, exist_ok=True)
    if args[0] == "configure":
        (build / "compile_commands.json").write_text(json.dumps([{"directory": str(build), "arguments": ["nvcc", "--fdevice-time-trace=trace", "-o", "header.o"]}]))
    label = args[0]
else:
    build = Path(args[1])
    side = {"source-a": "baseline", "source-b": "current"}[next(part for part in build.parts if part in ("source-a", "source-b"))]
    label = "clean" if "clean" in args else "warmup"
with open(os.environ["PAIR_OPERATIONS"], "a") as log:
    log.write(f"{label}:{side}\\n")
trace_dir = build / "compile_time" / "raw_traces" / "target"
trace_dir.mkdir(parents=True, exist_ok=True)
if label in ("measure", "warmup"):
    event = {"ph":"X", "ts":0, "dur":1000000, "name":"Processing Header File", "args":{"detail":str(Path.cwd() / "header.h")}}
    (trace_dir / ("warmup-only.json" if label == "warmup" else "header.json")).write_text(json.dumps({"traceEvents":[event], "otherData":{"inputFiles":["/generated/headers/target/header.cu"]}}))
    if label == "measure" and os.environ.get("FAIL_MEASURE") == "1":
        raise SystemExit(42)
""")
        fake.chmod(0o755)
        log = self.root / f"operations-{order}"
        env = dict(
            os.environ,
            PATH=f"{tools}:{os.environ['PATH']}",
            CCCL_BUILD_INFIX="pair-test",
            PAIR_OPERATIONS=str(log),
            CMAKE_CXX_COMPILER_LAUNCHER="sccache",
            CMAKE_CUDA_COMPILER_LAUNCHER="sccache",
            FAIL_MEASURE="1" if fail else "0",
        )
        env.pop("GITHUB_RUN_NUMBER", None)
        env.pop("GITHUB_RUN_ATTEMPT", None)
        expected_order = "baseline-first" if order == "auto" else order
        command = [
            "bash",
            str(repo / "ci" / "build_compile_time_bench.sh"),
            "-baseline-ref",
            "HEAD",
            "-build-order",
            order,
            "-no-prepare-perfetto",
            "--",
            "-f",
            "file-processing",
            "-e",
        ]
        completed = subprocess.run(
            command, cwd=repo, env=env, capture_output=True, text=True
        )
        self.assertEqual(
            completed.returncode, 42 if fail else 0, completed.stdout + completed.stderr
        )
        if fail:
            self.assertFalse((report_root / "event_reports" / "summary.json").exists())
            self.assertEqual(len(list((report_root / "raw_traces").rglob("*.json"))), 1)
            self.assertEqual((repo / "header.h").read_text(), "current")
            worktrees = subprocess.check_output(
                ["git", "-C", str(repo), "worktree", "list", "--porcelain"], text=True
            )
            self.assertEqual(worktrees.count("worktree "), 1)
            return
        first, second = (
            ("current", "baseline")
            if expected_order == "current-first"
            else ("baseline", "current")
        )
        self.assertEqual(
            log.read_text().splitlines(),
            [
                f"configure:{first}",
                f"configure:{second}",
                f"warmup:{first}",
                f"clean:{first}",
                f"warmup:{second}",
                f"clean:{second}",
                f"measure:{first}",
                f"measure:{second}",
            ],
        )
        report_root = repo / "build" / "pair-test" / "all-dev" / "compile_time"
        self.assertEqual(len(list((report_root / "raw_traces").rglob("*.json"))), 1)
        self.assertEqual(
            len(list((report_root / "baseline_raw_traces").rglob("*.json"))), 1
        )
        summary = json.loads(
            (report_root / "event_reports" / "summary.json").read_text()
        )
        self.assertEqual(summary["build_pair"]["build_order"], expected_order)
        self.assertEqual((repo / "header.h").read_text(), "current")
        self.assertEqual((repo / "new.h").read_text(), "new current header")
        worktrees = subprocess.check_output(
            ["git", "-C", str(repo), "worktree", "list", "--porcelain"], text=True
        )
        self.assertEqual(worktrees.count("worktree "), 1)


if __name__ == "__main__":
    unittest.main()
