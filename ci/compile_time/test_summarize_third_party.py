#!/usr/bin/env python3
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

import io
import json
import tempfile
import unittest
from pathlib import Path, PurePosixPath
from unittest.mock import patch

from ci.compile_time import summarize_third_party as reports
from ci.compile_time.test_summarize_events import TraceBuilder, csv_rows
from ci.compile_time.trace_json import iter_trace_events


class ThirdPartySummaryTest(unittest.TestCase):
    def setUp(self):
        self.tempdir = tempfile.TemporaryDirectory()
        self.addCleanup(self.tempdir.cleanup)
        self.work = Path(self.tempdir.name)
        self.traces = self.work / "traces"
        self.output = self.work / "reports"
        self.builder = TraceBuilder(Path("/home/coder"))
        self.source_root = PurePosixPath("/home/coder")

    def write_trace(self, name, entries):
        self.builder.write_trace(self.traces / f"{name}.o.json", entries, name)

    def run_summary(self, **kwargs):
        return reports.summarize(
            self.traces, self.output, self.source_root, "rapids", "OFF", **kwargs
        )

    def rows(self, slice_id):
        return {
            row["event_key"]: row for row in csv_rows(self.output / f"{slice_id}.csv")
        }

    def test_header_paths(self):
        cases = {
            "/home/coder/cudf/build/_deps/cccl-src/cub/cub/block/block_reduce.cuh": "cub/cub/block/block_reduce.cuh",
            "/home/coder/raft/build/_deps/cccl-src/lib/cmake/libcudacxx/../../../libcudacxx/include/cuda/std/type_traits": "libcudacxx/include/cuda/std/type_traits",
            "/home/coder/cccl/build/pytorch/cuda/targets/x86_64-linux/include/cccl/cuda/std/tuple": "libcudacxx/include/cuda/std/tuple",
            "/home/coder/cccl/build/pytorch/cuda/include/cccl/nv/target": "libcudacxx/include/nv/target",
            "/home/coder/cccl/build/pytorch/cuda/include/cccl/cub/device/device_reduce.cuh": "cub/cub/device/device_reduce.cuh",
            "/home/coder/cccl/build/pytorch/cuda/include/cccl/thrust/detail/config.h": "thrust/thrust/detail/config.h",
            "/home/coder/cccl/build/pytorch/cccl-install/include/cuda/std/tuple": "libcudacxx/include/cuda/std/tuple",
            "/usr/local/cuda/include/cccl/cuda/std/tuple": None,
            "/home/coder/.conda/envs/rapids/include/cuda_runtime.h": None,
            "/home/coder/cudf/include/cudf/table.hpp": None,
            "/home/coder/cudf/build/_deps/rmm-src/include/rmm/cuda_stream.hpp": None,
            "/home/coder/cudf/build/_deps/cccl-src/build/generated.h": None,
            "/home/coder/../elsewhere/build/_deps/cccl-src/cub/cub/foo.cuh": None,
            "/home/coder/cudf/build/_deps/cccl-src/../../other.h": None,
            "_deps/cccl-src/libcudacxx/include/cuda/std/tuple": None,
        }
        for detail, expected in cases.items():
            with self.subTest(detail=detail):
                self.assertEqual(
                    reports.cccl_header(detail, self.source_root), expected
                )

    def test_symbol_ownership(self):
        for symbol in (
            "cuda::std::tuple [cuda::std::tuple<int>]",
            "const cuda::std::tuple<int> &cuda::std::get<0>(cuda::std::tuple<int> &)",
            "void thrust::sort<int>(int *, int *)",
            "cub::BlockReduce<int>::Reduce()",
            "::cccl::detail::helper()",
            "cuda::std::operator<(cuda::std::byte, cuda::std::byte)",
            "_ZN4cuda3std3fooIiEEvv",
            "_ZNK3cub3Foo3barEv",
        ):
            with self.subTest(symbol=symbol):
                self.assertTrue(reports.cccl_symbol(symbol))
        for symbol in (
            "std::tuple [std::tuple<cuda::std::tuple<int>>]",
            "cuda::std::tuple<int> matx::foo()",
            "cuda::std::tuple<int> &matx::foo()",
            "void at::cuda::foo(cuda::std::tuple<int>)",
            "c10::cuda::warning_state()",
            "std::operator<<(std::ostream &, cuda::std::tuple<int>)",
            "_ZN3std3fooIN4cuda3std3FooEEEvv",
            "_Z3fooN4cuda3std3FooE",
            "cudaMalloc(void **, size_t)",
        ):
            with self.subTest(symbol=symbol):
                self.assertFalse(reports.cccl_symbol(symbol))

    def test_cccl_only_timing_and_primary_grouping(self):
        event = self.builder.event
        specialization = "cuda::std::trait [cuda::std::trait<int>]"
        self.write_trace(
            "one",
            [
                event(
                    "Processing Header File",
                    "/home/coder/cudf/build/_deps/cccl-src/libcudacxx/include/cuda/std/tuple",
                    0,
                    100,
                ),
                event("Processing Header File", "/usr/include/stdlib.h", 10, 30),
                event("Scanning Function Body", "void cuda::std::foo()", 50, 10),
                event("Other work", "child", 51, 3),
                event(
                    "Scanning Function Body",
                    "void cudf::foo(cuda::std::tuple<int>)",
                    70,
                    10,
                ),
                event("Instantiating Template Class", specialization, 110, 12),
                event(
                    "Instantiating Template Class",
                    "cuda::std::trait [cuda::std::trait<float>]",
                    130,
                    14,
                ),
                event(
                    "Instantiating Template Class",
                    "std::trait [std::trait<cuda::std::tuple<int>>]",
                    150,
                    100,
                ),
            ],
        )
        self.write_trace(
            "two",
            [
                event(
                    "Processing Header File",
                    "/home/coder/raft/build/_deps/cccl-src/libcudacxx/include/cuda/std/tuple",
                    0,
                    100,
                ),
                event(
                    "Instantiating Template Class",
                    "cuda::std::trait [cuda::std::trait<float>]",
                    110,
                    16,
                ),
            ],
        )
        with patch.object(reports, "read_trace", wraps=reports.read_trace) as parse:
            self.assertEqual(self.run_summary(), 0)
            self.assertEqual(parse.call_count, 2)
        headers = self.rows("file-processing")
        self.assertEqual(list(headers), ["libcudacxx/include/cuda/std/tuple"])
        self.assertEqual(
            float(next(iter(headers.values()))["total_exclusive_s"]), 170 / 1e6
        )
        bodies = self.rows("function-bodies")
        self.assertEqual(list(bodies), ["void cuda::std::foo()"])
        self.assertEqual(
            float(next(iter(bodies.values()))["total_inclusive_s"]), 10 / 1e6
        )
        self.assertEqual(
            float(next(iter(bodies.values()))["total_exclusive_s"]), 7 / 1e6
        )
        grouped = self.rows("primary-template-instantiation")["cuda::std::trait"]
        self.assertEqual(float(grouped["total_inclusive_s"]), 42 / 1e6)
        self.assertEqual(
            (grouped["event_count"], grouped["trace_count"], grouped["root_tu_count"]),
            ("3", "2", "2"),
        )
        individual = self.rows("template-instantiation")[specialization]
        self.assertEqual(
            (
                individual["event_count"],
                individual["trace_count"],
                individual["root_tu_count"],
            ),
            ("1", "1", "1"),
        )
        manifest = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(manifest["trace_count"], 2)
        self.assertEqual(manifest["failed_traces"], [])
        self.assertIn("includes non-CCCL work", manifest["slices"][0]["title"])

    def test_global_top_100_and_group_before_limit(self):
        event = self.builder.event
        for index in range(2):
            # These disjoint one-off symbols outrank the repeated symbol in each
            # individual trace, but not after aggregating the complete dataset.
            entries = [
                event(
                    "Instantiating Template Class",
                    f"cuda::std::one_off_{index}_{n} [int]",
                    n * 20,
                    10,
                )
                for n in range(101)
            ]
            entries.extend(
                [
                    event(
                        "Instantiating Template Class",
                        "cuda::std::repeated [int]",
                        3000,
                        6,
                    ),
                    event(
                        "Instantiating Template Class",
                        f"cuda::std::grouped [{index}]",
                        3020,
                        6,
                    ),
                ]
            )
            self.write_trace(str(index), entries)
        self.assertEqual(self.run_summary(), 0)
        self.assertEqual(
            next(iter(self.rows("template-instantiation"))), "cuda::std::repeated [int]"
        )
        grouped = self.rows("primary-template-instantiation")
        self.assertEqual(grouped["cuda::std::grouped"]["rank"], "1")
        self.assertEqual(
            float(grouped["cuda::std::grouped"]["total_inclusive_s"]), 12 / 1e6
        )
        manifest = json.loads((self.output / "summary.json").read_text())
        for section in manifest["slices"]:
            self.assertLessEqual(len(section["rows"]), 100)
            self.assertLessEqual(len(csv_rows(self.output / section["csv"])), 100)
        self.assertEqual(len(self.rows("template-instantiation")), 100)
        self.assertEqual(len(grouped), 100)
        self.assertEqual({p.suffix for p in self.output.iterdir()}, {".csv", ".json"})
        self.assertEqual(len(list(self.output.iterdir())), 6)

    def test_partial_failure_and_phase_cleanup(self):
        self.write_trace(
            "good",
            [
                self.builder.event(
                    "Instantiating Template Class", "cuda::std::tuple [int]", 0, 10
                )
            ],
        )
        (self.traces / "broken.o.json").write_text('{"traceEvents":', encoding="utf-8")
        (self.traces / "unrelated.json").write_text("keep", encoding="utf-8")
        (self.traces / "latest").symlink_to(self.traces, target_is_directory=True)
        (self.traces / "alias.o.json").symlink_to(self.traces / "good.o.json")
        self.assertEqual(self.run_summary(delete_traces=True), 1)
        manifest = json.loads((self.output / "summary.json").read_text())
        self.assertEqual(manifest["trace_count"], 1)
        self.assertEqual(
            [f["trace"] for f in manifest["failed_traces"]], ["broken.o.json"]
        )
        self.assertEqual(len(self.rows("template-instantiation")), 1)
        self.assertFalse((self.traces / "good.o.json").exists())
        self.assertFalse((self.traces / "broken.o.json").exists())
        self.assertTrue((self.traces / "unrelated.json").exists())
        self.assertTrue((self.traces / "alias.o.json").is_symlink())
        # The ON phase cannot inherit an unchanged trace from OFF.
        self.output = self.work / "on"
        self.assertEqual(self.run_summary(), 1)
        self.assertEqual(self.rows("template-instantiation"), {})

    def test_no_traces_or_only_foreign_work_is_visible_failure(self):
        self.assertEqual(self.run_summary(), 1)
        manifest = json.loads((self.output / "summary.json").read_text())
        self.assertIn("No object-adjacent traces found", manifest["warnings"])
        self.write_trace(
            "foreign",
            [
                self.builder.event(
                    "Instantiating Template Class", "std::tuple [int]", 0, 10
                )
            ],
        )
        self.assertEqual(self.run_summary(), 1)
        manifest = json.loads((self.output / "summary.json").read_text())
        self.assertIn(
            "No CCCL headers or templates matched; check trace paths and filtering",
            manifest["warnings"],
        )

    def test_failed_report_write_keeps_traces(self):
        self.write_trace(
            "good",
            [
                self.builder.event(
                    "Instantiating Template Class", "cuda::std::tuple [int]", 0, 10
                )
            ],
        )
        with patch.object(
            reports.events, "write_json", side_effect=OSError("Disk full")
        ):
            with self.assertRaisesRegex(OSError, "Disk full"):
                self.run_summary(delete_traces=True)
        self.assertTrue((self.traces / "good.o.json").exists())
        self.assertFalse((self.output / "summary.json").exists())

    def test_streaming_json_boundaries(self):
        entries = [
            self.builder.event(
                "Scanning Function Body", 'void cuda::std::quoted_é("},]\\")', 1, 20
            )
        ]
        # Metadata may precede or follow events, and events need not occupy a line.
        for data in (
            {
                "otherData": {"inputFiles": ["a.cu"]},
                "traceEvents": entries,
                "version": 1.25e20,
            },
            {
                "traceEvents": entries,
                "version": -1.25e-20,
                "otherData": {"inputFiles": ["a.cu"]},
            },
        ):
            for chunk_size in (1, 2, 3, 7, 64, 65536):
                with self.subTest(chunk_size=chunk_size, data=data):
                    metadata = {}

                    class BoundedRead(io.StringIO):
                        def read(self, size=-1):
                            if not 0 < size <= max(chunk_size, 4 * 1024 * 1024):
                                raise AssertionError("Unbounded read")
                            return super().read(size)

                    self.assertEqual(
                        list(
                            iter_trace_events(
                                BoundedRead(json.dumps(data, ensure_ascii=False)),
                                metadata,
                                chunk_size=chunk_size,
                            )
                        ),
                        entries,
                    )
                    self.assertEqual(
                        metadata, {k: v for k, v in data.items() if k != "traceEvents"}
                    )
        for malformed in (
            '{"traceEvents":[{}',
            '{"traceEvents": [{} ,]}',
            '{"traceEvents":[] trailing}',
            '{"traceEvents":[]} junk',
        ):
            with self.subTest(malformed=malformed), self.assertRaises(ValueError):
                list(iter_trace_events(io.StringIO(malformed), {}, chunk_size=3))

        class LimitedReads(io.StringIO):
            remaining = 2

            def read(self, size=-1):
                self.remaining -= 1
                if self.remaining < 0:
                    raise AssertionError("Read past a definite syntax error")
                return super().read(size)

        with self.assertRaises(json.JSONDecodeError):
            list(
                iter_trace_events(
                    LimitedReads('{"traceEvents":[{"name": invalid' + "x" * 10000),
                    {},
                    chunk_size=64,
                )
            )

    def test_memory_failure_preserves_partial_reports(self):
        for index in range(2):
            self.write_trace(
                str(index),
                [
                    self.builder.event(
                        "Instantiating Template Class", "cuda::std::tuple [int]", 0, 10
                    )
                ],
            )
        original = reports.summarize_trace

        def fail_second(path, *args):
            if path.name == "1.o.json":
                raise MemoryError()
            return original(path, *args)

        with patch.object(reports, "summarize_trace", side_effect=fail_second):
            self.assertEqual(self.run_summary(), 1)
        manifest = json.loads((self.output / "summary.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["trace_count"], 1)
        self.assertEqual(manifest["input_trace_count"], 2)
        self.assertEqual(
            manifest["failed_traces"], [{"trace": "1.o.json", "error": "MemoryError"}]
        )

    def test_large_foreign_symbol_keeps_exclusive_timing(self):
        self.write_trace(
            "large",
            [
                self.builder.event(
                    "Scanning Function Body", "void cuda::std::foo()", 0, 100
                ),
                self.builder.event(
                    "Generating Function IR",
                    "foreign::" + ('quoted_"}[\\]' * 200000),
                    10,
                    50,
                ),
                self.builder.event(
                    "Instantiating Template Class", "cuda::std::tuple [int]", 110, 10
                ),
            ],
        )
        self.assertEqual(self.run_summary(), 0)
        body = self.rows("function-bodies")["void cuda::std::foo()"]
        self.assertEqual(float(body["total_inclusive_s"]), 100 / 1e6)
        self.assertEqual(float(body["total_exclusive_s"]), 50 / 1e6)

    def test_failed_manifest_write_keeps_previous_report(self):
        self.write_trace(
            "good",
            [
                self.builder.event(
                    "Instantiating Template Class", "cuda::std::tuple [int]", 0, 10
                )
            ],
        )
        self.assertEqual(self.run_summary(), 0)
        previous = (self.output / "summary.json").read_bytes()

        def fail_write(path, payload):
            path.write_text('{"partial":', encoding="utf-8")
            raise OSError("Disk full")

        with patch.object(reports.events, "write_json", side_effect=fail_write):
            with self.assertRaisesRegex(OSError, "Disk full"):
                self.run_summary()
        self.assertEqual((self.output / "summary.json").read_bytes(), previous)

    def test_interrupted_preprocessing_preserves_checkpoint(self):
        for index in range(26):
            self.write_trace(
                f"{index:02}",
                [
                    self.builder.event(
                        "Instantiating Template Class", "cuda::std::tuple [int]", 0, 10
                    )
                ],
            )
        original = reports.summarize_trace

        def interrupt_last(path, *args):
            if path.name == "25.o.json":
                raise KeyboardInterrupt()
            return original(path, *args)

        with patch.object(reports, "summarize_trace", side_effect=interrupt_last):
            with self.assertRaises(KeyboardInterrupt):
                self.run_summary()
        manifest = json.loads((self.output / "summary.json").read_text())
        self.assertFalse(manifest["complete"])
        self.assertEqual(manifest["trace_count"], 25)
        self.assertEqual(manifest["input_trace_count"], 26)
        self.assertEqual(len(self.rows("template-instantiation")), 1)


if __name__ == "__main__":
    unittest.main()
