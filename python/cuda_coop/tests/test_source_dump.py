# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor

from cuda.coop._core._source_dump import dump_source


def test_source_dump_distinguishes_source_target_and_backend(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_COOP_SOURCE_DUMP_DIR", str(tmp_path))
    source = "// CUDA source with a Unicode comment: π\n"
    first = dump_source(source, backend="numba_mlir", identity=(90, "lto"))
    repeated = dump_source(source, backend="numba_mlir", identity=(90, "lto"))
    changed_source = dump_source(
        "// different\n", backend="numba_mlir", identity=(90, "lto")
    )
    changed_target = dump_source(
        source, backend="numba_mlir", identity=(120, "lto")
    )
    changed_backend = dump_source(
        source, backend="cutlass", identity=(90, "lto")
    )

    assert first == repeated
    assert len({first, changed_source, changed_target, changed_backend}) == 4
    assert first.read_bytes() == source.encode("utf-8")


def test_source_dump_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_COOP_SOURCE_DUMP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert dump_source("// source", backend="numba_mlir") is None
    assert not list(tmp_path.iterdir())


def test_concurrent_source_dumps_leave_complete_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_COOP_SOURCE_DUMP_DIR", str(tmp_path))
    source = "// shared translation unit\n" * 1000

    def write(_):
        return dump_source(source, backend="numba_mlir")

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = set(executor.map(write, range(12)))

    assert len(paths) == 1
    assert set(tmp_path.iterdir()) == paths
    assert paths.pop().read_bytes() == source.encode("utf-8")
