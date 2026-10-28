# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor

from cuda.coop.cutlass._planning._source_dump import dump_source


def test_source_dump_distinguishes_source_target_and_mode(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("CUDA_COOP_SOURCE_DUMP_DIR", str(tmp_path))
    source = "// CUDA source with a Unicode comment: π\n"
    first = dump_source(source, backend="cutlass", identity=(90, "lto"))
    repeated = dump_source(source, backend="cutlass", identity=(90, "lto"))
    changed_source = dump_source(
        "// different\n", backend="cutlass", identity=(90, "lto")
    )
    changed_target = dump_source(
        source, backend="cutlass", identity=(120, "lto")
    )
    changed_mode = dump_source(
        source, backend="cutlass", identity=(90, "cubin")
    )

    assert first == repeated
    assert len({first, changed_source, changed_target, changed_mode}) == 4
    assert first.read_bytes() == source.encode("utf-8")


def test_source_dump_disabled_by_default(tmp_path, monkeypatch):
    monkeypatch.delenv("CUDA_COOP_SOURCE_DUMP_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    assert dump_source("// source", backend="cutlass") is None
    assert not list(tmp_path.iterdir())


def test_concurrent_source_dumps_leave_complete_files(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_COOP_SOURCE_DUMP_DIR", str(tmp_path))
    source = "// shared translation unit\n" * 1000

    def write(_):
        return dump_source(source, backend="cutlass")

    with ThreadPoolExecutor(max_workers=4) as executor:
        paths = set(executor.map(write, range(12)))

    assert len(paths) == 1
    assert set(tmp_path.iterdir()) == paths
    assert paths.pop().read_bytes() == source.encode("utf-8")
