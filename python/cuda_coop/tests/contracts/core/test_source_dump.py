# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from concurrent.futures import ThreadPoolExecutor

import pytest

from cuda.coop._core._source_dump import dump_source


@pytest.mark.parametrize("backend", ("numba_mlir", "cutlass", "lang"))
def test_source_dump_is_content_addressed(tmp_path, monkeypatch, backend):
    monkeypatch.setenv("CUDA_COOP_SOURCE_DUMP_DIR", str(tmp_path))
    source = "// CUDA source with a Unicode comment: π\n"
    first = dump_source(source, backend=backend, identity=(90, "lto"))
    repeated = dump_source(source, backend=backend, identity=(90, "lto"))
    changed_source = dump_source(
        "// different\n", backend=backend, identity=(90, "lto")
    )
    changed_target = dump_source(source, backend=backend, identity=(120, "lto"))

    assert first == repeated
    assert len({first, changed_source, changed_target}) == 3
    assert first.name.startswith(f"cuda_coop_{backend}_")
    assert first.suffix == ".cu"
    assert first.read_bytes() == source.encode("utf-8")


def test_cross_backend_dump_names(tmp_path, monkeypatch):
    monkeypatch.setenv("CUDA_COOP_SOURCE_DUMP_DIR", str(tmp_path))
    paths = {
        dump_source("// same source\n", backend=backend)
        for backend in ("numba_mlir", "cutlass", "lang")
    }
    assert len(paths) == 3
    assert set(tmp_path.iterdir()) == paths


@pytest.mark.parametrize("shared_value", (None, "", "shared"))
@pytest.mark.parametrize(
    ("backend", "retired_env"),
    (
        ("numba_mlir", "CUDA_COOP_NUMBA_MLIR_NVRTC_DUMP_DIR"),
        ("cutlass", "CUDA_COOP_CUTLASS_PROVIDER_DUMP_DIR"),
    ),
)
def test_retired_settings_are_ignored(
    tmp_path, monkeypatch, shared_value, backend, retired_env
):
    monkeypatch.setenv(retired_env, str(tmp_path / "retired"))
    if shared_value is None:
        monkeypatch.delenv("CUDA_COOP_SOURCE_DUMP_DIR", raising=False)
    else:
        monkeypatch.setenv(
            "CUDA_COOP_SOURCE_DUMP_DIR",
            str(tmp_path / shared_value) if shared_value else "",
        )

    path = dump_source("// source", backend=backend)
    if shared_value:
        assert path.parent == tmp_path / "shared"
        assert path.read_text() == "// source"
    else:
        assert path is None
        assert not list(tmp_path.iterdir())
    assert not (tmp_path / "retired").exists()


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
