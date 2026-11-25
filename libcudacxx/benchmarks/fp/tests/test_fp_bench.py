# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
"""Unit tests of fp_bench.py: the build-tree layout and the CMake cache reading.

pytest -q libcudacxx/benchmarks/fp/tests
"""

import os

import fp_bench


def test_preset_build_dir_follows_the_presets(monkeypatch):
    monkeypatch.delenv("CCCL_BUILD_INFIX", raising=False)
    assert fp_bench.preset_build_dir("/src") == os.path.join(
        "/src", "build", "libcudacxx-benchmark"
    )
    monkeypatch.setenv("CCCL_BUILD_INFIX", "cuda13.0-gcc14")
    assert fp_bench.preset_build_dir("/src") == os.path.join(
        "/src", "build", "cuda13.0-gcc14", "libcudacxx-benchmark"
    )
    monkeypatch.setenv("CCCL_BUILD_INFIX", "")
    assert fp_bench.preset_build_dir("/src") == os.path.join(
        "/src", "build", "libcudacxx-benchmark"
    )


def test_cache_vars(tmp_path):
    assert fp_bench.cache_vars(str(tmp_path)) == {}
    (tmp_path / "CMakeCache.txt").write_text(
        "// comment\nCMAKE_CUDA_ARCHITECTURES:STRING=native\nOTHER:BOOL=ON\n"
        "CMAKE_HOME_DIRECTORY:INTERNAL=/src/cccl\nCPM_DIRECTORY:INTERNAL=/src/cccl/cmake\n"
    )
    assert fp_bench.cache_vars(str(tmp_path)) == {
        "CMAKE_CUDA_ARCHITECTURES": "native",
        "CMAKE_HOME_DIRECTORY": "/src/cccl",
        "CPM_DIRECTORY": "/src/cccl/cmake",
    }


def test_tree_is_current_checks_the_checkout_and_cpm_directory(tmp_path):
    (tmp_path / "build.ninja").write_text("")
    root = str(tmp_path / "cccl")
    (tmp_path / "cccl").mkdir()
    have = {"CMAKE_HOME_DIRECTORY": root, "CPM_DIRECTORY": root + "/cmake"}
    assert fp_bench.tree_is_current(have, root, str(tmp_path))
    # the source recorded under a symlinked path still names this checkout
    link = tmp_path / "link"
    link.symlink_to(tmp_path / "cccl")
    assert fp_bench.tree_is_current(
        {**have, "CMAKE_HOME_DIRECTORY": str(link)}, root, str(tmp_path)
    )
    # but CPM's cached directory must be the one its module is included from now, or the
    # regeneration fails with "Unknown CMake command CPMAddPackage"
    assert not fp_bench.tree_is_current(
        {**have, "CPM_DIRECTORY": "/elsewhere/cccl/cmake"}, root, str(tmp_path)
    )
    # another checkout, no cache, or a cache without a generated build: not current
    assert not fp_bench.tree_is_current(have, str(tmp_path / "other"), str(tmp_path))
    assert not fp_bench.tree_is_current({}, root, str(tmp_path))
    assert not fp_bench.tree_is_current(have, root, str(tmp_path / "nowhere"))


def test_families_lists_the_cu_files(tmp_path):
    src = tmp_path / "libcudacxx" / "benchmarks" / "bench" / "fp" / "fpmp"
    src.mkdir(parents=True)
    for name in ("arith.cu", "trig.cu", "fpmp_bench.cuh"):
        (src / name).write_text("")
    assert fp_bench.families(str(tmp_path), "fpmp", None) == ["arith", "trig"]
    assert fp_bench.families(str(tmp_path), "fpmp", ["trig"]) == ["trig"]


def test_storage_probes(tmp_path):
    # a small fake checkout: the read probe returns a per-file figure and touches only headers
    hdr = tmp_path / "libcudacxx" / "include" / "cuda" / "std" / "__bit"
    hdr.mkdir(parents=True)
    for i in range(3):
        (hdr / f"h{i}.h").write_text("// header\n")
    assert fp_bench.probe_reads(str(tmp_path)) >= 0.0
    assert fp_bench.probe_reads(str(tmp_path / "nowhere")) == 0.0
    # the write probe leaves nothing behind
    assert fp_bench.probe_writes(str(tmp_path), count=3) >= 0.0
    assert sorted(p.name for p in tmp_path.iterdir()) == ["libcudacxx"]


def test_fs_type_of_root_is_known_on_linux():
    if not os.path.exists("/proc/mounts"):
        return
    assert fp_bench.fs_type("/") != ""
    assert fp_bench.fs_type("/definitely/not/a/mount/point") == fp_bench.fs_type("/")
