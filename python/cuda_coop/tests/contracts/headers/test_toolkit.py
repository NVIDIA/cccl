# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from pathlib import Path
from types import SimpleNamespace

import cuda.pathfinder
import pytest

from cuda.coop._headers import _toolkit


class _FakeNvJitLinkVersion:
    def __init__(self, version):
        self.version = version
        self.argtypes = None
        self.restype = None

    def __call__(self, major, minor):
        pointer_type = _toolkit.ctypes.POINTER(_toolkit.ctypes.c_uint)
        assert self.argtypes == (pointer_type, pointer_type)
        assert self.restype is _toolkit.ctypes.c_int
        _toolkit.ctypes.cast(major, pointer_type)[0] = self.version[0]
        _toolkit.ctypes.cast(minor, pointer_type)[0] = self.version[1]
        return 0


class _FakeNvJitLinkLibrary:
    def __init__(self, version):
        self.nvJitLinkVersion = _FakeNvJitLinkVersion(version)


@pytest.fixture(autouse=True)
def _isolated_exact_library_handles():
    with _toolkit._PRELOAD_LOCK:
        _toolkit._EXACT_LIBRARY_HANDLES.clear()
    yield
    with _toolkit._PRELOAD_LOCK:
        _toolkit._EXACT_LIBRARY_HANDLES.clear()


def _write_cuda_header(include_dir: Path, encoded_version: int) -> None:
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_text(
        f"#define CUDART_VERSION {encoded_version}\n",
        encoding="utf-8",
    )


def _library_path(directory: Path, kind: str, major: int) -> Path:
    """Return the platform spelling used by the preload implementation."""

    return directory / _toolkit._library_names(kind, major)[0]


@pytest.mark.parametrize(
    ("os_name", "kind", "expected"),
    (
        ("posix", "nvrtc", ("libnvrtc.so.13",)),
        ("posix", "nvJitLink", ("libnvJitLink.so.13",)),
        ("nt", "nvrtc", ("nvrtc64_130_0.dll",)),
        ("nt", "nvJitLink", ("nvJitLink_130_0.dll",)),
    ),
)
def test_library_names_match_platform_spelling(
    monkeypatch,
    os_name,
    kind,
    expected,
):
    monkeypatch.setattr(_toolkit, "os", SimpleNamespace(name=os_name))

    assert _toolkit._library_names(kind, 13) == expected


def test_preload_reuses_exact_nvjitlink_handle(monkeypatch, tmp_path):
    first_include = tmp_path / "first" / "include"
    second_include = tmp_path / "second" / "include"
    first_lib = tmp_path / "first" / "lib"
    second_lib = tmp_path / "second" / "lib64"
    _write_cuda_header(first_include, 13000)
    _write_cuda_header(second_include, 13000)
    first_lib.mkdir()
    second_lib.mkdir()

    nvrtc = _library_path(first_lib, "nvrtc", 13)
    nvjitlink = _library_path(second_lib, "nvJitLink", 13)
    ignored = _library_path(first_lib, "nvrtc", 12)
    for library in (nvrtc, nvjitlink, ignored):
        library.touch()

    loaded = []
    fake_nvjitlink = _FakeNvJitLinkLibrary((13, 0))

    def load_library(path, *, mode):
        path = Path(path)
        loaded.append((path, mode))
        return fake_nvjitlink if path == nvjitlink else object()

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
    )
    libraries = _toolkit.preload_toolkit_compiler_libraries(
        (first_include, second_include)
    )

    assert [path for path, _ in loaded] == [nvrtc, nvjitlink]
    assert sum(path == nvjitlink for path, _ in loaded) == 1
    assert libraries.nvrtc_path == str(nvrtc)
    assert libraries.nvjitlink_path == str(nvjitlink)
    assert libraries.toolkit_version == (13, 0)


def test_preload_accepts_compatible_newer_nvjitlink_fallback(
    monkeypatch,
    tmp_path,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    _library_path(lib_dir, "nvrtc", 13).touch()

    fallback_nvjitlink = _library_path(tmp_path / "fallback", "nvJitLink", 13)
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: _FakeNvJitLinkLibrary((13, 3)),
    )
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(
                _library_path(lib_dir, "nvrtc", 13)
                if name == "nvrtc"
                else fallback_nvjitlink
            )
        ),
    )
    libraries = _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert libraries.nvjitlink_path == str(fallback_nvjitlink)


@pytest.mark.parametrize("actual_version", [(12, 8), (13, 0)])
def test_preload_rejects_version_mismatched_nvjitlink_fallback(
    monkeypatch,
    tmp_path,
    actual_version,
):
    include_dir = tmp_path / "toolkit" / "include"
    _write_cuda_header(include_dir, 13020)
    fallback_nvrtc = _library_path(tmp_path / "fallback", "nvrtc", 13)
    fallback_nvjitlink = _library_path(
        tmp_path / "fallback", "nvJitLink", actual_version[0]
    )

    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(fallback_nvrtc if name == "nvrtc" else fallback_nvjitlink)
        ),
    )
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: _FakeNvJitLinkLibrary(actual_version),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"headers report Toolkit 13\.2, but loaded nvJitLink .* reports "
            rf"{actual_version[0]}\.{actual_version[1]}"
        ),
    ):
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))


def test_preload_tries_all_exact_toolkit_candidates(monkeypatch, tmp_path):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    lib64_dir = tmp_path / "toolkit" / "lib64"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    lib64_dir.mkdir()
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    first_nvjitlink = _library_path(lib_dir, "nvJitLink", 13)
    second_nvjitlink = _library_path(lib64_dir, "nvJitLink", 13)
    for library in (nvrtc, first_nvjitlink, second_nvjitlink):
        library.touch()

    attempts = []
    fake_nvjitlink = _FakeNvJitLinkLibrary((13, 0))

    def load_library(path, *, mode):
        del mode
        path = Path(path)
        attempts.append(path)
        if path == first_nvjitlink:
            raise OSError("first candidate is unavailable")
        return fake_nvjitlink if path == second_nvjitlink else object()

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else second_nvjitlink)
        ),
    )

    libraries = _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert [path for path in attempts if "nvJitLink" in path.name] == [
        first_nvjitlink,
        second_nvjitlink,
    ]
    assert libraries.nvjitlink_path == str(second_nvjitlink)


def test_preload_reports_all_exact_toolkit_library_load_failures(
    monkeypatch,
    tmp_path,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    lib64_dir = tmp_path / "toolkit" / "lib64"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    lib64_dir.mkdir()
    _library_path(lib_dir, "nvrtc", 13).touch()
    candidates = (
        _library_path(lib_dir, "nvJitLink", 13),
        _library_path(lib64_dir, "nvJitLink", 13),
    )
    for candidate in candidates:
        candidate.touch()

    def fail_load(path, *, mode):
        del mode
        if Path(path) in candidates:
            raise OSError(f"test load failure for {Path(path).parent.name}")
        return object()

    fallback_calls = []
    monkeypatch.setattr(_toolkit.ctypes, "CDLL", fail_load)
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: fallback_calls.append(name),
    )
    with pytest.raises(
        RuntimeError,
        match="failed loading all resolved CUDA Toolkit nvJitLink candidates",
    ) as exc_info:
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "test load failure for lib64"
    assert all(str(candidate) in str(exc_info.value) for candidate in candidates)
    assert fallback_calls == []


def test_validate_nvrtc_version_rejects_header_mismatch():
    libraries = _toolkit.ToolkitCompilerLibraries(
        nvrtc_path="/toolkit/lib/libnvrtc.so.12",
        nvjitlink_path="/toolkit/lib/libnvJitLink.so.12",
        toolkit_version=(13, 2),
    )

    with pytest.raises(RuntimeError, match="headers report Toolkit 13.2"):
        _toolkit.validate_nvrtc_version(libraries, (12, 8))
