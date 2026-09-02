# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

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


@pytest.fixture
def pathfinder(monkeypatch):
    module = ModuleType("cuda.pathfinder")
    monkeypatch.setitem(sys.modules, "cuda.pathfinder", module)
    return module


def _write_cuda_header(include_dir: Path, encoded_version: int) -> Path:
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_text(
        f"#define CUDART_VERSION {encoded_version}\n",
        encoding="utf-8",
    )
    return include_dir


def _library_path(
    directory: Path,
    kind: str,
    major: int,
    minor: int | None = None,
) -> Path:
    """Return the platform spelling used by the preload implementation."""

    names = (
        _toolkit._nvrtc_builtins_names(major, minor)
        if kind == "nvrtc-builtins" and minor is not None
        else _toolkit._library_names(kind, major)
    )
    return directory / names[0]


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


@pytest.mark.parametrize(
    ("os_name", "expected"),
    (
        ("posix", ("libnvrtc-builtins.so.13.2",)),
        ("nt", ("nvrtc-builtins64_132.dll",)),
    ),
)
def test_nvrtc_builtins_names_match_platform_spelling(
    monkeypatch,
    os_name,
    expected,
):
    monkeypatch.setattr(_toolkit, "os", SimpleNamespace(name=os_name))

    assert _toolkit._nvrtc_builtins_names(13, 2) == expected


def test_preload_loads_nvrtc_builtins_before_nvrtc(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    builtins = _library_path(lib_dir, "nvrtc-builtins", 13, 0)
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    nvjitlink = _library_path(lib_dir, "nvJitLink", 13)
    for library in (builtins, nvrtc, nvjitlink):
        library.touch()

    loaded = []
    fake_nvjitlink = _FakeNvJitLinkLibrary((13, 0))

    def load_library(path, *, mode):
        path = Path(path)
        loaded.append((path, mode))
        return fake_nvjitlink if path == nvjitlink else object()

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
        raising=False,
    )

    _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert [path for path, _ in loaded] == [builtins, nvrtc, nvjitlink]
    assert str(builtins.resolve()) in _toolkit._EXACT_LIBRARY_HANDLES


def test_preload_does_not_mix_nvrtc_and_builtins_roots(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    first_include = tmp_path / "first" / "include"
    second_include = tmp_path / "second" / "include"
    first_lib = tmp_path / "first" / "lib"
    second_lib = tmp_path / "second" / "lib"
    _write_cuda_header(first_include, 13000)
    _write_cuda_header(second_include, 13000)
    first_lib.mkdir()
    second_lib.mkdir()

    first_builtins = _library_path(first_lib, "nvrtc-builtins", 13, 0)
    first_nvrtc = _library_path(first_lib, "nvrtc", 13)
    second_builtins = _library_path(second_lib, "nvrtc-builtins", 13, 0)
    nvrtc = _library_path(second_lib, "nvrtc", 13)
    nvjitlink = _library_path(second_lib, "nvJitLink", 13)
    for library in (first_nvrtc, second_builtins, nvrtc, nvjitlink):
        library.touch()

    loaded = []
    fake_nvjitlink = _FakeNvJitLinkLibrary((13, 0))

    def load_library(path, *, mode):
        del mode
        path = Path(path)
        loaded.append(path)
        return fake_nvjitlink if path == nvjitlink else object()

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
        raising=False,
    )

    _toolkit.preload_toolkit_compiler_libraries((first_include, second_include))

    assert loaded == [second_builtins, nvrtc, nvjitlink]
    assert first_builtins not in loaded
    assert first_nvrtc not in loaded


def test_preload_reports_adjacent_nvrtc_builtins_failure(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    builtins = _library_path(lib_dir, "nvrtc-builtins", 13, 0)
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    nvjitlink = _library_path(lib_dir, "nvJitLink", 13)
    for library in (builtins, nvrtc, nvjitlink):
        library.touch()

    attempts = []

    def fail_builtins(path, *, mode):
        del mode
        path = Path(path)
        attempts.append(path)
        if path == builtins:
            raise OSError("builtins load failed")
        return object()

    fallback_calls = []
    monkeypatch.setattr(_toolkit.ctypes, "CDLL", fail_builtins)
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: fallback_calls.append(name),
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="failed loading all resolved CUDA Toolkit NVRTC and builtins candidates",
    ) as exc_info:
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert isinstance(exc_info.value.__cause__, OSError)
    assert str(exc_info.value.__cause__) == "builtins load failed"
    assert attempts == [builtins]
    assert fallback_calls == []


def test_preload_stops_if_nvrtc_fails_after_builtins_becomes_global(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dirs = []
    pairs = []
    for name in ("first", "second"):
        include_dir = tmp_path / name / "include"
        lib_dir = tmp_path / name / "lib"
        include_dirs.append(_write_cuda_header(include_dir, 13000))
        lib_dir.mkdir()
        pair = (
            _library_path(lib_dir, "nvrtc-builtins", 13, 0),
            _library_path(lib_dir, "nvrtc", 13),
        )
        for library in pair:
            library.touch()
        pairs.append(pair)

    attempts = []

    def fail_first_nvrtc(path, *, mode):
        del mode
        path = Path(path)
        attempts.append(path)
        if path == pairs[0][1]:
            raise OSError("NVRTC load failed")
        return object()

    fallback_calls = []
    monkeypatch.setattr(_toolkit.ctypes, "CDLL", fail_first_nvrtc)
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: fallback_calls.append(name),
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="failed loading same-root NVRTC after its builtins",
    ) as exc_info:
        _toolkit.preload_toolkit_compiler_libraries(include_dirs)

    assert isinstance(exc_info.value.__cause__, OSError)
    assert attempts == list(pairs[0])
    assert fallback_calls == []


def test_preload_rejects_nvrtc_without_adjacent_builtins(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    nvjitlink = _library_path(lib_dir, "nvJitLink", 13)
    nvrtc.touch()
    nvjitlink.touch()

    loaded = []
    fallback_calls = []
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: loaded.append((Path(path), mode)),
    )
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: fallback_calls.append(name),
        raising=False,
    )

    with pytest.raises(
        RuntimeError,
        match="missing adjacent NVRTC builtins",
    ) as exc_info:
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert isinstance(exc_info.value.__cause__, FileNotFoundError)
    assert "libnvrtc-builtins.so.13.0" in str(exc_info.value)
    assert loaded == []
    assert fallback_calls == []


def test_preload_reuses_nvrtc_builtins_and_compiler_libraries(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    builtins = _library_path(lib_dir, "nvrtc-builtins", 13, 0)
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    nvjitlink = _library_path(lib_dir, "nvJitLink", 13)
    for library in (builtins, nvrtc, nvjitlink):
        library.touch()

    loaded = []
    fake_nvjitlink = _FakeNvJitLinkLibrary((13, 0))

    def load_library(path, *, mode):
        del mode
        path = Path(path)
        loaded.append(path)
        return fake_nvjitlink if path == nvjitlink else object()

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
        raising=False,
    )

    for _ in range(2):
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert loaded == [builtins, nvrtc, nvjitlink]


def test_preload_reuses_exact_nvjitlink_handle(monkeypatch, tmp_path, pathfinder):
    first_include = tmp_path / "first" / "include"
    second_include = tmp_path / "second" / "include"
    first_lib = tmp_path / "first" / "lib"
    second_lib = tmp_path / "second" / "lib64"
    _write_cuda_header(first_include, 13000)
    _write_cuda_header(second_include, 13000)
    first_lib.mkdir()
    second_lib.mkdir()

    builtins = _library_path(first_lib, "nvrtc-builtins", 13, 0)
    nvrtc = _library_path(first_lib, "nvrtc", 13)
    nvjitlink = _library_path(second_lib, "nvJitLink", 13)
    ignored = _library_path(first_lib, "nvrtc", 12)
    for library in (builtins, nvrtc, nvjitlink, ignored):
        library.touch()

    loaded = []
    fake_nvjitlink = _FakeNvJitLinkLibrary((13, 0))

    def load_library(path, *, mode):
        path = Path(path)
        loaded.append((path, mode))
        return fake_nvjitlink if path == nvjitlink else object()

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else nvjitlink)
        ),
        raising=False,
    )

    libraries = _toolkit.preload_toolkit_compiler_libraries(
        (first_include, second_include)
    )

    assert [path for path, _ in loaded] == [builtins, nvrtc, nvjitlink]
    assert sum(path == nvjitlink for path, _ in loaded) == 1
    assert libraries.nvrtc_path == str(nvrtc)
    assert libraries.nvrtc_builtins_path == str(builtins)
    assert libraries.nvjitlink_path == str(nvjitlink)
    assert libraries.toolkit_version == (13, 0)


def test_preload_accepts_compatible_newer_nvjitlink_fallback(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    builtins = _library_path(lib_dir, "nvrtc-builtins", 13, 0)
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    builtins.touch()
    nvrtc.touch()

    fallback_nvjitlink = _library_path(tmp_path / "fallback", "nvJitLink", 13)
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: _FakeNvJitLinkLibrary((13, 3)),
    )
    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else fallback_nvjitlink)
        ),
        raising=False,
    )

    libraries = _toolkit.preload_toolkit_compiler_libraries((include_dir,))

    assert libraries.nvjitlink_path == str(fallback_nvjitlink)


@pytest.mark.parametrize("actual_version", [(12, 8), (13, 0)])
def test_preload_rejects_version_mismatched_nvjitlink_fallback(
    monkeypatch,
    tmp_path,
    actual_version,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    _write_cuda_header(include_dir, 13020)
    lib_dir = tmp_path / "toolkit" / "lib"
    lib_dir.mkdir()
    _library_path(lib_dir, "nvrtc-builtins", 13, 2).touch()
    nvrtc = _library_path(lib_dir, "nvrtc", 13)
    nvrtc.touch()
    fallback_nvjitlink = _library_path(
        tmp_path / "fallback", "nvJitLink", actual_version[0]
    )

    monkeypatch.setattr(
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else fallback_nvjitlink)
        ),
        raising=False,
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


def test_preload_tries_all_exact_toolkit_candidates(
    monkeypatch,
    tmp_path,
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    lib64_dir = tmp_path / "toolkit" / "lib64"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    lib64_dir.mkdir()
    _library_path(lib_dir, "nvrtc-builtins", 13, 0).touch()
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
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: SimpleNamespace(
            abs_path=str(nvrtc if name == "nvrtc" else second_nvjitlink)
        ),
        raising=False,
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
    pathfinder,
):
    include_dir = tmp_path / "toolkit" / "include"
    lib_dir = tmp_path / "toolkit" / "lib"
    lib64_dir = tmp_path / "toolkit" / "lib64"
    _write_cuda_header(include_dir, 13000)
    lib_dir.mkdir()
    lib64_dir.mkdir()
    _library_path(lib_dir, "nvrtc-builtins", 13, 0).touch()
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
        pathfinder,
        "load_nvidia_dynamic_lib",
        lambda name: fallback_calls.append(name),
        raising=False,
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
        nvrtc_builtins_path="/toolkit/lib/libnvrtc-builtins.so.12.8",
        nvjitlink_path="/toolkit/lib/libnvJitLink.so.12",
        toolkit_version=(13, 2),
    )

    with pytest.raises(RuntimeError, match="headers report Toolkit 13.2"):
        _toolkit.validate_nvrtc_version(libraries, (12, 8))


def _libraries(version: tuple[int, int] | None) -> _toolkit.ToolkitCompilerLibraries:
    return _toolkit.ToolkitCompilerLibraries(
        nvrtc_path="/toolkit/lib/libnvrtc.so",
        nvrtc_builtins_path="/toolkit/lib/libnvrtc-builtins.so",
        nvjitlink_path="/toolkit/lib/libnvJitLink.so",
        toolkit_version=version,
    )


def test_toolkit_version_rejects_disagreeing_header_roots(tmp_path: Path) -> None:
    first = _write_cuda_header(tmp_path / "cuda-12" / "include", 12080)
    second = _write_cuda_header(tmp_path / "cuda-13" / "include", 13020)

    with pytest.raises(RuntimeError, match="disagree on Toolkit version: 12.8, 13.2"):
        _toolkit._toolkit_version((first, second))


def test_toolkit_version_rejects_unparseable_selected_header(
    tmp_path: Path,
) -> None:
    valid = _write_cuda_header(tmp_path / "valid" / "include", 13020)
    malformed = tmp_path / "malformed" / "include"
    malformed.mkdir(parents=True)
    malformed_header = malformed / "cuda_runtime_api.h"
    malformed_header.write_text("#define CUDA_VERSION 13020\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="failed parsing CUDART_VERSION") as exc_info:
        _toolkit._toolkit_version((valid, malformed))

    assert str(malformed_header) in str(exc_info.value)


@pytest.mark.parametrize("actual_version", ((12, 8), (13, 1), (13, 3)))
def test_nvrtc_version_must_match_headers_exactly(
    actual_version: tuple[int, int],
) -> None:
    libraries = _libraries((13, 2))

    with pytest.raises(RuntimeError, match="loaded NVRTC"):
        _toolkit.validate_nvrtc_version(libraries, actual_version)


@pytest.mark.parametrize("actual_version", ((12, 8), (13, 1), (14, 0)))
def test_nvjitlink_version_must_be_compatible_with_headers(
    actual_version: tuple[int, int],
) -> None:
    libraries = _libraries((13, 2))

    with pytest.raises(RuntimeError, match="same major release and be no older"):
        _toolkit.validate_nvjitlink_version(libraries, actual_version)


def test_matching_nvrtc_and_newer_same_major_nvjitlink_are_accepted() -> None:
    libraries = _libraries((13, 2))

    _toolkit.validate_nvrtc_version(libraries, (13, 2))
    _toolkit.validate_nvjitlink_version(libraries, (13, 2))
    _toolkit.validate_nvjitlink_version(libraries, (13, 3))
