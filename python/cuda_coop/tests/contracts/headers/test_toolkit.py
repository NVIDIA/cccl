# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
from pathlib import Path
from types import SimpleNamespace

import cuda.pathfinder
import pytest

from cuda.coop._headers import _toolkit


class _FakeNvrtcVersion:
    def __init__(self, version: tuple[int, int]):
        self.version = version
        self.argtypes = None
        self.restype = None

    def __call__(self, major, minor):
        pointer_type = _toolkit.ctypes.POINTER(_toolkit.ctypes.c_int)
        assert self.argtypes == (pointer_type, pointer_type)
        assert self.restype is _toolkit.ctypes.c_int
        _toolkit.ctypes.cast(major, pointer_type)[0] = self.version[0]
        _toolkit.ctypes.cast(minor, pointer_type)[0] = self.version[1]
        return 0


class _FakeNvJitLinkVersion:
    def __init__(self, version: tuple[int, int]):
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


class _FakeNvrtcLibrary:
    def __init__(self, version: tuple[int, int]):
        self.nvrtcVersion = _FakeNvrtcVersion(version)


class _FakeNvJitLinkLibrary:
    def __init__(self, version: tuple[int, int]):
        self.nvJitLinkVersion = _FakeNvJitLinkVersion(version)


@pytest.fixture(autouse=True)
def _isolated_process_toolkit_state():
    with _toolkit._PRELOAD_LOCK:
        _toolkit._EXACT_LIBRARY_HANDLES.clear()
        _toolkit._PROCESS_TOOLKIT_SELECTION = None
    yield
    with _toolkit._PRELOAD_LOCK:
        _toolkit._EXACT_LIBRARY_HANDLES.clear()
        _toolkit._PROCESS_TOOLKIT_SELECTION = None


def _write_cuda_header(include_dir: Path, encoded_version: int) -> None:
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_text(
        f"#define CUDART_VERSION {encoded_version}\n",
        encoding="utf-8",
    )


def _library_path(directory: Path, kind: str, major: int) -> Path:
    return directory / _toolkit._library_names(kind, major)[0]


def _builtins_path(directory: Path, major: int, minor: int) -> Path:
    return directory / _toolkit._nvrtc_builtins_names(major, minor)[0]


def _write_complete_toolkit(
    root: Path,
    encoded_version: int,
) -> dict[str, Path]:
    major = encoded_version // 1000
    minor = (encoded_version % 1000) // 10
    include_dir = root / "include"
    lib_dir = root / "lib"
    _write_cuda_header(include_dir, encoded_version)
    lib_dir.mkdir()
    paths = {
        "root": root,
        "include": include_dir,
        "nvrtc": _library_path(lib_dir, "nvrtc", major),
        "builtins": _builtins_path(lib_dir, major, minor),
        "nvjitlink": _library_path(lib_dir, "nvJitLink", major),
    }
    for name in ("nvrtc", "builtins", "nvjitlink"):
        paths[name].touch()
    return paths


def _write_split_wheel_toolkit(
    nvidia_root: Path,
    encoded_version: int,
    *,
    library_dir_name: str = "lib",
) -> dict[str, Path]:
    major = encoded_version // 1000
    minor = (encoded_version % 1000) // 10
    include_dir = nvidia_root / "cuda_runtime" / "include"
    nvrtc_dir = nvidia_root / "cuda_nvrtc" / library_dir_name
    nvjitlink_dir = nvidia_root / "nvjitlink" / library_dir_name
    _write_cuda_header(include_dir, encoded_version)
    nvrtc_dir.mkdir(parents=True)
    nvjitlink_dir.mkdir(parents=True)
    paths = {
        "root": nvidia_root,
        "include": include_dir,
        "nvrtc": _library_path(nvrtc_dir, "nvrtc", major),
        "builtins": _builtins_path(nvrtc_dir, major, minor),
        "nvjitlink": _library_path(nvjitlink_dir, "nvJitLink", major),
    }
    for name in ("nvrtc", "builtins", "nvjitlink"):
        paths[name].touch()
    return paths


def _patch_compiler_loaders(
    monkeypatch: pytest.MonkeyPatch,
    paths: dict[str, Path],
    *,
    nvrtc_version: tuple[int, int],
    nvjitlink_version: tuple[int, int],
    actual_nvrtc: Path | None = None,
    actual_nvjitlink: Path | None = None,
    failures: dict[Path, str] | None = None,
) -> tuple[list[Path], list[str]]:
    attempts: list[Path] = []
    pathfinder_calls: list[str] = []
    failures = failures or {}

    def load_library(path, *, mode):
        assert mode == getattr(_toolkit.ctypes, "RTLD_GLOBAL", 0)
        candidate = Path(path)
        attempts.append(candidate)
        if candidate in failures:
            raise OSError(failures[candidate])
        if candidate == paths["nvrtc"]:
            return _FakeNvrtcLibrary(nvrtc_version)
        if candidate.name == paths["nvjitlink"].name:
            return _FakeNvJitLinkLibrary(nvjitlink_version)
        return object()

    def load_with_pathfinder(kind: str):
        pathfinder_calls.append(kind)
        return SimpleNamespace(
            abs_path=str(
                (actual_nvrtc or paths["nvrtc"])
                if kind == "nvrtc"
                else (actual_nvjitlink or paths["nvjitlink"])
            )
        )

    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        load_with_pathfinder,
    )
    return attempts, pathfinder_calls


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
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    kind: str,
    expected: tuple[str, ...],
) -> None:
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
    monkeypatch: pytest.MonkeyPatch,
    os_name: str,
    expected: tuple[str, ...],
) -> None:
    monkeypatch.setattr(_toolkit, "os", SimpleNamespace(name=os_name))

    assert _toolkit._nvrtc_builtins_names(13, 2) == expected


def test_preload_loads_one_same_root_monolithic_set_and_reuses_exact_handles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    attempts, pathfinder_calls = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(13, 2),
        nvjitlink_version=(13, 4),
    )

    first = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))
    second = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [paths["builtins"], paths["nvrtc"], paths["nvjitlink"]]
    assert pathfinder_calls == ["nvrtc", "nvJitLink", "nvrtc", "nvJitLink"]
    assert first == second
    assert first.toolkit_root == str(paths["root"].resolve())
    assert first.nvrtc_path == str(paths["nvrtc"])
    assert first.nvrtc_builtins_path == str(paths["builtins"])
    assert first.nvjitlink_path == str(paths["nvjitlink"])
    assert first.toolkit_version == (13, 2)
    assert first.nvrtc_version == (13, 2)
    assert first.nvjitlink_version == (13, 4)


def test_preload_loads_split_wheel_set_from_one_nvidia_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_split_wheel_toolkit(
        tmp_path / "site-packages" / "nvidia",
        12090,
    )
    attempts, pathfinder_calls = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(12, 9),
        nvjitlink_version=(12, 9),
    )

    libraries = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [paths["builtins"], paths["nvrtc"], paths["nvjitlink"]]
    assert pathfinder_calls == ["nvrtc", "nvJitLink"]
    assert libraries.toolkit_root == str(paths["root"].resolve())
    assert libraries.nvrtc_path == str(paths["nvrtc"])
    assert libraries.nvrtc_builtins_path == str(paths["builtins"])
    assert libraries.nvjitlink_path == str(paths["nvjitlink"])
    assert libraries.toolkit_version == (12, 9)


@pytest.mark.parametrize(
    ("os_name", "library_dir_name"),
    (("posix", "lib"), ("nt", "bin")),
)
def test_split_wheel_candidate_layout_matches_platform(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    os_name: str,
    library_dir_name: str,
) -> None:
    monkeypatch.setattr(_toolkit, "os", SimpleNamespace(name=os_name))
    paths = _write_split_wheel_toolkit(
        tmp_path / "site-packages" / "nvidia",
        12090,
        library_dir_name=library_dir_name,
    )

    candidates, diagnostic = _toolkit._toolkit_root_candidates(
        paths["include"],
        major=12,
        minor=9,
    )

    assert diagnostic == ""
    assert candidates == _toolkit._ToolkitRootCandidates(
        toolkit_root=paths["root"].resolve(),
        nvrtc_pairs=((paths["nvrtc"], paths["builtins"]),),
        nvjitlink=(paths["nvjitlink"],),
    )


def test_preload_rejects_split_wheel_components_from_later_nvidia_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _write_split_wheel_toolkit(
        tmp_path / "first" / "site-packages" / "nvidia",
        12090,
    )
    second = _write_split_wheel_toolkit(
        tmp_path / "second" / "site-packages" / "nvidia",
        12090,
    )
    first["nvjitlink"].unlink()
    attempts: list[Path] = []
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: attempts.append(Path(path)),
    )
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda kind: pytest.fail(f"unexpected fallback load for {kind}"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "require NVRTC, nvrtc-builtins, and nvJitLink from one CUDA Toolkit root"
        ),
    ) as exc_info:
        _toolkit.preload_toolkit_compiler_libraries(
            (first["include"], second["include"])
        )

    assert str(first["root"]) in str(exc_info.value)
    assert str(second["root"]) in str(exc_info.value)
    assert attempts == []


def test_preload_rejects_split_wheel_nvrtc_version_mismatch(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_split_wheel_toolkit(
        tmp_path / "site-packages" / "nvidia",
        12090,
    )
    _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(12, 8),
        nvjitlink_version=(12, 9),
    )

    with pytest.raises(
        RuntimeError,
        match=r"headers report Toolkit 12\.9, but loaded NVRTC .* reports 12\.8",
    ):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))


def test_preload_rejects_nvjitlink_from_different_toolkit_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first_root = tmp_path / "first"
    second_root = tmp_path / "second"
    first_include = first_root / "include"
    second_include = second_root / "include"
    first_lib = first_root / "lib"
    second_lib = second_root / "lib"
    _write_cuda_header(first_include, 13020)
    _write_cuda_header(second_include, 13020)
    first_lib.mkdir()
    second_lib.mkdir()
    _library_path(first_lib, "nvrtc", 13).touch()
    _builtins_path(first_lib, 13, 2).touch()
    _library_path(second_lib, "nvJitLink", 13).touch()
    attempts: list[Path] = []
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: attempts.append(Path(path)),
    )
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        lambda kind: pytest.fail(f"unexpected fallback load for {kind}"),
    )

    with pytest.raises(
        RuntimeError,
        match=(
            "require NVRTC, nvrtc-builtins, and nvJitLink from one CUDA Toolkit root"
        ),
    ) as exc_info:
        _toolkit.preload_toolkit_compiler_libraries((first_include, second_include))

    assert str(first_root) in str(exc_info.value)
    assert str(second_root) in str(exc_info.value)
    assert attempts == []


def test_preload_rejects_nvrtc_without_adjacent_builtins(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    root = tmp_path / "toolkit"
    include_dir = root / "include"
    lib_dir = root / "lib"
    _write_cuda_header(include_dir, 13020)
    lib_dir.mkdir()
    _library_path(lib_dir, "nvrtc", 13).touch()
    _library_path(lib_dir, "nvJitLink", 13).touch()
    monkeypatch.setattr(
        _toolkit.ctypes,
        "CDLL",
        lambda path, *, mode: pytest.fail(f"unexpected load of {path}"),
    )

    with pytest.raises(RuntimeError, match="without adjacent builtins"):
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))


def test_preload_stops_if_nvrtc_fails_after_builtins_becomes_global(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    attempts, pathfinder_calls = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(13, 2),
        nvjitlink_version=(13, 2),
        failures={paths["nvrtc"]: "NVRTC load failed"},
    )

    with pytest.raises(
        RuntimeError,
        match="failed loading same-root NVRTC after its builtins library",
    ):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [paths["builtins"], paths["nvrtc"]]
    assert pathfinder_calls == []
    assert _toolkit._PROCESS_TOOLKIT_SELECTION == (
        str(paths["root"].resolve()),
        (13, 2),
    )


def test_preload_tries_nvjitlink_candidates_only_within_selected_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    lib64 = paths["root"] / "lib64"
    lib64.mkdir()
    second_nvjitlink = _library_path(lib64, "nvJitLink", 13)
    second_nvjitlink.touch()
    attempts, _ = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(13, 2),
        nvjitlink_version=(13, 3),
        actual_nvjitlink=second_nvjitlink,
        failures={paths["nvjitlink"]: "first candidate unavailable"},
    )

    libraries = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [
        paths["builtins"],
        paths["nvrtc"],
        paths["nvjitlink"],
        second_nvjitlink,
    ]
    assert libraries.nvjitlink_path == str(second_nvjitlink)


@pytest.mark.parametrize("layout", ["monolithic", "split-wheel"])
@pytest.mark.parametrize("kind", ["nvrtc", "nvJitLink"])
def test_preload_rejects_pathfinder_library_from_another_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layout: str,
    kind: str,
) -> None:
    paths = (
        _write_complete_toolkit(tmp_path / "toolkit", 13020)
        if layout == "monolithic"
        else _write_split_wheel_toolkit(
            tmp_path / "site-packages" / "nvidia",
            12090,
        )
    )
    version = (13, 2) if layout == "monolithic" else (12, 9)
    mismatched = (
        tmp_path
        / "other"
        / (paths["nvrtc"].name if kind == "nvrtc" else paths["nvjitlink"].name)
    )
    _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=version,
        nvjitlink_version=version,
        actual_nvrtc=mismatched if kind == "nvrtc" else None,
        actual_nvjitlink=mismatched if kind == "nvJitLink" else None,
    )

    with pytest.raises(
        RuntimeError,
        match=rf"process uses .* for {kind}",
    ):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))


def test_preload_rejects_nvrtc_version_mismatched_with_headers(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(13, 1),
        nvjitlink_version=(13, 2),
    )

    with pytest.raises(
        RuntimeError,
        match=r"headers report Toolkit 13\.2, but loaded NVRTC .* reports 13\.1",
    ):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))


@pytest.mark.parametrize("actual_version", [(12, 8), (13, 1)])
def test_preload_rejects_incompatible_same_root_nvjitlink_version(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    actual_version: tuple[int, int],
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(13, 2),
        nvjitlink_version=actual_version,
    )

    with pytest.raises(
        RuntimeError,
        match=(
            r"headers report Toolkit 13\.2, but loaded nvJitLink .* reports "
            rf"{actual_version[0]}\.{actual_version[1]}"
        ),
    ):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))


def test_process_global_toolkit_selection_rejects_a_later_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    first = _write_complete_toolkit(tmp_path / "first", 13020)
    second = _write_complete_toolkit(tmp_path / "second", 13020)
    _patch_compiler_loaders(
        monkeypatch,
        first,
        nvrtc_version=(13, 2),
        nvjitlink_version=(13, 2),
    )
    _toolkit.preload_toolkit_compiler_libraries((first["include"],))

    with pytest.raises(RuntimeError, match="resolved headers select different"):
        _toolkit.preload_toolkit_compiler_libraries((second["include"],))


def test_toolkit_version_rejects_disagreeing_header_roots(tmp_path: Path) -> None:
    first = tmp_path / "first" / "include"
    second = tmp_path / "second" / "include"
    _write_cuda_header(first, 13020)
    _write_cuda_header(second, 12080)

    with pytest.raises(RuntimeError, match="disagree on Toolkit version"):
        _toolkit.preload_toolkit_compiler_libraries((first, second))


def test_toolkit_version_rejects_unparseable_selected_header(tmp_path: Path) -> None:
    include_dir = tmp_path / "toolkit" / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_text(
        "// no CUDART_VERSION\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="failed parsing CUDART_VERSION"):
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))


@pytest.mark.parametrize("layout", ["monolithic", "split-wheel"])
@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symbolic links")
def test_preload_rejects_library_symlink_escaping_toolkit_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    layout: str,
) -> None:
    paths = (
        _write_complete_toolkit(tmp_path / "toolkit", 13020)
        if layout == "monolithic"
        else _write_split_wheel_toolkit(
            tmp_path / "site-packages" / "nvidia",
            12090,
        )
    )
    version = (13, 2) if layout == "monolithic" else (12, 9)
    outside = tmp_path / "outside" / paths["nvjitlink"].name
    outside.parent.mkdir()
    outside.touch()
    paths["nvjitlink"].unlink()
    paths["nvjitlink"].symlink_to(outside)
    attempts, pathfinder_calls = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=version,
        nvjitlink_version=version,
    )

    with pytest.raises(RuntimeError, match="library escapes"):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [paths["builtins"], paths["nvrtc"]]
    assert pathfinder_calls == []
