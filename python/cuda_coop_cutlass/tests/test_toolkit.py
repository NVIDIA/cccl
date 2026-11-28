# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Check CUDA Toolkit library selection with temporary files and fake loaders.

The selected libraries must come from the header installation and have
compatible versions. The fakes record load order and emulate version queries
so the tests can cover Linux and Windows without loading CUDA.
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from types import SimpleNamespace

import cuda.pathfinder
import pytest

from cuda.coop.cutlass._headers import _toolkit


class _VersionFunction:
    """Emulate a C version query that writes through two integer pointers.

    Assert the ctypes signature because NVRTC and nvJitLink use different
    integer types for their version outputs.
    """

    def __init__(self, version, integer_type):
        self.version = version
        self.pointer_type = _toolkit.ctypes.POINTER(integer_type)
        self.argtypes = None
        self.restype = None

    def __call__(self, major, minor):
        assert self.argtypes == (self.pointer_type, self.pointer_type)
        assert self.restype is _toolkit.ctypes.c_int
        _toolkit.ctypes.cast(major, self.pointer_type)[0] = self.version[0]
        _toolkit.ctypes.cast(minor, self.pointer_type)[0] = self.version[1]
        return 0


@pytest.fixture(autouse=True)
def _isolated_process_toolkit_state():
    """Reset fake library handles and toolkit selection around each test."""
    with _toolkit._PRELOAD_LOCK:
        _toolkit._windows_toolkit_candidates.cache_clear()
        _toolkit._EXACT_LIBRARY_HANDLES.clear()
        _toolkit._PROCESS_TOOLKIT_SELECTION = None
    yield
    with _toolkit._PRELOAD_LOCK:
        _toolkit._windows_toolkit_candidates.cache_clear()
        _toolkit._EXACT_LIBRARY_HANDLES.clear()
        _toolkit._PROCESS_TOOLKIT_SELECTION = None


def _write_cuda_header(include_dir: Path, encoded_version: int) -> None:
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_text(
        f"#define CUDART_VERSION {encoded_version}\n",
        encoding="utf-8",
    )


def _library_path(directory: Path, kind: str, major: int) -> Path:
    if _toolkit.os.name == "nt":
        name = (
            f"nvrtc64_{major}0_0.dll"
            if kind == "nvrtc"
            else f"nvJitLink_{major}0_0.dll"
        )
    else:
        name = f"lib{kind}.so.{major}"
    return directory / name


def _builtins_path(directory: Path, major: int, minor: int) -> Path:
    name = (
        f"nvrtc-builtins64_{major}{minor}.dll"
        if _toolkit.os.name == "nt"
        else f"libnvrtc-builtins.so.{major}.{minor}"
    )
    return directory / name


def _write_complete_toolkit(
    root: Path,
    encoded_version: int,
    *,
    library_dir_name: str = "lib",
) -> dict[str, Path]:
    """Create a version header and empty libraries under one toolkit root."""
    major = encoded_version // 1000
    minor = (encoded_version % 1000) // 10
    include_dir = root / "include"
    lib_dir = root / library_dir_name
    _write_cuda_header(include_dir, encoded_version)
    lib_dir.mkdir(parents=True)
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
    """Create sibling runtime, NVRTC, and nvJitLink wheel directories."""
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
    """Replace compiler loaders and return their call logs for assertions.

    The ctypes fake uses the requested versions and optional load failures.
    The pathfinder fake can report a different loaded path to test a mismatch
    with the selected toolkit. Neither fake opens a native library.
    """
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
            return SimpleNamespace(
                nvrtcVersion=_VersionFunction(
                    nvrtc_version, _toolkit.ctypes.c_int
                )
            )
        if candidate.name == paths["nvjitlink"].name:
            return SimpleNamespace(
                nvJitLinkVersion=_VersionFunction(
                    nvjitlink_version, _toolkit.ctypes.c_uint
                )
            )
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

    # A real Pathfinder probe cannot load the empty DLLs in these fixtures.
    def probe(toolkit_root, names, wheel_site=None):
        result = []
        for name in names:
            path = paths[name.lower()]
            base = paths["root"] if len(names) == 2 else path.parent.parent
            result.append(str(toolkit_root / path.relative_to(base)))
        return result

    monkeypatch.setattr(_toolkit, "_windows_compiler_paths", probe)
    monkeypatch.setattr(_toolkit.ctypes, "CDLL", load_library)
    monkeypatch.setattr(
        cuda.pathfinder,
        "load_nvidia_dynamic_lib",
        load_with_pathfinder,
    )
    return attempts, pathfinder_calls


@pytest.mark.parametrize(
    "os_name,library_dir_name",
    [
        ("posix", "lib"),
        ("nt", "bin"),
        ("nt", "pathfinder-selected/native-dlls"),
    ],
)
def test_preload_loads_one_same_root_monolithic_set_and_reuses_exact_handles(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    os_name: str,
    library_dir_name: str,
) -> None:
    monkeypatch.setattr(
        _toolkit, "os", SimpleNamespace(name=os_name, path=os.path)
    )
    paths = _write_complete_toolkit(
        tmp_path / "toolkit", 13020, library_dir_name=library_dir_name
    )
    attempts, _ = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(13, 2),
        nvjitlink_version=(13, 4),
    )

    first = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))
    second = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [paths["builtins"], paths["nvrtc"], paths["nvjitlink"]]
    assert first == second
    assert first.toolkit_version == (13, 2)
    assert first.nvjitlink_version == (13, 4)


@pytest.mark.parametrize(
    "failure,diagnostic",
    [
        ("foreign", "library escapes"),
        ("missing", "missing nvJitLink DLL"),
        ("no-path", "no absolute path for nvJitLink"),
        ("relative", "no absolute path for nvJitLink"),
        ("load-error", "test nvJitLink load failed"),
    ],
)
def test_windows_discovery_can_retry_without_loading_into_the_compiler_process(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    failure: str,
    diagnostic: str,
) -> None:
    monkeypatch.setattr(
        _toolkit,
        "os",
        SimpleNamespace(name="nt", path=os.path, environ=os.environ),
    )
    paths = _write_complete_toolkit(
        tmp_path / "toolkit", 13020, library_dir_name="native/compiler"
    )
    imports = tmp_path / "imports"
    package = imports / "cuda"
    package.mkdir(parents=True)
    (package / "__init__.py").touch()
    (package / "pathfinder.py").write_text(
        """import json
import os
import site
from pathlib import Path
from types import SimpleNamespace

def load_nvidia_dynamic_lib(name):
    assert site.PREFIXES == []
    assert 'CUDA_HOME' not in os.environ and 'CONDA_PREFIX' not in os.environ
    assert not os.environ.get('PATH')
    root = Path(os.environ["CUDA_PATH"])
    with (root / "calls.txt").open("a") as calls:
        print(os.getpid(), name, file=calls)
    path = json.loads((root / "paths.json").read_text())[name]
    if path == "load-error":
        raise OSError("test nvJitLink load failed")
    return SimpleNamespace(abs_path=path)
""",
        encoding="utf-8",
    )
    monkeypatch.syspath_prepend(str(imports))
    monkeypatch.setenv("CUDA_PATH", str(tmp_path / "other toolkit"))
    monkeypatch.setenv("CUDA_HOME", str(tmp_path / "other toolkit"))
    monkeypatch.setenv("CONDA_PREFIX", str(tmp_path / "other conda"))
    probe = _toolkit._windows_compiler_paths
    attempts, pathfinder_calls = _patch_compiler_loaders(
        monkeypatch, paths, nvrtc_version=(13, 2), nvjitlink_version=(13, 2)
    )
    monkeypatch.setattr(_toolkit, "_windows_compiler_paths", probe)

    foreign = tmp_path / paths["nvjitlink"].name
    foreign.touch()
    bad = {
        "foreign": str(foreign),
        "missing": str(paths["root"] / "missing.dll"),
        "no-path": None,
        "relative": paths["nvjitlink"].name,
        "load-error": "load-error",
    }
    state = paths["root"] / "paths.json"
    discovered = {"nvrtc": str(paths["nvrtc"]), "nvJitLink": bad[failure]}
    state.write_text(json.dumps(discovered))
    with pytest.raises(RuntimeError, match=diagnostic):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))
    assert not attempts and not pathfinder_calls
    assert _toolkit._PROCESS_TOOLKIT_SELECTION is None

    discovered["nvJitLink"] = str(paths["nvjitlink"])
    state.write_text(json.dumps(discovered))
    first = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))
    assert first == _toolkit.preload_toolkit_compiler_libraries(
        (paths["include"],)
    )
    assert attempts == [paths["builtins"], paths["nvrtc"], paths["nvjitlink"]]
    calls = (paths["root"] / "calls.txt").read_text().splitlines()
    assert [line.split()[1] for line in calls] == ["nvrtc", "nvJitLink"] * 2
    assert all(int(line.split()[0]) != os.getpid() for line in calls)
    assert os.environ["CUDA_PATH"] == str(tmp_path / "other toolkit")


@pytest.mark.parametrize(
    "os_name,library_dir_name", [("posix", "lib"), ("nt", "bin")]
)
def test_preload_loads_split_wheel_set_from_one_nvidia_anchor(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    os_name: str,
    library_dir_name: str,
) -> None:
    monkeypatch.setattr(
        _toolkit, "os", SimpleNamespace(name=os_name, path=os.path)
    )
    paths = _write_split_wheel_toolkit(
        tmp_path / "site-packages" / "nvidia",
        12090,
        library_dir_name=library_dir_name,
    )
    attempts, _ = _patch_compiler_loaders(
        monkeypatch,
        paths,
        nvrtc_version=(12, 9),
        nvjitlink_version=(12, 9),
    )

    libraries = _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert attempts == [paths["builtins"], paths["nvrtc"], paths["nvjitlink"]]
    assert libraries.toolkit_version == (12, 9)


def test_windows_scopes_pathfinder_to_the_selected_wheel_installation(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr(
        _toolkit, "os", SimpleNamespace(name="nt", path=os.path)
    )
    site = tmp_path / "target"
    paths = _write_complete_toolkit(site / "nvidia" / "cu13", 13040)
    calls = []

    def probe(root, names, wheel_site):
        calls.append((root, names, wheel_site))
        return [str(paths[name.lower()]) for name in names]

    monkeypatch.setattr(_toolkit, "_windows_compiler_paths", probe)
    _toolkit._windows_toolkit_candidates(paths["include"], 13, 4)
    assert calls == [(paths["root"], ("nvrtc", "nvJitLink"), site)]


@pytest.mark.parametrize("kind", ["nvrtc", "nvjitlink", "builtins"])
def test_windows_rejects_cuda13_component_beside_cuda12_wheels(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, kind: str
) -> None:
    monkeypatch.setattr(
        _toolkit, "os", SimpleNamespace(name="nt", path=os.path)
    )
    paths = _write_split_wheel_toolkit(
        tmp_path / "site-packages" / "nvidia", 12090
    )
    foreign = _write_complete_toolkit(paths["root"] / "cu13", 13020)
    probe = {"nvrtc": str(paths["nvrtc"]), "nvjitlink": str(paths["nvjitlink"])}
    if kind == "builtins":
        paths[kind].unlink()
        try:
            paths[kind].symlink_to(foreign[kind])
        except OSError:
            pytest.skip(
                "creating a file symlink requires Windows Developer Mode"
            )
    else:
        probe[kind] = str(foreign[kind])
    attempts, pathfinder_calls = _patch_compiler_loaders(
        monkeypatch, paths, nvrtc_version=(12, 9), nvjitlink_version=(12, 9)
    )
    monkeypatch.setattr(
        _toolkit,
        "_windows_compiler_paths",
        lambda root, names: [probe[name.lower()] for name in names],
    )
    with pytest.raises(RuntimeError, match="library escapes"):
        _toolkit.preload_toolkit_compiler_libraries((paths["include"],))

    assert not attempts and not pathfinder_calls
    assert _toolkit._PROCESS_TOOLKIT_SELECTION is None


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
    monkeypatch.setattr(
        _toolkit,
        "_windows_compiler_paths",
        lambda root, names: [
            str(first["nvrtc"] if names == ("nvrtc",) else second["nvjitlink"])
        ],
    )
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
            "require NVRTC, nvrtc-builtins, and nvJitLink "
            "from one CUDA Toolkit root"
        ),
    ):
        _toolkit.preload_toolkit_compiler_libraries(
            (first["include"], second["include"])
        )

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
        _toolkit,
        "_windows_compiler_paths",
        lambda root, names, wheel_site: tuple(
            str(_library_path(lib_dir, name, 13))
            for name in ("nvrtc", "nvJitLink")
        ),
    )
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
    # POSIX can try a second filename. Windows Pathfinder selects one DLL.
    monkeypatch.setattr(
        _toolkit, "os", SimpleNamespace(name="posix", path=os.path)
    )
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


@pytest.mark.parametrize("kind", ["nvrtc", "nvJitLink"])
def test_preload_rejects_pathfinder_library_from_another_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    kind: str,
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    version = (13, 2)
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
        match=(
            r"headers report Toolkit 13\.2, "
            r"but loaded NVRTC .* reports 13\.1"
        ),
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


def test_toolkit_version_rejects_disagreeing_header_roots(
    tmp_path: Path,
) -> None:
    first = tmp_path / "first" / "include"
    second = tmp_path / "second" / "include"
    _write_cuda_header(first, 13020)
    _write_cuda_header(second, 12080)

    with pytest.raises(RuntimeError, match="disagree on Toolkit version"):
        _toolkit.preload_toolkit_compiler_libraries((first, second))


def test_toolkit_version_rejects_unparseable_selected_header(
    tmp_path: Path,
) -> None:
    include_dir = tmp_path / "toolkit" / "include"
    include_dir.mkdir(parents=True)
    (include_dir / "cuda_runtime_api.h").write_text(
        "// no CUDART_VERSION\n",
        encoding="utf-8",
    )

    with pytest.raises(RuntimeError, match="failed parsing CUDART_VERSION"):
        _toolkit.preload_toolkit_compiler_libraries((include_dir,))


@pytest.mark.skipif(os.name == "nt", reason="requires POSIX symbolic links")
def test_preload_rejects_library_symlink_escaping_toolkit_root(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    paths = _write_complete_toolkit(tmp_path / "toolkit", 13020)
    version = (13, 2)
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
