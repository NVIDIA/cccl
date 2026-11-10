# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

from __future__ import annotations

import os
import platform
import re
import shutil
import subprocess
import textwrap
from dataclasses import dataclass
from pathlib import Path

import pytest

from cuda.coop.tile import (
    CppTileConfig,
    get_cpp_include_path,
    get_cpp_tile_library_path,
)

from ....support.paths import PACKAGE_ROOT

CCCL_ROOT = PACKAGE_ROOT.parents[1]
TILE_ROOT = CCCL_ROOT / "tile/cpp"
CPP_HEADER = CCCL_ROOT / "tile/cpp/include/cuda/coop/tile.cuh"
FRONTEND_SOURCE_ROOT = CCCL_ROOT / "tile/cpp/test/support"

CPP_API_FRONTEND_SOURCE = FRONTEND_SOURCE_ROOT / "tile_cpp_api_sm120.cu"
CPP_API_SHAPES_SOURCE = FRONTEND_SOURCE_ROOT / "tile_cpp_api_shapes.cpp"
ADJACENT_SHUFFLE_FRONTEND_SOURCE = (
    FRONTEND_SOURCE_ROOT / "tile_cpp_adjacent_shuffle_sm120.cu"
)
RADIX_RANK_FRONTEND_SOURCE = (
    FRONTEND_SOURCE_ROOT / "tile_cpp_radix_rank_sm120.cu"
)
RADIX_SORT_FRONTEND_SOURCE = (
    FRONTEND_SOURCE_ROOT / "tile_cpp_radix_sort_sm120.cu"
)
RUN_LENGTH_DECODE_FRONTEND_SOURCE = (
    FRONTEND_SOURCE_ROOT / "tile_cpp_run_length_decode_sm120.cu"
)
PROVIDER_DIAGNOSTIC_FRONTEND_SOURCE = (
    FRONTEND_SOURCE_ROOT / "tile_cpp_provider_diagnostic_sm120.cu"
)

_REQUIRED_ENV = (
    "CUDA_COOP_TILE_CPP_TEST_NVCC",
    "CUDA_COOP_TILE_CPP_TEST_TILEIRAS",
    "CUDA_COOP_TILE_CPP_TEST_CUDA_PATH",
    "CUDA_COOP_TILE_CPP_TEST_ARCH",
)

_CANONICAL_SM_ARCH = re.compile(r"sm_[1-9][0-9]*[af]?")
_TILEIRAS_EXECUTABLE_NAMES = frozenset(("tileiras", "tileiras.exe"))


@dataclass(frozen=True)
class FrontendResult:
    compile_trace: str
    stdout: str
    requested_arch: str


@dataclass(frozen=True)
class _RealFrontendConfig:
    nvcc: Path
    tileiras: Path
    cuda_path: Path
    arch: str
    config: CppTileConfig
    environment: dict[str, str]


def _tileiras_lookup_name(tileiras: Path, *, windows: bool) -> str:
    executable_names = (
        _TILEIRAS_EXECUTABLE_NAMES if windows else frozenset(("tileiras",))
    )
    lookup_name = tileiras.name.casefold() if windows else tileiras.name
    if not tileiras.is_file() or lookup_name not in executable_names:
        pytest.fail(
            "CUDA_COOP_TILE_CPP_TEST_TILEIRAS must name a tileiras executable: "
            f"{tileiras}"
        )
    return tileiras.name


def _frontend_path_entries(
    tileiras: Path,
    cuda_path: Path,
    *,
    windows: bool,
) -> tuple[Path, ...]:
    entries = [tileiras.parent]
    if windows:
        dll_arch = (
            "arm64"
            if platform.machine().lower() in ("arm64", "aarch64")
            else "x86_64"
        )
        versioned_dll_directory = cuda_path / "bin" / dll_arch
        if not versioned_dll_directory.is_dir():
            pytest.fail(
                "Windows C++ Tile tests require the installed CUDA component "
                f"DLL directory: {versioned_dll_directory}"
            )
        entries.append(versioned_dll_directory)
    return tuple(entries)


def _real_frontend_config(description: str) -> _RealFrontendConfig:
    missing = [name for name in _REQUIRED_ENV if not os.environ.get(name)]
    if missing:
        pytest.fail(
            f"real C++ Tile {description} test requires " + ", ".join(missing)
        )

    nvcc = (
        Path(os.environ["CUDA_COOP_TILE_CPP_TEST_NVCC"]).expanduser().resolve()
    )
    tileiras = (
        Path(os.environ["CUDA_COOP_TILE_CPP_TEST_TILEIRAS"])
        .expanduser()
        .resolve()
    )
    tile_library = get_cpp_tile_library_path().resolve()
    cuda_path = (
        Path(os.environ["CUDA_COOP_TILE_CPP_TEST_CUDA_PATH"])
        .expanduser()
        .resolve()
    )
    if not nvcc.is_file():
        pytest.fail(f"CUDA_COOP_TILE_CPP_TEST_NVCC is not a file: {nvcc}")
    windows = os.name == "nt"
    tileiras_lookup_name = _tileiras_lookup_name(tileiras, windows=windows)
    if not tile_library.is_file():
        pytest.fail(f"C++ Tile library is not a file: {tile_library}")
    if not cuda_path.is_dir():
        pytest.fail(
            f"CUDA_COOP_TILE_CPP_TEST_CUDA_PATH is not a directory: {cuda_path}"
        )

    arch = _validate_test_arch(os.environ["CUDA_COOP_TILE_CPP_TEST_ARCH"])
    config = CppTileConfig(
        include_path=get_cpp_include_path(),
        tile_library_path=tile_library,
        support_include_paths=(
            CCCL_ROOT / "thrust",
            CCCL_ROOT / "cub",
            CCCL_ROOT / "cudax" / "include",
            CCCL_ROOT / "libcudacxx" / "include",
        ),
    )
    environment = os.environ.copy()
    environment["CUDA_COOP_CCCL_ROOT"] = str(CCCL_ROOT)
    environment.pop("CUDA_COOP_TILE_PROVIDER_INCLUDE_PATHS", None)
    environment.pop("CUDA_COOP_TILE_INCLUDE_PATHS", None)
    # This exact-stack lane must prove that NVCC/tileiras supplies the
    # authoritative compilation context. Provider fallback behavior is tested
    # natively; retaining either fallback here would mask an empty-context
    # compiler regression.
    environment.pop("CUDA_COOP_TILE_PROVIDER_SM_ARCH", None)
    environment.pop("CUDA_ARCH", None)
    environment["CUDA_COOP_TILE_PROVIDER_TRACE"] = "1"
    # Keep NVCC, tileiras, and their runtime libraries on one packaged root
    # through the standard toolkit aliases inherited by both compiler stages.
    for name in ("CUDA_TOOLKIT_PATH", "CUDA_HOME", "CUDA_ROOT", "CUDA_PATH"):
        environment[name] = str(cuda_path)
    environment["PATH"] = os.pathsep.join(
        (
            *(
                str(path)
                for path in _frontend_path_entries(
                    tileiras,
                    cuda_path,
                    windows=windows,
                )
            ),
            environment.get("PATH", ""),
        )
    ).rstrip(os.pathsep)
    resolved_tileiras = shutil.which(
        tileiras_lookup_name,
        path=environment["PATH"],
    )
    if (
        resolved_tileiras is None
        or Path(resolved_tileiras).resolve() != tileiras
    ):
        pytest.fail(
            "NVCC tileiras lookup does not resolve to "
            f"CUDA_COOP_TILE_CPP_TEST_TILEIRAS: {resolved_tileiras}"
        )
    runtime_library_paths = (
        cuda_path / "bin" / "nvrtc" / "lib64",
        cuda_path / "bin" / "nvJitLink" / "lib64",
        cuda_path / "lib64",
        cuda_path / "nvvm" / "lib64",
    )
    environment["LD_LIBRARY_PATH"] = os.pathsep.join(
        (
            *(str(path) for path in runtime_library_paths if path.is_dir()),
            environment.get("LD_LIBRARY_PATH", ""),
        )
    ).rstrip(os.pathsep)

    return _RealFrontendConfig(
        nvcc=nvcc,
        tileiras=tileiras,
        cuda_path=cuda_path,
        arch=arch,
        config=config,
        environment=environment,
    )


def _validate_test_arch(arch: str) -> str:
    if _CANONICAL_SM_ARCH.fullmatch(arch) is None:
        pytest.fail(
            "CUDA_COOP_TILE_CPP_TEST_ARCH must use canonical sm_N, sm_Na, or "
            f"sm_Nf spelling: {arch!r}"
        )
    return arch


def make_cpp_frontend_stubs(include: Path) -> None:
    """Write the minimal CUDA Tile surface needed for host syntax checks."""
    (include / "cuda" / "std").mkdir(parents=True)
    (include / "cuda" / "functional").write_text(
        textwrap.dedent(
            """\
            #pragma once
            namespace cuda {
            template <typename T = void> struct minimum {};
            template <typename T = void> struct maximum {};
            namespace std {
            template <typename T = void> struct plus {};
            template <typename T = void> struct multiplies {};
            template <typename T = void> struct less {};
            template <typename T = void> struct greater {};
            template <typename T = void> struct bit_and {};
            template <typename T = void> struct bit_or {};
            template <typename T = void> struct bit_xor {};
            }
            }
            """
        ),
        encoding="utf-8",
    )
    (include / "cuda" / "std" / "functional").write_text(
        "#pragma once\n#include <cuda/functional>\n",
        encoding="utf-8",
    )
    (include / "cuda" / "std" / "type_traits").write_text(
        textwrap.dedent(
            """\
            #pragma once
            #include <type_traits>
            namespace cuda::std {
            using ::std::is_convertible_v;
            using ::std::is_integral_v;
            using ::std::is_same_v;
            using ::std::is_void_v;
            using ::std::type_identity_t;
            }
            """
        ),
        encoding="utf-8",
    )
    (include / "cuda" / "std" / "utility").write_text(
        "#pragma once\n#include <utility>\nnamespace cuda::std {\n"
        "using ::std::index_sequence;\nusing ::std::make_index_sequence;\n}\n",
        encoding="utf-8",
    )
    (include / "cuda_bf16.h").write_text(
        "#pragma once\nstruct __nv_bfloat16 {};\n",
        encoding="utf-8",
    )
    (include / "cuda_fp16.h").write_text(
        "#pragma once\nstruct __half {};\n",
        encoding="utf-8",
    )
    (include / "cuda_tile.h").write_text(
        textwrap.dedent(
            """\
            #pragma once
            #include <cstddef>
            #include <array>
            #include <utility>
            #define __CUDACC_TILE__ 1
            #define __CUDACC_TILE_PREVIEW_SIMT__ 1
            #define __tile__
            #define _CCCL_VISIBILITY_HIDDEN
            #define _CCCL_EXCLUDE_FROM_EXPLICIT_INSTANTIATION
            #define __experimental_foreign_decl__
            #define __experimental_foreign_name__(name)
            #define _CCCL_TILE_API
            #define _CCCL_CONSTEVAL consteval
            namespace cuda::tiles {
            template <::std::size_t... Extents> struct shape {
              static constexpr ::std::size_t rank() {
                return sizeof...(Extents);
              }
              static constexpr ::std::size_t size() {
                return (Extents * ... * 1);
              }
              static constexpr ::std::size_t static_extent(::std::size_t i) {
                return ::std::array<::std::size_t, sizeof...(Extents)>{
                  Extents...}[i];
              }
            };
            template <typename T, typename Shape> struct tile {
              using element_type = T;
              using shape_type = Shape;
              tile() = default;
              tile(T) noexcept {}
              explicit operator T() const noexcept { return T{}; }
            };
            template <typename T> tile(T) -> tile<T, shape<>>;
            template <typename Tile>
            inline constexpr ::std::size_t tile_size_v =
              Tile::shape_type::size();
            template <::std::size_t... Axes> struct dimension_map {};
            template <typename T, typename S, ::std::size_t... Axes>
            auto permute(tile<T, S>, dimension_map<Axes...>) noexcept
              -> tile<T, shape<S::static_extent(Axes)...>>;
            template <typename T, typename S, typename ResultShape>
            auto reshape(tile<T, S>, ResultShape) noexcept
              -> tile<T, ResultShape>;
            template <typename T, typename S, typename ResultShape>
            auto broadcast(tile<T, S>, ResultShape) noexcept
              -> tile<T, ResultShape>;
            template <typename Tile, ::std::size_t Axis, ::std::size_t... I>
            auto reduction_type(::std::index_sequence<I...>)
              -> tile<typename Tile::element_type,
                      shape<(I == Axis ? 1 :
                             Tile::shape_type::static_extent(I))...>>;
            template <typename Tile, ::std::size_t Axis>
              requires (Axis < Tile::shape_type::rank())
            using reduction_result_t = decltype(reduction_type<Tile, Axis>(
              ::std::make_index_sequence<Tile::shape_type::rank()>{}));
            namespace experimental {
            template <::std::size_t N> struct fixedString {
              char data[N]{};
              consteval fixedString(const char (&value)[N]) {
                for (::std::size_t i = 0; i < N; ++i) {
                  data[i] = value[i];
                }
              }
            };
            template <typename... Outputs> struct multi_outputs;
            template <typename T0, typename T1>
            struct multi_outputs<T0, T1> { T0 F0; T1 F1; };
            }
            }
            """
        ),
        encoding="utf-8",
    )


def assert_cpp_api_rejects(
    tmp_path: Path,
    *,
    expression: str,
    diagnostic: str | tuple[str, ...],
) -> None:
    """Compile one invalid API expression and check its public diagnostic."""
    compiler = shutil.which("c++")
    if compiler is None:
        pytest.skip("a C++20 compiler is required for C++ Tile API diagnostics")

    include = tmp_path / "include"
    make_cpp_frontend_stubs(include)
    source = tmp_path / "cpp_api_invalid.cpp"
    source.write_text(
        textwrap.dedent(
            f"""\
            #include <cuda/coop/tile.cuh>

            namespace coop = cuda::coop::tile;
            namespace ct = cuda::tiles;

            void reject(
              ct::tile<int, ct::shape<64>> ints,
              ct::tile<long long, ct::shape<32>> short_values,
              ct::tile<float, ct::shape<64>> floats,
              ct::tile<bool, ct::shape<64>> bools,
              ct::tile<unsigned char, ct::shape<256>> bytes,
              ct::tile<unsigned int, ct::shape<64>> run_lengths,
              ct::tile<unsigned int, ct::shape<32>> short_run_lengths,
              ct::tile<int, ct::shape<1>> scalar_control,
              ct::tile<int, ct::shape<2>> wide_control,
              ct::tile<unsigned int, ct::shape<2>> wide_unsigned_control) {{
              {expression}
            }}
            """
        ),
        encoding="utf-8",
    )

    result = subprocess.run(
        [
            compiler,
            "-std=c++20",
            "-fsyntax-only",
            f"-I{include}",
            f"-I{TILE_ROOT / 'include'}",
            str(source),
        ],
        check=False,
        capture_output=True,
        text=True,
    )

    assert result.returncode != 0
    diagnostics = (diagnostic,) if isinstance(diagnostic, str) else diagnostic
    assert any(message in result.stderr for message in diagnostics), (
        result.stderr
    )


def compile_and_run_frontend(
    tmp_path: Path,
    *,
    source: Path,
    executable_name: str,
    description: str,
    run_args: tuple[str, ...] = (),
    run_timeout: float = 60,
) -> FrontendResult:
    """Compile and execute a C++ Tile route in a scoped subprocess env."""
    frontend = _real_frontend_config(description)
    executable = tmp_path / executable_name
    command = [
        str(frontend.nvcc),
        *frontend.config.nvcc_options(),
        f"-arch={frontend.arch}",
        str(source),
        "-o",
        str(executable),
    ]

    compiled = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=frontend.environment,
        timeout=300,
    )
    if compiled.returncode:
        pytest.fail(
            f"C++ Tile {description} compilation failed\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{compiled.stdout}\n"
            f"stderr:\n{compiled.stderr}"
        )

    run_command = [str(executable), *run_args]
    try:
        executed = subprocess.run(
            run_command,
            check=False,
            capture_output=True,
            text=True,
            env=frontend.environment,
            timeout=run_timeout,
        )
    except subprocess.TimeoutExpired as error:
        stdout = (
            error.stdout.decode(errors="replace")
            if isinstance(error.stdout, bytes)
            else (error.stdout or "")
        )
        stderr = (
            error.stderr.decode(errors="replace")
            if isinstance(error.stderr, bytes)
            else (error.stderr or "")
        )
        pytest.fail(
            f"C++ Tile {description} runtime exceeded {run_timeout} seconds\n"
            f"command: {' '.join(run_command)}\n"
            f"stdout:\n{stdout}\n"
            f"stderr:\n{stderr}"
        )
    if executed.returncode:
        pytest.fail(
            f"C++ Tile {description} runtime failed\n"
            f"command: {' '.join(run_command)}\n"
            f"stdout:\n{executed.stdout}\n"
            f"stderr:\n{executed.stderr}"
        )

    return FrontendResult(
        compile_trace=f"{compiled.stdout}\n{compiled.stderr}",
        stdout=executed.stdout,
        requested_arch=frontend.arch,
    )


def assert_provider_trace_uses_requested_architecture(
    result: FrontendResult,
) -> None:
    """Require implementation callbacks to use the requested target family."""
    trace_lines = result.compile_trace.splitlines()
    resource_prefix = "[coop_linkage_provider][trace] resource_query_"
    implementation_trace = "\n".join(
        line for line in trace_lines if resource_prefix not in line
    )
    observed = set(re.findall(r"\bsm=(sm_[0-9]+[af]?)\b", implementation_trace))
    requested = _CANONICAL_SM_ARCH.fullmatch(result.requested_arch)
    assert requested is not None
    requested_code = result.requested_arch[3:].rstrip("af")
    requested_family = (
        result.requested_arch[-1] if result.requested_arch[-1] in "af" else ""
    )
    refinements = observed - {result.requested_arch}
    refinement_is_valid = (
        not requested_family
        and bool(observed)
        and len(refinements) <= 1
        and all(
            re.fullmatch(rf"sm_{requested_code}[af]", architecture)
            for architecture in refinements
        )
    )
    assert observed == {result.requested_arch} or refinement_is_valid, (
        "provider trace target mismatch: "
        f"requested {result.requested_arch}, observed {sorted(observed)}"
    )

    resource_observed = set(
        re.findall(
            r"\bsm=(sm_[0-9]+[af]?)\b",
            "\n".join(line for line in trace_lines if resource_prefix in line),
        )
    )
    allowed_resource_architectures = observed | {f"sm_{requested_code}"}
    assert resource_observed <= allowed_resource_architectures, (
        "provider resource-query target mismatch: "
        f"requested {result.requested_arch}, "
        f"observed {sorted(resource_observed)}"
    )


def assert_provider_trace_scored_selected_implementations(
    result: FrontendResult,
) -> None:
    """Match instantiated specializations to scores published at discovery."""
    discovered = {}
    selected = set()
    for line in result.compile_trace.splitlines():
        fields = dict(
            field.split("=", 1) for field in line.split() if "=" in field
        )
        if "[coop_linkage_provider][trace] discovery candidate=" in line:
            discovered[fields["semantic_key"]] = int(fields["provider_score"])
        elif "[coop_linkage_provider][trace] resource_estimate " in line:
            selected.add(fields["semantic_key"])
    assert selected, "the compiler did not instantiate any Coop implementations"
    assert selected <= discovered.keys(), (
        "the compiler instantiated Coop implementations "
        "without scored discovery "
        f"rows: {sorted(selected - discovered.keys())}"
    )
    assert all(discovered[identity] > 100 for identity in selected), (
        "selected Coop implementations lack their discovery preference score"
    )


def compile_frontend_expect_failure(
    tmp_path: Path,
    *,
    source: Path,
    object_name: str,
    description: str,
) -> str:
    """Compile one invalid real C++ Tile route and return its diagnostics."""
    frontend = _real_frontend_config(description)
    environment = frontend.environment.copy()
    # The assertions below must observe compiler-propagated structured errors,
    # never the provider's supplementary stderr/file trace channel.
    environment["CUDA_COOP_TILE_PROVIDER_TRACE"] = "0"
    environment.pop("CUDA_COOP_TILE_PROVIDER_TRACE_FILE", None)
    output = tmp_path / object_name
    command = [
        str(frontend.nvcc),
        *frontend.config.nvcc_options(),
        f"-arch={frontend.arch}",
        "-c",
        str(source),
        "-o",
        str(output),
    ]

    compiled = subprocess.run(
        command,
        check=False,
        capture_output=True,
        text=True,
        env=environment,
        timeout=300,
    )
    if compiled.returncode == 0:
        pytest.fail(
            f"C++ Tile {description} compilation unexpectedly succeeded\n"
            f"command: {' '.join(command)}\n"
            f"stdout:\n{compiled.stdout}\n"
            f"stderr:\n{compiled.stderr}"
        )
    return f"{compiled.stdout}\n{compiled.stderr}"
