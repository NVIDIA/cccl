---
description: |
  CCCL CMake internals — helper modules, custom commands, architecture-flag translation,
  metatarget dot-path system, compiler interface flags, and downstream-consumer target
  surface. Deep reference for questions about cccl_add_executable, all-major-cccl
  expansion, CCCL::CCCL linking, CCCLConfig, or CPM dependency wiring.
  Triggers: "cmake internals", "cccl_add_executable", "all-major-cccl expansion",
  "CCCL::CCCL target", "downstream consumer", "cmake helpers", "metatarget dot-path".
---

Deep-internals reference for CCCL's CMake layer. Covers every module under `cmake/`,
the custom commands authors call in tests/examples, architecture-flag expansion, the
metatarget dot-path naming system, and the exported target surface for downstream consumers.

## Helper module index

Full table in `references/cmake-module-index.md`.

Quick map:

| Module                     | Role                                                                                              |
|---------------------------|---------------------------------------------------------------------------------------------------|
| `AppendOptionIfAvailable`  | Probe-and-append a compiler flag via `check_cxx_compiler_flag`                                   |
| `CCCLAddExecutable`        | `cccl_add_executable` — build executable with standard CCCL setup                                |
| `CCCLAddSubdir` / `CCCLAddSubdirHelper` | Include a sub-library via its package config without `find_package` re-entry                      |
| `CCCLBuildCompilerTargets` | Define `cccl.compiler_interface` INTERFACE target; accumulate all warning/error flags              |
| `CCCLCheckCudaArchitectures` | Expand `all-major-cccl` / `all-cccl` pseudo-values in `CMAKE_CUDA_ARCHITECTURES`                  |
| `CCCLClangdCompileInfo`    | Enable `compile_commands.json` and symlink it to the source root                                 |
| `CCCLConfigureTarget`      | Apply standard properties (CXX/CUDA standard, dialect features, output dirs) to any target       |
| `CCCLDevBuildChecks`       | Enforce CXX == CUDA standard; default to C++17 when neither is set                               |
| `CCCLEnsureMetaTargets`    | Create dot-path custom targets (`cub`, `cub.test`, …) as umbrella build targets                  |
| `CCCLGenerateHeaderTests`  | Generate per-header `OBJECT` libraries + link-check executables for include hygiene              |
| `CCCLGetDependencies`      | CPM macros: `cccl_get_catch2`, `cccl_get_boost`, `cccl_get_nvbench`, etc.                        |
| `CCCLHideThirdPartyOptions` | `mark_as_advanced` for CPM / Catch2 / LLVM noise variables                                       |
| `CCCLInstallRules`         | `cccl_generate_install_rules` — header globs + CMake package install per sub-library             |
| `CCCLTestParams`           | `%PARAM%`-comment scanner — cartesian-product test variant expansion                             |
| `CCCLUtilities`            | `cccl_add_compile_test`, `cccl_add_xfail_compile_target_test`, `cccl_execute_non_fatal_process` |
| `CCCLAddTidyTarget`        | `cccl_tidy_init` / `cccl_tidy_add_target` — per-source `clang-tidy` custom targets                |

## Custom commands

Full signatures in `references/custom-commands.md`.

Core commands authors encounter:

- `cccl_add_executable(target SOURCES … [ADD_CTEST] [NO_METATARGETS] [NO_CLANG_TIDY] [DIALECT N] [METATARGET_PATH path])` — creates an executable, calls `cccl_configure_target`, optionally registers a CTest, and hooks into the metatarget hierarchy.
- `cccl_configure_target(target [DIALECT N])` — enforces standard, extension-off, and output directory properties. Called by all target-creating commands.
- `cccl_add_compile_test(result_var name_prefix subdir test_id [CTEST_COMMAND …] cmake_opts…)` — registers a CTest that does a full configure-build-test of an out-of-tree CMake project (used in `examples/`).
- `cccl_add_xfail_compile_target_test(target [ERROR_REGEX …] [SOURCE_FILE …] [ERROR_REGEX_LABEL …])` — wraps an expected-failure compile as a CTest; regex extracted from source comment annotations.
- `cccl_generate_header_tests(target include_path [GLOBS …] [EXCLUDES …] [PER_HEADER_DEFINES …])` — generates one `.cu`/`.cpp` file per header and a link-check executable that catches non-inline function definitions.
- `cccl_parse_variant_params(src …)` / `cccl_get_variant_data(…)` — `%PARAM%`-comment parser for test variant matrices.
- `cccl_generate_install_rules(PROJECT_NAME enable [NO_HEADERS] [HEADERS_SUBDIRS …] [PACKAGE])` — produces `install()` rules for headers and the CMake package.

## Architecture-flag translation

See `references/arch-flags.md` for the full expansion table and internal call chain.

`cmake/CCCLCheckCudaArchitectures.cmake` intercepts `CMAKE_CUDA_ARCHITECTURES` at configure time. Recognized pseudo-values:

| Value             | Expansion rule                                                        |
|-------------------|-----------------------------------------------------------------------|
| `all-cccl`        | All arches nvcc reports via `--help` ≥ `minimum_cccl_arch` (currently 75) |
| `all-major-cccl`  | Major-only subset of `all-cccl` (one per SM generation)               |

Both values tag every arch `-real` and the last arch additionally `-virtual`. Standard CMake values (`native`, `all`, `all-major`, numeric, `XX-real`, `XX-virtual`) pass through unchanged.

`minimum_cccl_arch` tracks the minimum architecture supported by the latest CTK; currently 75 (Turing) after CTK 13.x dropped pre-Turing.

## Metatarget dot-path system

`cccl_ensure_metatargets` splits a target name (or `METATARGET_PATH`) on `.` and creates a chain of `add_custom_target` umbrellas: `cudax` → `cudax.test` → `cudax.test.mytest`. This lets `ninja cudax` rebuild every cudax target and `ninja cudax.test` rebuild only tests.

## Downstream-consumer surface

See `references/downstream-consumers.md` for a worked `CMakeLists.txt` example.

`lib/cmake/cccl/cccl-config.cmake` is the entry point. `find_package(CCCL CONFIG)` transitively finds and links libcudacxx, CUB, and Thrust. `cudax` is included only when `CCCL_ENABLE_UNSTABLE` is set.

Exported targets:

| Target           | Provides                                                                   |
|------------------|---------------------------------------------------------------------------|
| `CCCL::CCCL`     | All components as a single link target                                    |
| `CCCL::libcudacxx` | Alias to `_libcudacxx_libcudacxx`                                          |
| `CCCL::CUB`      | IMPORTED INTERFACE wrapping `CUB::CUB` (not alias — supports downstream export sets) |
| `CCCL::Thrust`   | Created via `thrust_create_target` with host=CPP device=CUDA defaults     |
| `CCCL::cudax`    | Alias to `cudax::cudax` (or `_cudax_cudax` in internal builds)             |

Each sub-library also exports its own target (`libcudacxx::libcudacxx`, `CUB::CUB`, `Thrust::Thrust`, `cudax::cudax`) via per-library config files under `lib/cmake/<name>/`.

## Compiler interface and build options

`CCCLBuildCompilerTargets.cmake` defines `cccl.compiler_interface`, an INTERFACE target that accumulates warning flags (`-Wall`, `-Wextra`, Clang-only set, MSVC `/W4`) and definitions (`CCCL_DISABLE_EXCEPTIONS`, `CCCL_DISABLE_RTTI`, `_CCCL_NO_SYSTEM_HEADER`). Flags are applied conditionally via `$<COMPILE_LANG_AND_ID:…>` generators.

MSVC workaround: `CMAKE_MSVC_DEBUG_INFORMATION_FORMAT` is forced to `Embedded` so sccache can handle PDB generation without the `-Fd` flag conflict.

Developer build guard: `CMAKE_CUDA_HOST_COMPILER` must match `CMAKE_CXX_COMPILER` (enforced since CMake 3.31 via `CMAKE_CUDA_HOST_COMPILER_ID`/`VERSION`).

## CPM and third-party dependency wiring

`CCCLGetDependencies.cmake` provides `cccl_get_*` macros that gate CPM inclusion behind a guard so the `CPM.cmake` file is included only once. Key packages: Catch2 3.12.0, Boost 1.83.0, NVBench (SHA-pinned via `CCCL_NVBENCH_SHA` cache var), NVTX v3, dlpack v1.2.

Sub-library packages (CUB, Thrust, libcudacxx, cudax) are fetched via `find_package(… NO_DEFAULT_PATH HINTS …)` pointing at `CCCL_SOURCE_DIR/lib/cmake/<name>/` — no CPM involved for first-party deps.

## Additional resources

- `references/cmake-module-index.md` — full module inventory with file paths
- `references/custom-commands.md` — complete command signatures and parameter tables
- `references/arch-flags.md` — full arch-expansion table, internal call chain, minimum_cccl_arch history
- `references/downstream-consumers.md` — exported targets table and worked consumer example
