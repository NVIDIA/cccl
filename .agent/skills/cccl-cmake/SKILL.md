---
description: |
  CCCL's CMake configuration system — presets, per-library enable flags, architecture
  values, and non-preset builds. Covers what presets exist, how to use them, which
  options to toggle for local dev, and where helper modules live.
  Triggers: "cmake presets", "configure cccl", "what presets are available",
  "non-preset build", "list cmake options".
---

# CMake

Reference and orientation for CCCL's CMake configuration layer. Push cmake module
internals, custom-command definitions, and arch-flag mechanics to `cccl_detail-cmake`.

## Presets

`CMakePresets.json` at the repo root. List all user-visible presets:

```
cmake --list-presets
```

Configure with a preset (Ninja generator, build dir set automatically):

```
cmake --preset <name>
```

Build dir lands at `build/$CCCL_BUILD_INFIX/<preset-name>/` relative to the source root.

Key presets:

| Preset                        | Purpose |
|-------------------------------|---------|
| `all-dev`                     | All libraries, tests, examples — native arch only. Start here for local dev. |
| `all-dev-debug`               | Same as `all-dev`, Debug build type, device-side debug (`-G`). |
| `all-tidy`                    | clang-tidy run, C++17, clang as host and CUDA compiler. |
| `libcudacxx`, `cub`, `thrust`, `cudax` | Single-library dev with tests. |
| `libcudacxx-cpp17/20`, `cub-cpp17/20` | Per-library with explicit C++ standard. |
| `install`, `install-unstable` | Packaging — only stable (or stable+experimental) libs. |

Each per-library preset also has `-cpp17` and `-cpp20` variants. CUB has additional
launcher-configuration variants (`cub-nolid`, `cub-lid0`, etc.).

## Key CMake options

Toggle these via `-D` on the command line or in a preset:

| Option                        | Default           | Effect                                               |
|-------------------------------|-------------------|------------------------------------------------------|
| `CCCL_ENABLE_LIBCUDACXX`      | OFF               | libcudacxx developer build                           |
| `CCCL_ENABLE_CUB`             | OFF               | CUB developer build                                  |
| `CCCL_ENABLE_THRUST`          | OFF               | Thrust developer build                               |
| `CCCL_ENABLE_CUDAX`           | OFF               | cudax developer build (requires `CCCL_ENABLE_UNSTABLE`) |
| `CCCL_ENABLE_UNSTABLE`        | OFF               | Gate for experimental/unstable targets               |
| `CCCL_ENABLE_TESTING`         | OFF               | Top-level test targets                               |
| `CCCL_ENABLE_EXAMPLES`        | OFF               | Example targets                                      |
| `CCCL_ENABLE_BENCHMARKS`      | OFF               | NVBench benchmark targets (not available with NVHPC) |
| `CCCL_ENABLE_C_PARALLEL`      | OFF               | C Parallel library                                   |
| `CCCL_ENABLE_CLANG_TIDY`      | OFF               | clang-tidy integration                               |
| `CMAKE_CUDA_ARCHITECTURES`   | `all-major-cccl` | GPU arch list (see below)                            |

## CMAKE_CUDA_ARCHITECTURES

CCCL defines two custom values beyond the standard CMake ones:

- `all-major-cccl` — all major architectures supported by the current CTK, filtered to ≥ sm_75. Default in presets.
- `all-cccl` — all architectures (including minor variants) ≥ sm_75.
- `native` — detect the GPU in the build machine. Used by `all-dev`.

For an explicit list: `-DCMAKE_CUDA_ARCHITECTURES="80-real;90-real;90-virtual"`.

## Non-preset build

Without presets, configure manually:

```
cmake -B build \
  -DCMAKE_CUDA_ARCHITECTURES=native \
  -DCCCL_ENABLE_CUB=ON \
  -DCCCL_ENABLE_TESTING=ON
```

Minimum CMake version for dev builds: 3.21. For embedding via `add_subdirectory`: 3.18.

## Helper modules

All helpers live in `cmake/`. Notable files:

| File                            | Role                                                 |
|---------------------------------|------------------------------------------------------|
| `CCCLCheckCudaArchitectures.cmake` | Resolves `all-cccl` / `all-major-cccl` to concrete arch lists |
| `CCCLDevBuildChecks.cmake`      | Validation checks for top-level dev builds           |
| `CCCLAddSubdirHelper.cmake`     | Support for `add_subdirectory()` embedding           |
| `CCCLInstallRules.cmake`        | Install / packaging rules                            |
| `CCCLGetDependencies.cmake`     | Dependency fetch via CPM                             |
| `CCCLGenerateHeaderTests.cmake` | Header include-test generation                       |
| `CCCLConfigureTarget.cmake`     | Target configuration helpers                         |
| `CCCLUtilities.cmake`           | Common utilities (included first)                     |

Internals of these modules — custom command definitions, `cccl_add_compile_test`, per-lib
`CMakeLists.txt` structure — are covered in `cccl_detail-cmake`.

## See also

- `cccl-build` — `ci/util/build_and_test_targets.sh`, the recommended driver for targeted
  local builds and tests. Prefer it over direct `cmake --build` for CI-like iteration.
- `cccl_detail-cmake` — module internals, `cccl_add_compile_test` and related custom commands,
  per-library `CMakeLists.txt` walkthrough, arch-flag deep mechanics.
