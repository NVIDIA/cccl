# `ci/build_common.sh` usage

Shared build configuration library sourced by all `ci/build_*.sh` scripts. Not invoked directly.
Parses the common option set, validates compilers, sets up environment variables, defines helper
functions (`configure_preset`, `build_preset`, `test_preset`, `configure_and_build_preset`,
`print_environment_details`, `run_ci_timed_command`), and establishes the build directory layout.

## Location

`ci/build_common.sh`. Must be **sourced** (`source ci/build_common.sh`), not executed. Each
per-project `ci/build_*.sh` script sources this after extracting its own project-specific flags.

## Interface

```
Usage: <script> [OPTIONS]

The PARALLEL_LEVEL environment variable controls the amount of build parallelism.
Default is the number of cores minus one.

Options:
  -v/-verbose:        enable shell echo for debugging
  -configure:         Only run cmake to configure, do not build or test.
  -cuda:              CUDA compiler (Defaults to $CUDACXX if set, otherwise nvcc)
  -cxx:               Host compiler (Defaults to $CXX if set, otherwise g++)
  -std:               CUDA/C++ standard (Defaults to 17)
  -arch:              Target CUDA arches, e.g. "60-real;70;80-virtual" (Defaults to value in presets file)
  -pedantic/--pedantic: Enable strict warnings-as-errors and expose CCCL header warnings (default in CI)
  -cmake-options:     Additional options to pass to CMake

Examples:
  $ PARALLEL_LEVEL=8 ./ci/build_cub.sh
  $ PARALLEL_LEVEL=8 ./ci/build_cub.sh -cxx g++-9
  $ ./ci/build_cub.sh -cxx clang++-8
  $ ./ci/build_cub.sh -configure -arch 80
  $ ./ci/build_cub.sh -cxx g++-8 -std 14 -arch 80-real -v -cuda /usr/local/bin/nvcc
  $ ./ci/build_cub.sh -cmake-options "-DCMAKE_BUILD_TYPE=Debug -DCMAKE_CXX_FLAGS=-Wfatal-errors"
```

## Options

| Flag               | Default           | Description                                                             |
|--------------------|-------------------|-------------------------------------------------------------------------|
| `-cxx`             | `$CXX` or `g++`   | Host C++ compiler path or name.                                         |
| `-cuda`            | `$CUDACXX` or `nvcc` | CUDA compiler path or name.                                          |
| `-std`             | `17`              | C++ standard (14, 17, 20).                                              |
| `-arch`            | Preset default    | Semicolon-separated CMake `CUDA_ARCHITECTURES` value.                   |
| `-configure`       | off               | Configure only; skip build and test steps.                              |
| `-v` / `-verbose`  | off               | Enable `set -x` shell tracing for debugging.                            |
| `-pedantic`        | on in CI          | Enables `-DCCCL_ENABLE_WERROR=ON -DCCCL_ENABLE_PRAGMA_SYSTEM_HEADER=OFF`. |
| `-disable-benchmarks` | off           | Force-disable CUB benchmark targets (sets `DISABLE_CUB_BENCHMARKS=1`). |
| `-cmake-options`   | none              | Extra CMake flags appended to the configure command.                    |

## Environment

| Variable                  | Default                      | Effect                                                           |
|---------------------------|------------------------------|------------------------------------------------------------------|
| `PARALLEL_LEVEL`          | `nproc --all --ignore=1`     | Ninja and CTest parallelism.                                     |
| `CXX`                     | `g++`                        | Overrides default host compiler (superseded by `-cxx` flag).     |
| `CUDACXX`                 | `nvcc`                       | Overrides default CUDA compiler (superseded by `-cuda` flag).    |
| `VERBOSE`                 | off                          | Same effect as `-v` when set to non-empty.                       |
| `PEDANTIC`                | auto (on in CI)              | Enable strict warnings. Set to `1` to force on locally.          |
| `CCCL_BUILD_INFIX`        | `""`                         | Subdirectory under `../build/` for per-devcontainer isolation.   |
| `DISABLE_CUB_BENCHMARKS`  | off                          | Skip CUB benchmark targets when set to `1`.                      |
| `CCCL_CI_COMMAND_TIMEOUT` | `5.5h`                       | Per-step timeout in GHA; prevents orphaned jobs.                 |
| `MEMMON`                  | off                          | Enable memory monitor logging outside of GHA.                    |
| `MEMMON_POLL_INTERVAL`    | `5` (sec)                    | Sampling interval for `ci/util/memmon.sh`.                       |
| `MEMMON_LOG_THRESHOLD`    | `2` (GB)                     | Log entry threshold for memory monitor.                          |
| `MEMMON_PRINT_THRESHOLD`  | `5` (GB)                     | Print-to-console threshold for memory monitor.                   |

## Build directory layout

```
../build/
  ${CCCL_BUILD_INFIX}/       ← devcontainer-specific root
    ${PRESET}/               ← per-preset build artifacts
  latest -> ${CCCL_BUILD_INFIX}/
  preset-latest -> ${CCCL_BUILD_INFIX}/${PRESET}/
```

## Key functions

| Function                    | Called by            | What it does                                          |
|-----------------------------|----------------------|-------------------------------------------------------|
| `configure_preset`          | build scripts        | Runs `cmake --preset` with retry on CI.               |
| `build_preset`              | build scripts        | Runs `cmake --build --preset`; starts/stops memmon.   |
| `test_preset`               | build scripts        | Runs `ctest --preset`; prints time summary.           |
| `configure_and_build_preset`| build scripts        | Combines `configure_preset` + `build_preset`.         |
| `print_environment_details` | build scripts        | Logs compilers, versions, GPU info, sccache state.    |
| `run_ci_timed_command`      | build/test functions | Wraps commands with `timeout` in GHA.                 |

## Notes / gotchas

- `PEDANTIC` is automatically enabled inside GitHub Actions even if not passed on the command line.
- The `-arch` flag corresponds to CMake `CMAKE_CUDA_ARCHITECTURES`; use semicolons as separators (not commas).
  See `cccl-build` → `references/arch-flag.md` for all valid forms.
- `sccache` is used automatically when present on `PATH` (standard in the devcontainer).
- `CCCL_CI_COMMAND_TIMEOUT` only applies inside GitHub Actions. Local runs have no timeout.
