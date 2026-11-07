# Agent Instructions

This document provides guidelines for building, testing, and contributing to the CCCL repository. It is primarily written for agentic AIs, but the information is also useful for CCCL developers.

For `cuda.coop` work, including `python/cuda_coop/**` and
`docs/python/coop*`, read [python/cuda_coop/AGENTS.md](python/cuda_coop/AGENTS.md)
and apply its API, behavior, and example conventions.

---

## Overview

CCCL is a collection of CUDA C++ libraries and Python packages:

* **libcudacxx** — CUDA C++ Standard Library
* **CUB** — Block-level primitives
* **Thrust** — High-level parallel algorithms
* **cudax** — Experimental features
* **C Parallel Library** — C bindings for CCCL algorithms
* **Python CCCL packages** (`cuda-cccl`) — Python APIs for parallel primitives and programmatic access to CCCL headers

The repository uses **CMake** with the **Ninja** generator and provides standardized presets for consistent builds.

---

## Iteration Cycles

For a given task, you should:

1. Research. Search the web, read existing code, look up system/dependency headers / implementations of related functionality. Figure out best practices and common pitfalls. Look for existing tests of the functionality; if none exist, plan a new test that integrates with the relevant existing testing frameworks.
2. Plan. Create a high-level plan to implement the requested feature.
3. Review and Refine plan. Look for pitfalls, find ways to smooth out rough edges. Verify any assumptions, edgecases, or identified pitfalls. Repeat until the plan is solid.
4. Gather consistency context. Look at similar code (sibling classes if possible, otherwise just related source files) to learn the style and patterns used in the project. Consistency is important -- similar features should be organized and implemented similarly. Naming conventions should be followed.
5. If requested: Present the plan. Only do this if the user asks for a plan to do something -- if they just ask you implement something without requesting a plan, skip this step.
6. Draft. Implement the requested task to the best of your ability.
7. Review and Refine. Read through your changes. Verify that API calls are correct. Assess clarity, performance, and readability. Iterate as needed.
8. Style check. Ensure that your changes follow style and naming conventions.
9. Build and test. Once you're confident that your changes are functionally and stylistically correct start build, test, and iterate cycles. If you don't have permissions to do these, ask the user to run specific build/test commands for you.

---

## Known Agent Limitations

### OpenAI Codex

Codex cloud instances cannot:

* Run Docker containers with devcontainer scripts
* Access GPUs or run GPU-dependent tests

---

## Build and Test Tools

All CCCL subprojects are computationally expensive to build and test. Use the provided helper scripts to minimize work and target only what you need.

### CMake Presets

Presets are defined in `CMakePresets.json`. Names follow a `project` or `<project>-cpp<std>` format, such as `cub-cpp20`, `thrust-cpp17`, or `libcudacxx`. Use `cmake --list-presets` to view available options. Build trees are placed under `build/${CCCL_BUILD_INFIX}/${PRESET}`.

### `.devcontainer/launch.sh`

Launches a container configured with a CUDA Toolkit and host compiler. First startup may take time, but cached environments are faster. In agent environments, container launches may not be supported. To check if you are already inside a container, verify if `CCCL_BUILD_INFIX` is set.

Common options:

* `-d, --docker` — Run without VSCode (required for agents)
* `--cuda <version>` — Select CUDA Toolkit (optional)
* `--cuda-ext` — Use a docker image with extended CTK libraries
* `--host <compiler>` — Select host compiler (optional)
* `--gpus <request>` — GPU devices to add to the container (use `all` to pass all GPUs)
* `-e/--env`, `-v/--volume` — Environment variables / volume mounts
* `-- <script>` — Run script inside container after setup

Example:

```bash
.devcontainer/launch.sh -d --cuda 13.4 --host gcc14 -- <script> [args...]
```

### `ci/util/build_and_test_targets.sh`

Configures, builds, and tests selected Ninja, CTest, or lit targets. Many tests require GPUs. Options that generally work without GPUs include `--preset`, `--cmake-options`, `--configure-override`, `--build-targets`, `--lit-precompile-tests`, and `--custom-test-cmd`.

Key options:

* `--preset <name>` — Use a CMake preset
* `--cmake-options <str>` — Extra CMake arguments
* `--configure-override <cmd>` — Custom configuration command
* `--build-targets "<targets>"` — Space-separated Ninja targets
* `--ctest-targets "<regex>"` — Regex for CTest targets (may fail without GPUs)
* `--lit-precompile-tests "<paths>"` — Precompile specified libcudacxx lit tests (paths are relative to `libcudacxx/test/libcudacxx/`)
* `--lit-tests "<paths>"` — Run specified libcudacxx lit tests (also relative to `libcudacxx/test/libcudacxx/`)
* `--custom-test-cmd "<cmd>"` — Run arbitrary command after tests

### `ci/util/git_bisect.sh`

Wraps `git bisect` with the build/test helper. Useful for identifying regression commits. Can take a very long time—minimize scope by restricting build/test targets.

Extra options:

* `--good-ref <rev>` — Known good commit/tag, or `-Nd` for origin/main N days ago (default: latest release)
* `--bad-ref <rev>` — Known bad commit/tag, or `-Nd` (default: origin/main)

See `docs/cccl/development/build_and_bisect_tools.rst` for details.

---

## Building and Testing

Always prefer targeted builds and tests, as full builds are time-consuming. If required tools or hardware are unavailable, note this in the PR but run as many relevant tests as possible.

### Target Names

Target names do not encode the C++ dialect. The preset selects it, except for the C Parallel targets, which fix C++20 in their CMake files. Every dotted prefix of a target is also a Ninja metatarget, so `ninja cub.test.device` builds all CUB device tests; targets created with `NO_METATARGETS`, such as the C Parallel tests, are the exception. `--ctest-targets` takes CTest `-R` regexes, which match substrings: append `$` when a test name is a prefix of others.

* CUB: `cub.test.<name>[.<variant>]`. `<name>` is the source file name without `catch2_test_` or `test_`, with a leading `device_`, `block_`, `warp_`, `thread_`, or `util_` turned into `device.` and so on. Variants come from `// %PARAM%` lines and launcher ids (`lid_0`, `lid_1`, `lid_2`).
* Thrust: `thrust.<host>.<device>.test.<name>`, e.g. `thrust.cpp.cuda.test.reduce`.
* cudax: `cudax.test.<group>`; the groups are defined in `cudax/test/CMakeLists.txt`.
* C Parallel: `cccl.c.parallel.test.<name>` for `c/parallel/test/test_<name>.cpp`.
* libcudacxx: `libcudacxx.test.lit.precompile` precompiles every lit test.

To check a name, run `ninja -t targets all | grep <name>` in the build tree.

### Targeted Build and Test Examples

* **CUB** (`cub/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset cub-cpp20 \
  --build-targets "cub.test.iterator" \
  --ctest-targets "cub.test.iterator"
```

* **Thrust** (`thrust/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset thrust-cpp20 \
  --build-targets "thrust.cpp.cuda.test.reduce" \
  --ctest-targets "thrust.cpp.cuda.test.reduce$"
```

* **libcudacxx** (`libcudacxx/`):
  Avoid the expensive `libcudacxx.test.lit.precompile`. Instead, precompile and run a small set of lit tests:

```bash
ci/util/build_and_test_targets.sh \
  --preset libcudacxx \
  --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
  --lit-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"
```

* **CUDA Experimental** (`cudax/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset cudax \
  --build-targets "cudax.test.containers" \
  --ctest-targets "cudax.test.containers"
```

* **C Parallel API** (`c/parallel/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset cccl-c-parallel \
  --build-targets "cccl.c.parallel.test.reduce" \
  --ctest-targets "cccl.c.parallel.test.reduce"
```

### Full Builds

> ⚠️ **Important:** Full builds are costly. Always allow 60+ minutes for builds and 30+ minutes for tests. Do not cancel once started.

Use scripts like:

```bash
./ci/build_cub.sh [-cxx g++] [-std 17] [-arch "75;80;90;120"]
./ci/build_thrust.sh [-cxx clang++] [-std 17] [-arch "75;80;90;120"]
./ci/build_libcudacxx.sh [-cxx g++] [-std 17] [-arch "75;80;90;120"]
./ci/build_cudax.sh [-cxx g++] [-std 20] [-arch "75;80;90;120"]
./ci/build_cccl_c_parallel.sh [-cxx g++] [-std 17] [-arch "75;80;90;120"]
./ci/build_cuda_cccl_python.sh -py-version 3.10
```

### Architectures

* `<XX>` — Generate PTX and SASS
* `<XX-real>` — Generate only SASS
* `<XX-virtual>` — Generate only PTX
* `native` — Detect host GPU
* `all-major-cccl` — Default for PR builds

### Testing

> ⚠️ Requires an NVIDIA GPU. Tests take 15+ minutes. Use targeted testing whenever possible.

Examples:

```bash
./ci/test_cub.sh -cxx g++ -std 17 -arch "75;80;90;120"
./ci/test_thrust.sh -cxx g++ -std 17 -arch "75;80;90;120"
./ci/test_libcudacxx.sh -cxx g++ -std 17 -arch "75;80;90;120"
./ci/test_cudax.sh -cxx g++ -std 20 -arch "75;80;90;120"
ctest --preset=cub-cpp17
```

Options:

* `-compute-sanitizer-memcheck` — Run with memory checking or other compute-sanitizer tools (not all projects support this)

---

## Python CCCL Packages

Python components require different parameters than C++ builds. Use `-py-version` instead of compiler flags.

Supported versions: `3.10`, `3.11`, `3.12`, `3.13`, `3.14`

### Modules

* **cuda.compute** — Device-level algorithms, iterators, custom GPU types
* **cuda.stf._experimental** — Sequential Task Flow (CUDASTF) Python bindings in the `cuda-stf` package (Linux only)
* **cuda.cccl.headers** — Programmatic access to headers

### Installation

From PyPI:

```bash
pip install cuda-cccl[cu13] # or [cu12] for CTK 12.X
```

From conda-forge:

```bash
conda install -c conda-forge cccl-python
```

From source:

```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_cccl
pip install -e .[test-cu13] # or [test-cu12] for CTK 12.X
```

Requirements:

* Python 3.10+
* CUDA Toolkit 12.x or 13.x
* NVIDIA GPU (CC 7.5+)
* Base dependencies: `numba>=0.60.0`, `numpy`, `cuda-pathfinder>=1.2.3`, `cuda-core`, `typing_extensions`
* CUDA extras: `cuda-bindings` + `cuda-toolkit` + `numba-cuda` via `cuda-cccl[cu12]` or `cuda-cccl[cu13]`

### Usage Examples

```python
import cuda.compute
result = cuda.compute.reduce_into(input_array, output_scalar, init_val, binary_op)

import cuda.cccl.headers as headers
include_paths = headers.get_include_paths()
```

### Build and Test

```bash
./ci/build_cuda_cccl_python.sh -py-version 3.10
./ci/test_cuda_compute_python.sh -py-version 3.10
./ci/test_cuda_cccl_headers_python.sh -py-version 3.10
./ci/test_cuda_cccl_examples_python.sh -py-version 3.10
./ci/test_cuda_stf_python.sh -py-version 3.10  # Linux only
```

Test organization:

* `tests/compute` — Algorithms and iterators
* `tests/headers` — Header integration
* `python/cuda_stf/tests/stf` — Sequential Task Flow (separate `cuda-stf` package, Linux only)
* `test_examples.py` — Runs compute examples (STF examples live in `python/cuda_stf/tests/test_examples.py`)

---

## Continuous Integration (CI)

See `docs/infrastructure/ci/references/ci_overview.rst` for detailed examples and troubleshooting guidance.

CCCL's CI is built on GitHub Actions and relies on a dynamically generated job matrix plus several helper scripts.

### Key Components

* **`ci/matrix.yaml`**

  * Declares build and test jobs for `pull_request`, `nightly`, and `weekly` workflows.
  * Pull request (PR) runs typically spawn ~250 jobs.
  * To reduce overhead, you can add an override matrix in `workflows.override`. This limits the PR CI run to a targeted subset of jobs. Overrides are recommended when:
    * Changes touch high-dependency areas (e.g. top-level CI/devcontainers, libcudacxx, thrust, CUB). See `ci/project_files_and_dependencies.yaml` for dependency information.
    * A smaller subset of jobs is enough to validate the change (e.g. infra changes, targeted fixes).
  * Important rules:
    * PR merges are blocked while an override matrix is active.
    * The override must be reset to empty (not removed) before merging.
    * Only add overrides when starting a new draft that qualifies; never remove one without being asked.

* **`.github/actions/workflow-build/`**

  * Runs `build-workflow.py`.
  * Reads `ci/matrix.yaml` and prunes jobs using `ci/inspect_changes.py`.
  * Calls `prepare-workflow-dispatch.py` to produce a formatted job matrix for dispatch.

* **`.github/actions/workflow-run-job-{linux,windows}/`**

  * Runs a single matrix job inside a devcontainer.

* **`.github/actions/workflow-results/`**

  * Aggregates artifacts and results.
  * Marks workflow as failed if any job fails or an override matrix is present.

* **`.github/workflows/ci-workflow-{pull-request,nightly,weekly}.yml`**

  * Top-level GitHub Actions workflows invoking CI.

* **`ci/inspect_changes.py`**

  * Detects which subprojects changed between commits.
  * Reads project paths and dependencies from `ci/project_files_and_dependencies.yaml`. Each library is split into a public-header project and a tests/infra project. A change to a library's public headers runs its full matrix plus a reduced ("lite") matrix for every project that lists those headers as a dependency. A change confined to a library's tests does not trigger other libraries, only infra projects such as `tidy` and `packaging`. To see what a change selects, run `python3 ci/inspect_changes.py --refs origin/main HEAD`.
  * Allows `build-workflow.py` to skip unaffected jobs.

---

### Commit Message Controls

Tags appended to the commit summary (case-sensitive) control PR CI behavior:

* `[bench-only]`: Skip all non-benchmark CI jobs. Equivalent to `[skip-matrix][skip-vdc][skip-docs]`.
* `[skip-matrix]`: Skip CCCL project build/test jobs. (Docs and devcontainers still run.)
* `[skip-vdc]`: Skip "Verify Devcontainer" jobs. Safe unless CI or devcontainer infra is modified.
* `[skip-docs]`: Skip doc tests/previews. Safe if docs are unaffected.
* `[skip-compile-time-bench]`: Skip informational compile-time benchmark telemetry. Safe if compile-time benchmark scripts/configuration are unaffected.
* `[skip-sass-diff]`: Skip the informational CUB benchmark SASS comparison. The job already runs only when `ci/inspect_changes.py` marks CUB dirty, either directly or through a dependency such as libcudacxx or Thrust, so this tag is only necessary to skip a comparison that would otherwise run.

Third-party smoke tests (MatX, PyTorch, RAPIDS) run in nightly CI only.

> ⚠️ All of these tags block merging until removed and a full CI run (with no overrides) succeeds.

Use these tags for early iterations to save resources. Remove them before review/merge.

---

## Code Formatting and Linting

> ⚠️ Always run before committing. CI will fail otherwise.

```bash
pip install pre-commit
pre-commit install
pre-commit run --all-files
pre-commit run --files <file1> <file2>
```

### Style Guidance

When editing or reviewing CCCL code for style, read `.agent/skills/cccl-style/SKILL.md`. It routes to the common CCCL style reference and any path-specific style reference that applies to the files being changed.

### Test Guidance

When writing, updating, reviewing, or validating CCCL tests, read `.agent/skills/cccl-test/SKILL.md`. It routes to the common CCCL test reference and any path-specific test reference that applies to the files being changed.

---

## General Guidelines

* Validate changes with builds/tests; report results.
* Run `pre-commit` before committing.
* Review `CONTRIBUTING.md` and `docs/infrastructure/ci/references/ci_overview.rst` before starting work.

### Performance Tips

* Use development containers with `sccache` (CCCL team only).
* Limit architectures to reduce compile time (e.g. `-arch "native"` or `"80"` if no GPU).
* Build with Ninja for fast, parallel builds.


---

## Repository Structure

```
cccl/
├── .agent/skills/      # Agent skills (style, tests, review, benchmarking, SASS diff)
├── .github/            # Workflows
├── .devcontainer/      # Dev containers
├── libcudacxx/         # CUDA C++ Standard Library
├── cub/                # CUB primitives
├── thrust/             # Thrust algorithms
├── cudax/              # Experimental features
├── c/                  # C libraries: parallel/ (v1), parallel.v2/, experimental/stf/
├── c2h/                # Catch2 test harness shared by CUB, cudax, libcudacxx, and the C library
├── nvbench_helper/     # NVBench glue for benchmarks
├── benchmarks/         # Shared benchmark CMake and scripts
├── python/             # Python packages: cuda_cccl, cuda_coop, cuda_stf
├── ci/                 # Build/test scripts
├── docs/               # Sphinx documentation sources
├── examples/           # Usage examples
├── test/               # Cross-project tests (CUDA smoke tests, stdpar)
└── CMakePresets.json   # Preset configurations
```

Python package layout:

```
python/cuda_cccl/
├── cuda/
│   ├── compute/
│   └── cccl/
│       ├── parallel/
│       └── headers/
├── tests/
├── benchmarks/
└── pyproject.toml
```

`python/cuda_coop/` (`cuda.coop`) and `python/cuda_stf/` (`cuda.stf`) are separate packages with their own `pyproject.toml` and `tests/`.

---

## Architecture

### Library Layering

* **libcudacxx** is the base. `libcudacxx/include/cuda/std/` implements `cuda::std::`, and `libcudacxx/include/cuda/` adds `cuda::` extensions. `cuda/std/__cccl/` defines the `_CCCL_*` compiler, dialect, and visibility macros that every other library uses.
* **CUB** (`cub/cub/`) builds on libcudacxx. **Thrust** (`thrust/thrust/`) builds on both: its CUDA backend in `thrust/system/cuda/` calls CUB, and `cpp/`, `omp/`, and `tbb/` are host backends.
* **cudax** (`cudax/include/cuda/experimental/`, namespace `cuda::experimental`) builds on all three and includes CUDASTF (`__stf/`).

### Header Conventions

libcudacxx headers include `<cuda/std/detail/__config>`, then the system-header pragma block, then `<cuda/std/__cccl/prologue.h>` (closed by `epilogue.h`), and open their namespace with a macro such as `_CCCL_BEGIN_NAMESPACE_CUDA_STD`. CUB and Thrust use `CUB_NAMESPACE_BEGIN` and `THRUST_NAMESPACE_BEGIN`. Start a new header from a neighboring one. The macros are documented in `docs/cccl/development/macro.rst`.

### CUB Device Algorithms

A device-wide algorithm is layered across `cub/cub/`:

1. `device/device_<algo>.cuh`: public entry points.
2. `device/dispatch/dispatch_<algo>.cuh`: temporary storage, policy selection, and kernel launch.
3. `device/dispatch/tuning/tuning_<algo>.cuh`: per-architecture tuning policies.
4. `agent/`: per-thread-block tile processing.
5. `block/`, `warp/`, `thread/`: the cooperative primitives the agents are built from.

### Test Frameworks

* libcudacxx: lit tests under `libcudacxx/test/libcudacxx/{std,cuda}/`.
* CUB, cudax, and the C library: Catch2, through the shared `c2h/` harness (runner and data generators). CUB tests run under three launchers: `lid_0` launches from the host, `lid_1` from the device (CDP, built with RDC), and `lid_2` through a CUDA graph. The `cub-lid0`, `cub-lid1`, and `cub-lid2` presets each build one launcher.
* Thrust: its own framework in `thrust/testing/unittest/`.

### C Library and Python

`c/parallel` (v1) JIT-compiles CUB kernels at runtime with NVRTC and nvJitLink; `c/parallel.v2` uses HostJIT instead. `python/cuda_cccl` (`cuda.compute`) builds against v1 by default and against v2 when `CCCL_PYTHON_USE_V2=ON`. v2 has its own preset (`cccl-c-parallel-v2`) and CI scripts (`ci/*_v2.sh`).

---

⚠️ **Reminder:** Long-running builds/tests are normal. Never cancel them; allow to complete.
