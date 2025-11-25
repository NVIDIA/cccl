# Agent Instructions

This document provides guidelines for building, testing, and contributing to the CCCL repository. It is primarily written for agentic AIs, but the information is also useful for CCCL developers.

---

## Overview

CCCL is a collection of CUDA C++ libraries and Python packages:

* **libcudacxx** — CUDA C++ Standard Library
* **CUB** — Block-level primitives
* **Thrust** — High-level parallel algorithms
* **cudax** — Experimental features
* **C Parallel Library** — C bindings for CCCL algorithms
* **Python CCCL packages** (`cuda-cccl`) — Python bindings for parallel and cooperative primitives

The repository uses **CMake** with the **Ninja** generator and provides standardized presets for consistent builds.

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

Presets are defined in `CMakePresets.json`. Names follow a `project` or `<project>-cxx<std>` format, such as `cub-cpp20`, `thrust-cpp17`, or `libcudacxx`. Use `cmake --list-presets` to view available options. Build trees are placed under `build/${CCCL_BUILD_INFIX}/${PRESET}`.

### `.devcontainer/launch.sh`

Launches a container configured with a CUDA Toolkit and host compiler. First startup may take time, but cached environments are faster. In agent environments, container launches may not be supported. To check if you are already inside a container, verify if `CCCL_BUILD_INFIX` is set.

Common options:

* `-d, --docker` — Run without VSCode (required for agents)
* `--cuda <version>` — Select CUDA Toolkit (optional)
* `--host <compiler>` — Select host compiler (optional)
* `--gpus all` — Expose GPUs (omit in GPU-less environments)
* `-e/--env`, `-v/--volume` — Environment variables / volume mounts
* `-- <script>` — Run script inside container after setup

Example:

```bash
.devcontainer/launch.sh -d --cuda 13.0 --host gcc14 -- <script> [args...]
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

### Targeted Build and Test Examples

* **CUB** (`cub/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset cub-cpp20 \
  --build-targets "cub.cpp20.test.iterator" \
  --ctest-targets "cub.cpp20.test.iterator"
```

* **Thrust** (`thrust/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset thrust-cpp20 \
  --build-targets "thrust.cpp20.test.reduce" \
  --ctest-targets "thrust.cpp20.test.reduce"
```

* **libcudacxx** (`libcudacxx/`):
  Avoid the expensive `libcudacxx.cpp20.precompile.lit`. Instead, precompile and run a small set of lit tests:

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
  --build-targets "cudax.cpp20.test.async_buffer" \
  --ctest-targets "cudax.cpp20.test.async_buffer"
```

* **C Parallel API** (`c/parallel/`):

```bash
ci/util/build_and_test_targets.sh \
  --preset cccl-c-parallel \
  --build-targets "cccl.c.test.reduce" \
  --ctest-targets "cccl.c.test.reduce"
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

Supported versions: `3.10`, `3.11`, `3.12`, `3.13`

### Modules

* **cuda.compute** — Device-level algorithms, iterators, custom GPU types
* **cuda.coop** — Block/warp-level primitives
* **cuda.cccl.headers** — Programmatic access to headers

### Installation

From PyPI:

```bash
pip install cuda-cccl[cu13] # or [cu12] for CTK 12.X
```

From source:

```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_cccl
pip install -e .[test-cu13] # or [test-cu12] for CTK 12.X
```

Requirements:

* Python 3.9+
* CUDA Toolkit 12.x
* NVIDIA GPU (CC 6.0+)
* Dependencies: `numba>=0.60.0`, `numpy`, `cuda-bindings>=12.9.1`, `cuda-core`, `numba-cuda>=0.18.0`

### Usage Examples

```python
import cuda.compute
result = cuda.compute.reduce_into(input_array, output_scalar, init_val, binary_op)

from cuda import coop
@cuda.jit
def kernel(data):
    coop.block.reduce(data, binary_op)

import cuda.cccl.headers as headers
include_paths = headers.get_include_paths()
```

### Build and Test

```bash
./ci/build_cuda_cccl_python.sh -py-version 3.10
./ci/test_cuda_cccl_parallel_python.sh -py-version 3.10
./ci/test_cuda_cccl_cooperative_python.sh -py-version 3.10
./ci/test_cuda_cccl_headers_python.sh -py-version 3.10
./ci/test_cuda_cccl_examples_python.sh -py-version 3.10
```

Test organization:

* `tests/parallel` — Algorithms and iterators
* `tests/cooperative` — Cooperative primitives
* `tests/headers` — Header integration
* `examples/` — Usage demonstrations

---

## Continuous Integration (CI)

See `ci-overview.md` for detailed examples and troubleshooting guidance.

CCCL's CI is built on GitHub Actions and relies on a dynamically generated job matrix plus several helper scripts.

### Key Components

* **`ci/matrix.yaml`**

  * Declares build and test jobs for `pull_request`, `nightly`, and `weekly` workflows.
  * Pull request (PR) runs typically spawn ~250 jobs.
  * To reduce overhead, you can add an override matrix in `workflows.override`. This limits the PR CI run to a targeted subset of jobs. Overrides are recommended when:
    * Changes touch high-dependency areas (e.g. top-level CI/devcontainers, libcudacxx, thrust, CUB). See `ci/inspect_changes.py` for dependency information.
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
  * Defines internal dependencies between CCCL projects. If a project is marked dirty, all dependent projects are also marked dirty and tested.
  * Allows `build-workflow.py` to skip unaffected jobs.

---

### Commit Message Controls

Tags appended to the commit summary (case-sensitive) control CI behavior:

* `[skip-matrix]`: Skip CCCL project build/test jobs. (Docs, devcontainers, and third-party builds still run.)
* `[skip-vdc]`: Skip "Verify Devcontainer" jobs. Safe unless CI or devcontainer infra is modified.
* `[skip-docs]`: Skip doc tests/previews. Safe if docs are unaffected.
* `[skip-third-party-testing]` / `[skip-tpt]`: Skip third-party smoke tests (MatX, PyTorch, RAPIDS).
* `[skip-matx]`: Skip building the MatX third-party smoke test.
* `[skip-pytorch]`: Skip building the PyTorch third-party smoke test.

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

---

## General Guidelines

* Validate changes with builds/tests; report results.
* Run `pre-commit` before committing.
* Review `CONTRIBUTING.md` and `ci-overview.md` before starting work.

### Performance Tips

* Use development containers with `sccache` (CCCL team only).
* Limit architectures to reduce compile time (e.g. `-arch "native"` or `"80"` if no GPU).
* Build with Ninja for fast, parallel builds.


---

## Repository Structure

```
cccl/
├── .github/            # Workflows
├── .devcontainer/      # Dev containers
├── libcudacxx/         # CUDA C++ Standard Library
├── cub/                # CUB primitives
├── thrust/             # Thrust algorithms
├── cudax/              # Experimental features
├── c/                  # C Parallel library
├── python/cuda_cccl/   # Python bindings
├── ci/                 # Build/test scripts
├── examples/           # Usage examples
└── CMakePresets.json   # Preset configurations
```

Python package layout:

```
python/cuda_cccl/
├── cuda/cccl/
│   ├── parallel/
│   ├── cooperative/
│   └── headers/
├── tests/
├── benchmarks/
└── pyproject.toml
```

---

⚠️ **Reminder:** Long-running builds/tests are normal. Never cancel them; allow to complete.
