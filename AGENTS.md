# CUDA Core Compute Libraries (CCCL) Development Guide

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

CCCL is a collection of CUDA C++ libraries and Python packages: **libcudacxx** (CUDA C++ Standard Library), **CUB** (block-level primitives), **Thrust** (high-level parallel algorithms), **cudax** (experimental features), **C parallel library**, and **Python CCCL packages** (cuda-cccl). The repository uses CMake with Ninja generator and provides standardized presets for consistent builds.

## Known Agent Limitations

### OpenAI Codex

OpenAI Codex Cloud instances cannot:

- Run docker containers using the devcontainer scripts.
- Access GPUs or run tests that require a GPU.

## Build and Test Tools
All subprojects are expensive to compile and test. Use the provided helper scripts to target the smallest set of work.

### CMake Presets
Presets live in `CMakePresets.json` at the repo root. Names follow a `<project>-cxx<std>` style such as `cub-cpp20`, `thrust-cpp17`, or `libcudacxx-cpp20`. See the file or run `cmake --list-presets` for a complete list. When a preset is used, the build tree is placed in `build/${CCCL_BUILD_INFIX}/${PRESET}`.

### `.devcontainer/launch.sh`

Launch a container configured with a specific CUDA Toolkit and host compiler. Initial startup is costly, but subsequent usage is cheaper once cached. The agent may not even be able to launch docker images, so prefer local builds when the environment is suitable. The agent may already be running inside of such a container -- check for the presence of the `CCCL_BUILD_INFIX` environment variable. If set, it will return information about the current CTK and host compiler.

Common options:
- `-d`, `--docker` – run without VSCode; required for agent use
- `--cuda <version>` – choose CUDA Toolkit version (optional; defaults to a recent release)
- `--host <compiler>` – select host compiler (optional; defaults to a recent compiler)
- `--gpus all` – expose all GPUs (omit in agent environment that lacks GPUs)
- `-e/--env`, `-v/--volume` – set environment variables or mount volumes
- `-- <script>` – run a script in the container after setup; arguments after `--` are passed to that script

Example:
```bash
.devcontainer/launch.sh -d --cuda 13.0 --host gcc14 -- <script> [script_args...]
```

### `ci/util/build_and_test_targets.sh`

Configures, builds, and tests selected Ninja, CTest, and lit targets.

Many tests require GPUs and a CUDA driver. Options that generally work without GPUs are `--preset`, `--cmake-options`, `--configure-override`, `--build-targets`, `--lit-precompile-tests`, and `--custom-test-cmd`.

Omitting an option skips that phase entirely; for example, leaving out `--build-targets` means nothing is built.

Either `--preset` or `--configure-override` are required. When `--configure-override` is given, both `--preset` and `--cmake-options` are ignored.

### Common Options:

- `--preset <name>`
  CMake preset to configure the build.

- `--cmake-options <str>`
  Additional CMake arguments.

- `--configure-override <cmd>`
  Custom configuration command (overrides preset).

- `--build-targets "<targets>"`
  Space-separated Ninja targets to build.

- `--ctest-targets "<regex>"`
  Space-separated CTest `-R` patterns (tests may fail without GPUs).

- `--lit-precompile-tests "<paths>"`
  Compile only specified libcudacxx lit tests. Paths are relative to `libcudacxx/test/libcudacxx`.

- `--lit-tests "<paths>"`
  Execute specified libcudacxx lit tests. Paths are relative to `libcudacxx/test/libcudacxx` and may require GPUs.

- `--custom-test-cmd "<cmd>"`
  Run an arbitrary command after tests; a non-zero exit code stops the script.

There are project-specific examples below.

### `ci/util/git_bisect.sh`

Wraps `git bisect` around the build/test helper. This can be useful for investigating regressions and finding culprit commits. This can take a *very* long time to execute, it is essential restrict the build/test targets as much as possible.

Accepts all options from `build_and_test_targets.sh` plus:
- `--good-ref <rev>` – known good commit, tag, or `-Nd` for origin/main N days ago (defaults to latest release)
- `--bad-ref <rev>` – known bad commit, tag, or `-Nd` for origin/main N days ago (defaults to `origin/main`)

See `docs/cccl/development/build_and_bisect_tools.rst` for complete usage information.

## Building and Testing

There are many ways to interact with CCCL's build system. We'll describe them here. Always prefer more targeted approaches when possible, as these are **significantly** faster than doing full builds.

If required tools or hardware are unavailable, note this in the PR but do your best to run relevant tests.

### Targeted Build / Test Commands

Each project has its own set of build and test targets. Invoke `ci/util/build_and_test_targets.sh` with the appropriate library / C++ standard preset. Adjust target names to just those needed to build/test the features you are modifying.

- **CUB (`cub/`):**

  ```bash
  ci/util/build_and_test_targets.sh \
    --preset cub-cpp20 \
    --build-targets "cub.cpp20.test.iterator" \
    --ctest-targets "cub.cpp20.test.iterator"
  ```

- **Thrust (`thrust/`):**

  ```bash
  ci/util/build_and_test_targets.sh \
    --preset thrust-cpp20 \
    --build-targets "thrust.cpp20.test.reduce" \
    --ctest-targets "thrust.cpp20.test.reduce"
  ```

- **libcudacxx (`libcudacxx/`):**

  Avoid the expensive `libcudacxx.cpp20.precompile.lit` target; precompile only
  a few lit tests at a time.

  ```bash
  ci/util/build_and_test_targets.sh \
    --preset libcudacxx-cpp20 \
    --lit-precompile-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp" \
    --lit-tests "std/algorithms/alg.nonmodifying/alg.any_of/any_of.pass.cpp"
  ```

- **CUDA Experimental (`cudax/`):**

  ```bash
  ci/util/build_and_test_targets.sh \
    --preset cudax-cpp20 \
    --build-targets "cudax.cpp20.test.async_buffer" \
    --ctest-targets "cudax.cpp20.test.async_buffer"
  ```

- **C Parallel API (`c/parallel/`):**

  ```bash
  ci/util/build_and_test_targets.sh \
    --preset cccl-c-parallel \
    --build-targets "cccl.c.test.reduce" \
    --ctest-targets "cccl.c.test.reduce"
  ```

### Full Build Commands - NEVER CANCEL, Always Use Long Timeouts

**CRITICAL TIMING**: All build operations can take 45+ minutes. Test operations require GPU and take 15+ minutes. Always set timeouts to 60+ minutes for builds and 30+ minutes for tests. Because of the extreme cost of these methods, always prefer the `build_and_test_targets.sh` approach above.

#### Using CI Build Scripts

**Build individual components** (60+ minute timeout):
```bash
# Build CUB - NEVER CANCEL, takes up to 45 minutes
./ci/build_cub.sh [-cxx g++] [-std 17] [-arch "75;80;90;120"]

# Build Thrust - NEVER CANCEL, takes up to 45 minutes
./ci/build_thrust.sh [-cxx clang++] [-std 17] [-arch "75;80;90;120"]

# Build libcudacxx - NEVER CANCEL, takes up to 45 minutes
./ci/build_libcudacxx.sh [-cxx g++] [-std 17] [-arch "75;80;90;120"]

# Build cudax (experimental) - NEVER CANCEL, takes up to 45 minutes
./ci/build_cudax.sh [-cxx g++] [-std 20] [-arch "75;80;90;120"]

# Build C parallel library - NEVER CANCEL, takes up to 30 minutes
./ci/build_cccl_c_parallel.sh [-cxx g++] [-std 17] [-arch "75;80;90;120"]

# Build Python packages - NEVER CANCEL, takes up to 20 minutes
./ci/build_cuda_cccl_python.sh -py-version 3.10
```

**Available compiler/standard combinations:**
- Host compilers: `g++`, `clang++` (version 7+ for GCC, 14+ for Clang)
- C++ standards: `17`, `20` (cudax requires C++20)
- CUDA architectures:
  - < 75: not supported
  - `"75;80;90"` (default - PTX + SASS for all)
  - `"75;80-virtual"` (PTX + SASS for 75; PTX only for 80)
  - `"75-real;80"` (SASS only for 75, PTX + SASS for 80)
  - `"80"` (single architecture for faster builds)
  - `"native"` (automatically detect current GPU)
  - `"all-major-cccl"` (default arches for CCCL PR builds)

**Architecture format:**
- `XX` - Generate both PTX and SASS
- `XX-real` - Generate only SASS (device code)
- `XX-virtual` - Generate only PTX (forward compatibility)

#### Using CMake Presets (Alternative)

**Configure and build** (60+ minute timeout):
```bash
# List available presets
cmake --list-presets

# Configure - takes up to 10 minutes
cmake --preset=cub-cpp17

# Build - NEVER CANCEL, takes up to 45 minutes
cmake --build --preset=cub-cpp17

# Build all components - NEVER CANCEL, takes up to 60 minutes
cmake --preset=all-dev
cmake --build --preset=all-dev
```

**Common presets:**
- `cub-cpp17`, `cub-cpp20` - CUB library
- `thrust-cpp17`, `thrust-cpp20` - Thrust library
- `libcudacxx-cpp17`, `libcudacxx-cpp20` - libcudacxx library
- `all-dev` - All components with examples and tests
- `install` - Installation/packaging build

#### Testing - NEVER CANCEL, Requires GPU

**CRITICAL**: Testing requires NVIDIA GPU with drivers. Tests take 15+ minutes, use 30+ minute timeouts. Prefer targeted build/test commands.

```bash
# Test individual components - NEVER CANCEL, takes up to 15 minutes each
./ci/test_cub.sh -cxx g++ -std 17 -arch "75;80;90;120"
./ci/test_thrust.sh -cxx g++ -std 17 -arch "75;80;90;120"
./ci/test_libcudacxx.sh -cxx g++ -std 17 -arch "75;80;90;120"
./ci/test_cudax.sh -cxx g++ -std 20 -arch "75;80;90;120"

# Test C parallel library - NEVER CANCEL, takes up to 10 minutes
./ci/test_cccl_c_parallel.sh -cxx g++ -std 17 -arch "75;80;90;120"

# Test Python packages - NEVER CANCEL, takes up to 10 minutes
./ci/test_cuda_cccl_parallel_python.sh -py-version 3.10
./ci/test_cuda_cccl_headers_python.sh -py-version 3.10
./ci/test_cuda_cccl_examples_python.sh -py-version 3.10
./ci/test_cuda_cccl_cooperative_python.sh -py-version 3.10

# Using CMake presets - NEVER CANCEL, takes up to 15 minutes
ctest --preset=cub-cpp17
```

**Test configuration options:**
- `-no-lid` - Run without limited device identifiers
- `-lid0`, `-lid1`, `-lid2` - Test with specific device limitations
- `-compute-sanitizer-memcheck` - Run with memory checking (slower)

## Python CCCL Packages

**CRITICAL**: Python packages require different build parameters than C++ components. Use `-py-version X.Y` format instead of `-cxx` and `-std` parameters.

**Available Python versions:** `3.10`, `3.11`, `3.12`, `3.13`

### Python Package Components

CCCL provides comprehensive Python bindings and algorithms:

**cuda.cccl.parallel** - Device-level parallel algorithms (experimental):
- **Algorithms**: `reduce_into`, `exclusive_scan`, `inclusive_scan`, `radix_sort`, `merge_sort`, `histogram_even`, `unique_by_key`, `binary_transform`, `unary_transform`, `segmented_reduce`
- **Iterators**: `CountingIterator`, `ConstantIterator`, `TransformIterator`, `ReverseInputIterator`, `ZipIterator`, `CacheModifiedInputIterator`
- **Custom types**: `gpu_struct` decorator for user-defined data types
- **Performance**: Hand-optimized CUDA kernels, portable across GPU architectures

**cuda.cccl.cooperative** - Block and warp-level cooperative algorithms:
- **Block-level primitives**: reduce, scan, sort operations within thread blocks
- **Warp-level primitives**: cooperative operations within warps
- **Integration**: Designed for use within Numba CUDA kernels
- **Custom kernels**: Building blocks for advanced CUDA kernel development

**cuda.cccl.headers** - Include paths for CCCL headers in Python:
- **Header access**: Programmatic access to libcudacxx, CUB, Thrust, and cudax headers
- **Integration**: Use with Numba, PyCUDA, or other Python CUDA libraries
- **Path management**: `get_include_paths()` function for compiler integration

### Python Installation and Usage

**Install from PyPI** (recommended for users):
```bash
pip install cuda-cccl
```

**Install from source** (for development):
```bash
git clone https://github.com/NVIDIA/cccl.git
cd cccl/python/cuda_cccl
pip install -e .[test]  # Development mode with test dependencies
```

**Requirements:**
- Python 3.9+
- CUDA Toolkit 12.x
- NVIDIA GPU with Compute Capability 6.0+
- Dependencies: `numba>=0.60.0`, `numpy`, `cuda-bindings>=12.9.1`, `cuda-core`, `numba-cuda>=0.18.0`

**Basic usage examples:**
```python
# Device-level parallel reduction
import cuda.cccl.parallel.experimental as parallel
result = parallel.reduce_into(input_array, output_scalar, initial_value, binary_op)

# Cooperative block-level reduction in Numba kernel
import cuda.cccl.cooperative.experimental as cooperative
@cuda.jit
def my_kernel(data):
    cooperative.block.reduce(data, binary_op)

# Header paths for custom compilation
import cuda.cccl.headers as headers
include_paths = headers.get_include_paths()
```

### Python Build and Test Parameters

**Python build commands** use different parameter format:
```bash
# Build Python wheel - NEVER CANCEL, takes up to 20 minutes
./ci/build_cuda_cccl_python.sh -py-version 3.10

# Test Python modules - NEVER CANCEL, takes up to 10 minutes each
./ci/test_cuda_cccl_parallel_python.sh -py-version 3.10     # Parallel algorithms
./ci/test_cuda_cccl_cooperative_python.sh -py-version 3.10  # Cooperative primitives
./ci/test_cuda_cccl_headers_python.sh -py-version 3.10      # Header access
./ci/test_cuda_cccl_examples_python.sh -py-version 3.10     # Example validation
```

**Python test organization:**
- **Parallel tests**: Located in `python/cuda_cccl/tests/parallel/` - covers algorithms, iterators, custom types
- **Cooperative tests**: Located in `python/cuda_cccl/tests/cooperative/` - covers block/warp primitives
- **Header tests**: Located in `python/cuda_cccl/tests/headers/` - validates include path access
- **Examples**: Comprehensive example collection demonstrating usage patterns

**Pytest markers and options:**
- `pytest -n auto` - Parallel test execution
- `pytest -n 6` - Specific parallel worker count
- `pytest -m "not large"` - Skip large memory tests
- `pytest -m "large"` - Run only large memory tests (requires significant GPU memory)

## Code Formatting and Linting

**CRITICAL**: Always run before committing or CI will fail.

```bash
# If needed, install and run pre-commit hooks - takes 2-5 minutes
pip install pre-commit

# Run on all files
pre-commit run --all-files

# Format and lint specific changes only - takes 30 seconds
pre-commit run

# Target a specific set of files
pre-commit run --files <file1> <file2> ...

# Auto-install hooks to run on every commit
pre-commit install
```

## General Rules:

- Always validate changes by building and testing affected components.
- Always run `pre-commit` formatting before committing.
- Review `CONTRIBUTING.md` for contribution guidelines

### Performance Tips:

- When possible, use Development Containers with `sccache` for faster builds (authentication required for NVIDIA employees)
- Limit CUDA architectures to your target hardware only: `-arch "native"` or `--cmake-options "-DCMAKE_CUDA_ARCHITECTURES=native"`.
- Use `ninja` with parallel builds: `cmake --build --preset=<preset> --parallel <N>`


## Repository Structure
```
cccl/
├── .github/             # GitHub workflows and configurations
├── .devcontainer/       # Development container configs
├── libcudacxx/          # CUDA C++ Standard Library headers
├── cub/                 # CUB block-level primitives
├── thrust/              # Thrust parallel algorithms
├── cudax/               # Experimental CUDA features (C++20)
├── c/                   # CCCL C parallel library
├── python/cuda_cccl/    # Python bindings and parallel algorithms
├── ci/                  # Build and test scripts
├── examples/            # Usage examples
└── CMakePresets.json    # Standardized build configurations
```

**Python package structure:**
```
python/cuda_cccl/
├── cuda/cccl/
│   ├── parallel/         # Device-level parallel algorithms (experimental)
│   │   └── experimental/ # Algorithms, iterators, custom types
│   ├── cooperative/      # Block and warp-level cooperative algorithms
│   │   └── experimental/ # Block/warp primitives for Numba kernels
│   └── headers/          # Include paths for CCCL headers
├── tests/                # Comprehensive test suites
│   ├── parallel/         # Algorithm and iterator tests
│   ├── cooperative/      # Block/warp primitive tests
│   └── headers/          # Header access tests
├── benchmarks/           # Performance benchmarks
└── pyproject.toml        # Package configuration and dependencies
```

**REMINDER**: NEVER CANCEL long-running builds or tests. The timings above are normal and expected. Always wait for completion.
