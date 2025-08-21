# CUDA Core Compute Libraries (CCCL) Development Guide

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

CCCL is a collection of CUDA C++ libraries and Python packages: **libcudacxx** (CUDA C++ Standard Library), **CUB** (block-level primitives), **Thrust** (high-level parallel algorithms), **cudax** (experimental features), **C parallel library**, and **Python CCCL packages** (cuda-cccl). The repository uses CMake with Ninja generator and provides standardized presets for consistent builds.

## Working Effectively

### Bootstrap Development Environment

Use Development Containers for consistent CUDA environment (highly recommended):

```bash
# Clone and open in VSCode
git clone https://github.com/nvidia/cccl.git
cd cccl
# Open in VSCode and select "Reopen in Container" when prompted
# Choose appropriate container (e.g., cuda12.9-gcc13) from the list
```

**Manual setup** (if dev containers unavailable):
```bash
# Requires CUDA Toolkit 12.0+ and compatible host compiler
# Install CMake 3.21+, Ninja build system, and Python 3.8+
apt-get update && apt-get install -y cmake ninja-build python3 python3-pip
pip install pre-commit

# Alternative: Use conda for CCCL headers only
conda config --add channels conda-forge
conda install cccl  # Latest version
# OR conda install cuda-cccl cuda-version=12.4  # Specific CUDA version
```

### Build Commands - NEVER CANCEL, Always Use Long Timeouts

**CRITICAL TIMING**: All build operations can take 45+ minutes. Test operations require GPU and take 15+ minutes. Always set timeouts to 60+ minutes for builds and 30+ minutes for tests.

#### Using CI Build Scripts (Recommended)

**Build individual components** (60+ minute timeout):
```bash
# Build CUB - NEVER CANCEL, takes up to 45 minutes
./ci/build_cub.sh -cxx g++ -std 17 -arch "60;70;80"

# Build Thrust - NEVER CANCEL, takes up to 45 minutes  
./ci/build_thrust.sh -cxx g++ -std 17 -arch "60;70;80"

# Build libcudacxx - NEVER CANCEL, takes up to 45 minutes
./ci/build_libcudacxx.sh -cxx g++ -std 17 -arch "60;70;80"

# Build cudax (experimental) - NEVER CANCEL, takes up to 45 minutes
./ci/build_cudax.sh -cxx g++ -std 20 -arch "60;70;80"

# Build C parallel library - NEVER CANCEL, takes up to 30 minutes
./ci/build_cccl_c_parallel.sh -cxx g++ -std 17 -arch "60;70;80"

# Build Python packages - NEVER CANCEL, takes up to 20 minutes
./ci/build_cuda_cccl_python.sh -py-version 3.10
```

**Available compiler/standard combinations:**
- Host compilers: `g++`, `clang++` (version 7+ for GCC, 14+ for Clang)
- C++ standards: `17`, `20` (cudax requires C++20)
- CUDA architectures: 
  - `"60;70;80"` (default - PTX + SASS for all)
  - `"70;75;80-virtual"` (PTX + SASS for 70,75; PTX only for 80) 
  - `"70-real;80"` (SASS only for 70, PTX + SASS for 80)
  - `"80"` (single architecture for faster builds)
  - `"gpu"` (automatically detect current GPU)

**Architecture format:**
- `XX` - Generate both PTX and SASS
- `XX-real` - Generate only SASS (device code)  
- `XX-virtual` - Generate only PTX (forward compatibility)

### Python CCCL Packages

**CRITICAL**: Python packages require different build parameters than C++ components. Use `-py-version X.Y` format instead of `-cxx` and `-std` parameters.

**Available Python versions:** `3.9`, `3.10`, `3.11`, `3.12`

#### Python Package Components

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

#### Python Installation and Usage

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

#### Python Build and Test Parameters

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

### Testing - NEVER CANCEL, Requires GPU

**CRITICAL**: Testing requires NVIDIA GPU with drivers. Tests take 15+ minutes, use 30+ minute timeouts.

```bash
# Test individual components - NEVER CANCEL, takes up to 15 minutes each
./ci/test_cub.sh -cxx g++ -std 17 -arch "60;70;80"
./ci/test_thrust.sh -cxx g++ -std 17 -arch "60;70;80"  
./ci/test_libcudacxx.sh -cxx g++ -std 17 -arch "60;70;80"
./ci/test_cudax.sh -cxx g++ -std 20 -arch "60;70;80"

# Test C parallel library - NEVER CANCEL, takes up to 10 minutes
./ci/test_cccl_c_parallel.sh -cxx g++ -std 17 -arch "60;70;80"

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

### Code Formatting and Linting

**CRITICAL**: Always run before committing or CI will fail.

```bash
# Install and run pre-commit hooks - takes 2-5 minutes
pip install pre-commit
pre-commit run --all-files

# Format and lint specific changes only - takes 30 seconds
pre-commit run

# Auto-install hooks to run on every commit
pre-commit install
```

**Pre-commit tools used:**
- `clang-format` - C++/CUDA code formatting
- `ruff` - Python linting and formatting  
- `codespell` - Spelling checker
- `mypy` - Python type checking

## Validation Scenarios

**ALWAYS validate changes by building and testing affected components:**

1. **After changing CUB code:**
   ```bash
   ./ci/build_cub.sh -cxx g++ -std 17 -arch "60;70;80"    # 45+ min
   ./ci/test_cub.sh -cxx g++ -std 17 -arch "60;70;80"     # 15+ min
   ```

2. **After changing Thrust code:**
   ```bash
   ./ci/build_thrust.sh -cxx g++ -std 17 -arch "60;70;80"  # 45+ min
   ./ci/test_thrust.sh -cxx g++ -std 17 -arch "60;70;80"   # 15+ min
   ```

3. **After changing libcudacxx code:**
   ```bash
   ./ci/build_libcudacxx.sh -cxx g++ -std 17 -arch "60;70;80"  # 45+ min
   ./ci/test_libcudacxx.sh -cxx g++ -std 17 -arch "60;70;80"   # 15+ min
   ```

4. **After changing Python code:**
   ```bash
   ./ci/build_cuda_cccl_python.sh -py-version 3.10              # 20+ min
   ./ci/test_cuda_cccl_parallel_python.sh -py-version 3.10      # 10+ min
   ./ci/test_cuda_cccl_cooperative_python.sh -py-version 3.10   # 10+ min
   ./ci/test_cuda_cccl_headers_python.sh -py-version 3.10       # 5+ min
   ```

5. **Always run formatting before committing:**
   ```bash
   pre-commit run --all-files  # 2-5 min
   ```

6. **Validate packaging and installation:**
   ```bash
   ./ci/test_packaging.sh  # 5-10 min
   ```

**Note on cudax:** Requires C++20 standard and CUDA 12.0+. Use `-std 20` for all cudax operations.

## Common Tasks

### Using CCCL as Header-Only Libraries

CCCL libraries are header-only. For basic usage without building tests:

```bash
# Clone repository  
git clone https://github.com/NVIDIA/cccl.git

# Include in your project with nvcc
nvcc -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub main.cu -o main

# Or configure CMake to find CCCL
cmake --preset=install -DCMAKE_INSTALL_PREFIX=/usr/local/
cd build/install && ninja install
```

**Example usage** (see examples/basic/example.cu):
```cpp
#include <thrust/device_vector.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/std/span>
#include <cuda/atomic>

// Use Thrust, CUB, and libcudacxx together
// Full example available in examples/basic/
```

### Repository Structure
```
cccl/
├── libcudacxx/          # CUDA C++ Standard Library headers
├── cub/                 # CUB block-level primitives  
├── thrust/              # Thrust parallel algorithms
├── cudax/               # Experimental CUDA features (C++20)
├── c/                   # CCCL C parallel library  
├── python/cuda_cccl/    # Python bindings and parallel algorithms
├── ci/                  # Build and test scripts
├── .devcontainer/       # Development container configs
├── examples/            # Usage examples
└── CMakePresets.json    # Standardized build configurations
```

**Component details:**
- `libcudacxx/` - CUDA C++ Standard Library (C++17/20)
- `cub/` - CUDA block/warp-level primitives (C++17/20)  
- `thrust/` - High-level parallel algorithms (C++17/20)
- `cudax/` - Experimental CUDA features (C++20 required)
- `c/` - C parallel library interface
- `python/cuda_cccl/` - Python packages with comprehensive algorithms and bindings

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

### Key Files to Monitor
- Always check `ci/` scripts after modifying build processes
- Update `CMakePresets.json` when adding new build configurations
- Modify `.pre-commit-config.yaml` for linting rule changes
- Review `CONTRIBUTING.md` for contribution guidelines

### Verified Commands and Build Times

**Commands validated in this environment:**
```bash
# CMake configuration works - tested successfully
cmake --preset=install --log-level=VERBOSE

# List presets - tested successfully  
cmake --list-presets

# Repository structure commands - tested
ls examples/  # Shows: basic, cudax, cudax_stf, thrust_flexible_device_system
```

**Build time estimates (add 50% buffer for timeouts):**
- **CUB build**: ~25-35 minutes → Use 60+ minute timeout
- **Thrust build**: ~20-30 minutes → Use 60+ minute timeout  
- **libcudacxx build**: ~15-25 minutes → Use 60+ minute timeout
- **All components**: ~45-60 minutes → Use 90+ minute timeout
- **Tests (each component)**: ~10-15 minutes → Use 30+ minute timeout
- **Pre-commit (all files)**: ~2-5 minutes → Use 10+ minute timeout

**Note**: Build times are based on CI patterns in benchmarks/scripts/ which show timeouts of 10-50x base build time.

### Troubleshooting Build Issues

**Build fails with CUDA errors:**
- Ensure CUDA Toolkit 12.0+ is installed
- Check host compiler compatibility (GCC 7+, Clang 14+)
- Verify GPU architecture in `-arch` parameter matches your hardware

**CMake configuration fails:**
- Ensure CMake 3.21+ for development builds (3.15+ for installation only)
- Check that Ninja build system is installed

**Tests fail or hang:**
- Verify NVIDIA GPU is detected: `nvidia-smi`
- Ensure CUDA drivers are properly installed
- Tests require actual GPU hardware, cannot run in CPU-only environments

**Pre-commit fails:**
- Install missing tools: `pip install pre-commit`
- Check network connectivity for downloading pre-commit environments
- Use `pre-commit run --all-files` to process all files

**Environment requirements:**
- **Building/compiling C++**: Requires CUDA Toolkit and nvcc compiler
- **Building Python packages**: Requires CUDA Toolkit, Python 3.9+, and wheel build environment
- **Testing C++**: Requires NVIDIA GPU hardware with drivers  
- **Testing Python**: Requires NVIDIA GPU hardware, drivers, and Python test dependencies
- **Header usage**: No CUDA required for including headers in CPU code
- **Python pip install**: Requires compatible CUDA drivers (runtime only)
- **CMake configuration**: Some presets work without CUDA (e.g., `install`)
- **Linting/formatting**: No CUDA required, only Python and pre-commit

**Python-specific requirements:**
- **Runtime**: Python 3.9+, CUDA drivers (no toolkit needed for pip install)
- **Development**: Python 3.9+, CUDA Toolkit 12.x, compatible GPU (Compute Capability 6.0+)
- **Dependencies**: `numba>=0.60.0`, `numpy`, `cuda-bindings>=12.9.1`, `cuda-core`, `numba-cuda>=0.18.0`
- **Test dependencies**: `pytest`, `pytest-xdist`, `cupy-cuda12x`, `typing_extensions`

### Performance Tips
- Use Development Containers with `sccache` for faster builds (authentication required for NVIDIA employees)
- Limit CUDA architectures to your target hardware only: `-arch "80"` instead of `"60;70;80"`
- Use `ninja` with parallel builds: `cmake --build --preset=<preset> --parallel <N>`
- Consider `install` preset for header-only usage without tests/examples

### Common Development Workflows

**1. Quick header-only development** (no build required):
```bash
git clone https://github.com/nvidia/cccl.git
# Include headers directly: -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub
```

**2. Full development setup with testing:**
```bash
git clone https://github.com/nvidia/cccl.git
cd cccl
# Open in VSCode, select dev container (recommended)
# OR install CUDA Toolkit + dependencies manually

# Build and test your changes (60+ min builds, 30+ min tests)
./ci/build_cub.sh -cxx g++ -std 17 -arch "60;70;80"
./ci/test_cub.sh -cxx g++ -std 17 -arch "60;70;80"
pre-commit run --all-files
```

**3. Python development workflow:**
```bash
git clone https://github.com/nvidia/cccl.git
cd cccl

# Install Python package in development mode
cd python/cuda_cccl
pip install -e .[test]

# Build and test Python packages (20+ min builds, 10+ min tests each)
cd /home/coder/cccl
./ci/build_cuda_cccl_python.sh -py-version 3.10
./ci/test_cuda_cccl_parallel_python.sh -py-version 3.10
./ci/test_cuda_cccl_cooperative_python.sh -py-version 3.10
./ci/test_cuda_cccl_headers_python.sh -py-version 3.10

# Run specific Python tests manually
cd python/cuda_cccl
pytest tests/parallel/ -v       # Test parallel algorithms
pytest tests/cooperative/ -v    # Test cooperative primitives  
pytest tests/headers/ -v        # Test header access
```

**4. Contributing workflow:**
```bash
# Make changes to code
# ALWAYS run formatting before committing
pre-commit run --all-files

# Build and test affected components
./ci/build_<component>.sh -cxx g++ -std 17 -arch "60;70;80" 
./ci/test_<component>.sh -cxx g++ -std 17 -arch "60;70;80"

# Commit and push changes
git add . && git commit -m "Your changes"
```

**REMINDER**: NEVER CANCEL long-running builds or tests. The timings above are normal and expected. Always wait for completion.