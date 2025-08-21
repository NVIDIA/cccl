# CUDA Core Compute Libraries (CCCL) Development Guide

Always reference these instructions first and fallback to search or bash commands only when you encounter unexpected information that does not match the info here.

CCCL is a collection of three main header-only CUDA C++ libraries: **libcudacxx** (CUDA C++ Standard Library), **CUB** (block-level primitives), and **Thrust** (high-level parallel algorithms), plus experimental **cudax** features. The repository uses CMake with Ninja generator and provides standardized presets for consistent builds.

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
```

**Available compiler/standard combinations:**
- Host compilers: `g++`, `clang++` (version 7+ for GCC, 14+ for Clang)
- C++ standards: `17`, `20` 
- CUDA architectures: `"60;70;80"` (default), `"70;75;80-virtual"`, `"70-real;80"` etc.

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

# Test Python packages - NEVER CANCEL, takes up to 10 minutes
./ci/test_cuda_cccl_parallel_python.sh
./ci/test_cuda_cccl_headers_python.sh

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

4. **Always run formatting before committing:**
   ```bash
   pre-commit run --all-files  # 2-5 min
   ```

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
├── cudax/               # Experimental CUDA features
├── python/cuda_cccl/    # Python bindings and parallel algorithms
├── ci/                  # Build and test scripts
├── .devcontainer/       # Development container configs
├── examples/            # Usage examples
└── CMakePresets.json    # Standardized build configurations
```

**Python components:**
- `python/cuda_cccl/` - Python packages for CCCL parallel algorithms
- Test with: `./ci/test_cuda_cccl_parallel_python.sh`, `./ci/test_cuda_cccl_headers_python.sh`
```
cccl/
├── libcudacxx/          # CUDA C++ Standard Library headers
├── cub/                 # CUB block-level primitives  
├── thrust/              # Thrust parallel algorithms
├── cudax/               # Experimental CUDA features
├── ci/                  # Build and test scripts
├── .devcontainer/       # Development container configs
├── examples/            # Usage examples
└── CMakePresets.json    # Standardized build configurations
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
- **Building/compiling**: Requires CUDA Toolkit and nvcc compiler
- **Testing**: Requires NVIDIA GPU hardware with drivers  
- **Header usage**: No CUDA required for including headers in CPU code
- **CMake configuration**: Some presets work without CUDA (e.g., `install`)
- **Linting/formatting**: No CUDA required, only Python and pre-commit

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

**3. Contributing workflow:**
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