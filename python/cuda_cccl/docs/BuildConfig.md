# NVRTC Compile Options Support

This document describes how to use `BuildConfig` to pass additional compilation options to NVRTC when building CCCL algorithms.

## Overview

The `BuildConfig` class allows you to specify additional NVRTC compilation flags and include directories when building CCCL algorithms. This is useful for controlling optimization levels, enabling debugging information, or adding custom header paths.

## API

```python
from cuda.compute import BuildConfig

config = BuildConfig(
    extra_compile_flags=["-fmad=true", "-use_fast_math"],
    extra_include_dirs=["/path/to/custom/headers"]
)
```

### Parameters

- `extra_compile_flags` (list[str], optional): Additional compilation flags to pass to NVRTC
- `extra_include_dirs` (list[str], optional): Additional include directories for compilation

### Supported NVRTC Flags

Common NVRTC compilation flags include:

- `-fmad=true/false`: Enable/disable fused multiply-add operations
- `-use_fast_math`: Enable fast math operations (implies several optimization flags)
- `-lineinfo`: Generate line number information for profiling with tools like Nsight Compute
- `-g`: Generate debug information
- `-G`: Generate device debug information
- `-O0`, `-O1`, `-O2`, `-O3`: Set optimization level
- `-ftz=true/false`: Enable/disable flush-to-zero mode for denormal numbers
- `-prec-div=true/false`: Use precise division
- `-prec-sqrt=true/false`: Use precise square root
- `-fma=true/false`: Enable/disable fused multiply-add

For a complete list of supported options, see the [NVRTC documentation](https://docs.nvidia.com/cuda/nvrtc/index.html#supported-compile-options).

## Usage

### With `reduce_into`

```python
import cuda.compute
from cuda.compute import BuildConfig, OpKind
import cupy as cp
import numpy as np

# Setup
d_input = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
d_output = cp.empty(1, dtype=np.float32)
h_init = np.array([0.0], dtype=np.float32)

# Create BuildConfig with compile options
build_config = BuildConfig(extra_compile_flags=["-fmad=true", "-lineinfo"])

# Perform reduction with custom compile options
cuda.compute.reduce_into(
    d_input, d_output, OpKind.PLUS, len(d_input), h_init,
    build_config=build_config
)
```

### With `make_reduce_into`

```python
import cuda.compute
from cuda.compute import BuildConfig, OpKind
import cupy as cp
import numpy as np

# Setup
d_input = cp.array([1, 2, 3, 4, 5], dtype=np.float32)
d_output = cp.empty(1, dtype=np.float32)
h_init = np.array([0.0], dtype=np.float32)

# Create BuildConfig
build_config = BuildConfig(extra_compile_flags=["-use_fast_math"])

# Create a reusable reducer with custom compile options
reducer = cuda.compute.make_reduce_into(
    d_input, d_output, OpKind.PLUS, h_init,
    build_config=build_config
)

# Use the reducer (cached for efficiency)
from cuda.compute._utils.temp_storage_buffer import TempStorageBuffer
tmp_storage_bytes = reducer(None, d_input, d_output, len(d_input), h_init)
tmp_storage = TempStorageBuffer(tmp_storage_bytes)
reducer(tmp_storage, d_input, d_output, len(d_input), h_init)
```

## Implementation Details

### C Layer

The C parallel library (`c/parallel`) already provides the `cccl_build_config` struct:

```c
typedef struct cccl_build_config
{
  const char** extra_compile_flags;
  size_t num_extra_compile_flags;
  const char** extra_include_dirs;
  size_t num_extra_include_dirs;
} cccl_build_config;
```

All algorithms have `_ex` variants that accept a `cccl_build_config*` parameter:
- `cccl_device_reduce_build_ex`
- `cccl_device_scan_build_ex`
- `cccl_device_histogram_build_ex`
- etc.

### Python Layer

The Python bindings expose this functionality through the `BuildConfig` class, which:
1. Wraps the C `cccl_build_config` struct
2. Manages memory for the compile flags and include directories
3. Is passed through the algorithm build pipeline

### Caching

Build results are cached based on:
- Input/output types
- Operation type
- Initial value type
- **BuildConfig identity** (using `id(build_config)`)

This means that using the same `BuildConfig` object will use cached kernels, while different `BuildConfig` objects will trigger new compilations even if they have the same flags.

## Supported Algorithms

Currently, `BuildConfig` is supported for:
- `reduce_into` / `make_reduce_into`

Support for other algorithms can be added following the same pattern.

## Examples

See `python/cuda_cccl/tests/compute/examples/reduction/reduce_with_build_config.py` for a complete working example.

## Testing

Tests for `BuildConfig` are located in `python/cuda_cccl/tests/compute/test_build_config.py`.
