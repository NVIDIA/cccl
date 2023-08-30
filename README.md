# CUDA C++ Core Libraries (CCCL)

// Add logo

Welcome to the CUDA C++ Core Libraries (CCCL) where our mission is to make CUDA C++ more delightful.

This repository unifies three essential CUDA C++ libraries into a single, convenient repository: 

1. [Thrust](https://github.com/nvidia/thrust)
2. [libcudacxx](https://github.com/nvidia/libcudacxx)
3. [CUB](https://github.com/nvidia/cub)

Our goal is to provide CUDA C++ developers with the building blocks they need to make it easier to write safe and efficient code. By bringing these libraries together, we hope to streamline your development process and broaden your ability to leverage the power of CUDA C++.
For more information about our decision to unify these projects, see our announcement here: (TODO)

## Overview

The concept for the CUDA C++ Core Libraries (CCCL) grew organically out of the Thrust, CUB, and libcudacxx projects that were developed independently over the years with a similar goal: to provide a high-quality, high-performance, and easy-to-use C++ abstractions for CUDA developers.
Naturally, there was a lot of overlap among the three projects, and we realized that we could better serve the community by unifying them into a single repository.

- **Thrust** is the C++ parallel algorithms library which inspired the introduction of parallel algorithms to the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs via configurable backends that allow using multiple parallel programming frameworks (such as CUDA, TBB, and OpenMP).

- **libcudacxx** is the CUDA C++ Standard Library. It provides an implementation of the C++ Standard Library that works in both host and device code. Additionally, it provides abstractions for CUDA-specific hardware features like synchronization primitives, cache control, atomics, and more. 

- **CUB** is a lower-level, CUDA-specific library designed for speed-of-light parallel algorithms across all GPU architectures. In addition to device-wide algorithms, it provides *cooperative algorithms* like block-wide reduce and warp-wide scan, providing CUDA kernel developers with building blocks to create speed-of-light, custom kernels. 

Bringing these libraries together enables us to better deliver the features you need as well as provide a more cohesive experience where everything “just works” out of the box.  
Our long-term goal is to offer a single, unified, modern C++ library that merges and extends the functionality of our existing libraries and more. 
We envision the CUDA C++ Core Libraries to fill a similar role that the Standard C++ Library fills for Standard C++ – equipping you with general-purpose, speed-of-light tools to focus on solving the problems you care about, not boilerplate.

## Example

This is a simple example demonstrating the use of CCCL functionality from Thrust, CUB, and libcudacxx.

It shows how to use Thrust/CUB/libcudacxx to implement a simple parallel reduction kernel. Each thread block
computes the sum of a subset of the array using `cub::BlockRecuce`. The sum of each block is then reduced 
to a single value using an atomic add via `cuda::atomic_ref` from libcudacxx.

It then shows how the same reduction can be done using Thrust's `reduce` algorithm and compares the results.

```cpp
#include <thrust/device_vector.h>
#include <cstdio>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>

constexpr int block_size = 256;

__global__ void reduce(int const* data, int* result, std::size_t N) {
  using BlockReduce = cub::BlockReduce<int, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int index = threadIdx.x + blockIdx.x * blockDim.x;
  int sum = 0;
  if (index < N) {
    sum += data[index];
  }
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(*result);
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}

int main() {
  std::size_t N = 1000;
  // Allocate and initialize input data
  thrust::device_vector<int> data(N);
  thrust::fill(data.begin(), data.end(), 1);
  // Allocate output data
  thrust::device_vector<int> kernel_result(1);

  // Compute the sum reduction of `data` using a custom kernel
  int num_blocks = (N + block_size - 1) / block_size;
  reduce<<<num_blocks, block_size>>>(thrust::raw_pointer_cast(data.data()),
                                     thrust::raw_pointer_cast(kernel_result.data()),
                                     N);

  auto err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  // Compute the same sum reduction using Thrust
  int thrust_result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);

  // Ensure the two solutions are identical
  std::printf("Custom Kernel Sum: %d\n", kernel_result[0]);
  std::printf("Thrust reduce Sum: %d\n", thrust_result);
  assert(kernel_result[0] == thrust_result);
  return 0;
}
```

## Getting Started

### Users

Generally speaking, because CCCL is a header-only library, users need only concern themselves with how they get the header files and how they incorporate them into their build system. 
Anyone interested in using CCCL in their CUDA C++ application can get started by referring to the information below. 

#### CUDA Toolkit 
The easiest way to get started using CCCL is if you already have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), then you already have the CCCL headers installed on your system.
When you compile with `nvcc` you can simply include the desired headers in your code with no additional configuration required.
If compiling with another compiler, you will need to add the include path (e.g., `/usr/local/cuda/include`) to the CCCL headers to your build system.

```cpp
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
```

#### GitHub

For users that want to stay on the cutting edge of CCCL development, we actively support and encourage users to use CCCL from GitHub. 
We support using a newer version of CCCL with an older version of the CUDA Toolkit, but not the other way around.
For complete information on compatibility between CCCL and the CUDA Toolkit, see [our platform support](#platform-support).

Everything in CCCL is header-only, so cloning and including it in a simple project is as easy as the following:
```bash
git clone https://github.com/NVIDIA/cccl.git
# Note: You need to use -I and not -isystem in order to ensure you're using the cloned headers and not the ones from the CUDA Toolkit
nvcc -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub main.cu -o main 
```

##### CMake Integration

For more complex projects, we recommend using CMake to integrate CCCL into your project.

CCCL uses [CMake](https://cmake.org/) for all of our build and installation infrastructure, including our tests as well as targets for users to link against their own projects.
As a result, we recommend anyone using CCCL from GitHub to use CMake to integrate CCCL into your project. 
For a complete example of how to do this using CMake Package Manager see [our example project](examples/example_project). 

Other build systems should work, but we only test CMake.
We welcome contributions that would simplify the process of integrating CCCL into other build systems.

### Contributors

## Platform Support

**Objective:** This section describes where users can expect CCCL to compile and run successfully. 

In general, CCCL should work everywhere the CUDA Toolkit is supported, however, the devil is in the details. 
The sections below describe the details of our support for different versions of the CUDA Toolkit, host compilers, and C++ dialects.
Furthermore, we describe our testing strategy to be confident that CCCL works everywhere it should. 

### CUDA Toolkit (CTK) Compatibility 

The CCCL team encourages users to ["live at head"](https://www.youtube.com/watch?v=tISy7EJQPzI) and always use the newest version of CCCL from GitHub to take advantage of the latest improvements.

We want to enable users to upgrade the version of CCCL they are using without upgrading their entire CUDA Toolkit. 
Therefore, CCCL is backwards compatible with the latest patch release of every minor CTK release within two major version series: current and preceding. 
When a new major CTK is released, we drop support for the oldest version.

| CCCL Version  | Supports CUDA Toolkit Version  |
|---------------|-------------------------------|
| 2.x           | 11.1 - 11.8, 12.x (only latest patch releases) |
| 3.x (Future)  | 12.x, 13.x  (only latest patch releases)       |

[Well-behaved code](#compatibility-guidelines) using the latest CCCL should compile and run successfully with any supported CTK version. 
Exceptions may occur for new features that depend on new CTK features, so those features would not work on older versions of the CTK. 
For example, C++20 support was not added to `nvcc` until CUDA 12.0, so CCCL features that depend on C++20 would not work with CTK 11.x. 

We want to enable users to bring a newer version of CCCL to an older CTK, but not vice versa.
This means you cannot bring an older version of CCCL to a newer CTK.
In other words, **CCCL is never forwards compatible with the CUDA Toolkit.** 

The table below summarizes compatibility of the CTK and CCCL:

| CTK Version | Included CCCL Version |   Trying to use...  | Supported? |                            Notes                           |
|:-----------:|:---------------------:|:-------------------:|:----------:|:----------------------------------------------------------:|
|   CTK `X.Y`   |    CCCL `MAJOR.MINOR`   | CCCL `MAJOR.MINOR+n`  |     Yes    |               Some new features may not work               |
|   CTK `X.Y`   |    CCCL `MAJOR.MINOR`   | CCCL `MAJOR+1.MINOR`  |     Yes    | Possible breaking changes.  Some new features may not work |
|   CTK `X.Y`   |    CCCL `MAJOR.MINOR`   | CCCL `MAJOR+2.MINOR`  |     No     |           CCCL supports only 2 CTK major verions           |
|   CTK `X.Y`   |    CCCL `MAJOR.MINOR`   | CCCL `MAJOR.MINOR-n`  |     No     |               CCCL is not forwards compatible              |
|   CTK `X.Y`   |    CCCL `MAJOR.MINOR`   | CCCL `MAJOR-n.MINOR`  |     No     |               CCCL is not forwards compatible              |


For more information on CCCL versioning, API/ABI compatibility, and breaking changes see the [Versioning](#versioning) section below.

**TL;DR:**
- The latest version of CCCL is backwards compatible with the current and preceding CTK major version series
- CCCL is never forwards compatible with any version of the CTK. Always use the same or newer than what is included with your CTK. 
- Minor version CCCL upgrades won't break existing code, but new features may not support all CTK versions

### Operating Systems

Unless otherwise specified, CCCL supports all the same operating systems as the CUDA toolkit, which are documented here:
 - [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)
 - [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)

### Host Compilers

Unless otherwise specified, CCCL supports all the same host compilers as the CUDA Toolkit, which are documented here:
- [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy)
- [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)

### C++ Dialects
- C++11 (Deprecated in Thrust/CUB, to be removed in next major version)
- C++14 (Deprecated in Thrust/CUB, to be removed in next major version)
- C++17 
- C++20

### GPU Architectures

Unless otherwise specified, CCCL supports all the same GPU architectures/Compute Capabilities as the CUDA Toolkit, which are documented here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability

Note that some features may only support certain architectures/Compute Capabilities. 

### Testing Strategy

Our testing strategy that balances the need to test as many configurations as possible with reasonable CI times. 

For CUDA Toolkit versions, we test against the oldest and the newest version that we support. 
For example, if the latest version of the CUDA Toolkit is 12.3, we test against 11.1 and 12.3.
For each CUDA version, we build against all of the supported host compilers with all of the supported C++ dialects.

Our testing strategy and matrix are constantly evolving. 
The matrix defined in the [`ci/matrix.yaml`](ci/matrix.yaml) file is our single source of truth and you can review it to see exactly what we test against.
For more information about our CI pipeline, see [here](ci-overview.md).

## Versioning

**Objective:** This section describes how CCCL is versioned, API/ABI stability guarantees, and compatibility guideliness to minimize upgrade headaches. 

**Summary**
- The entirety of CCCL's API shares a common semantic version across all components
- We support only the most recently released version (we do not backport fixes to prior releases) 
- API breaking changes and incrementing CCCL's major version will only coincide with a new major version release of the CUDA Toolkit 
- Not all source breaking changes are considered breaking changes of the public API that warrant bumping the major version number
- Do not rely on ABI stability of enities in the `cub::` or `thrust::` namespaces
- ABI breaking changes for symbols in the `cuda::` namespace may happen at any time, but will be reflected by incrementing the ABI version which is embedded in an inline namespace for all `cuda::` symbols. Multiple ABI versions may be supported concurrently. 

**Note:** Prior to merging Thrust, CUB, and libcudacxx into this repository, each library was independently versioned according to semantic versioning. 
Starting with the 2.1 release, all three libraries synchronized their release versions in their separate repositories. 
Moving forward, CCCL will continue to be released under a single [semantic version](https://semver.org/), with 2.2.0 being the first release from the [nvidia/cccl](www.github.com/nvidia/cccl) repository. 

### Breaking Change 

A Breaking Change is a change to **explicitly supported** functionality between released versions that would require a user to do work in order to upgrade to the newer version.

In the limit, [_any_ change](https://www.hyrumslaw.com/) has the potential to break someone somewhere. 
As a result, we do not consider all possible source breaking changes as Breaking Changes to the public API that warrant bumping the major semantic version. 

This section describes the details of CCCL's API, ABI, and what we consider to be Breaking Changes.

### Application Programming Interface (API)

CCCL's public API is the entirety of the functionality _intentionally_ exposed to provide the utility of the library.

In other words, CCCL's public API goes beyond just function signatures and includes:
- The location and names of headers intended for direct inclusion in user code
- The namespaces intended for direct use in user code
- The declarations and/or definitions of functions, classes, and variables located in headers and intended for direct use in user code
- The semantics of functions, classes, and variables intended for direct use in user code

Moreover, CCCL's public API does **not** include any of the following: 
- Any symbol prefixed with `_` or `__`
- Any symbol whose name contains `detail` including the `detail::` namespace or a macro
- Any header file contained in a `detail/` directory or sub-directory thereof
- The header files implicitly included by any header part of the public API

In general, we strive to not break public APIs and only do so when such changes benefits our users with better performance, easier-to-understand APIs, and/or more consistent APIs. 

Any breaking change to the public API will require bumping CCCL's major version number. 
In keeping with [CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility), 
API breaking changes and CCCL major version bumps will only occur coinciding with a new major version release of the CUDA Toolkit.

Anything not part of the public API may change at any time without warning.

#### API Versioning

The entirety of CCCL's public API across all components shares a common semantic version of `MAJOR.MINOR.PATCH`.
For historical reasons, these versions are encoded separately in each of Thrust/CUB/libcudacxx as follows:


|                  | libcudacxx                              | Thrust              | CUB                | Incremented when? |
|------------------|-----------------------------------------|---------------------|--------------------|--------------------------------------------------------|
| Header           | `<cuda/std/version>`                      | `<thrust/version.h>`  | `<cub/version.h>`    | -                                                      |
| Major Version    | `_LIBCUDACXX_CUDA_API_VERSION_MAJOR`     | `THRUST_MAJOR_VERSION` | `CUB_MAJOR_VERSION`  | Public API breaking changes (only at new CTK major release)       |
| Minor Version    | `_LIBCUDACXX_CUDA_API_VERSION_MINOR`      | `THRUST_MINOR_VERSION` | `CUB_MINOR_VERSION`  | Non-breaking feature additions             |
| Patch/Subminor Version | `_LIBCUDACXX_CUDA_API_VERSION_PATCH`      | `THRUST_SUBMINOR_VERSION` | `CUB_SUBMINOR_VERSION`  | Minor changes not covered by major/minor versions     |
| Concatenated Version | `_LIBCUDACXX_CUDA_API_VERSION` (`MMMmmmppp`)         | `THRUST_VERSION` (`MMMmmmpp`)      | CUB_VERSION (`MMMmmmpp`)        | -                                                      |


### Application Binary Interface (ABI)

The Application Binary Interface (ABI) of a software library is the convention for how:
- The library's entities are represented in machine code, and
- Library entities built in one translation unit interact with entities from another.

A library's ABI includes, but is not limited to:
- The mangled names of functions.
- The mangled names of types, including instantiations of class templates.
- The number of bytes (sizeof) and the alignment of objects and types.
- The semantics of the bytes in the binary representation of an object.
- The register-level calling conventions governing parameter passing and function invocation.

To learn more about ABI and why it is important, see [What is ABI, and What Should C++ Do About It?](https://wg21.link/P2028R0).

### Compatibility Guidelines

As mentioned above, we do not consider all possible source breaking changes to warrant incrementing CCCL's API major version number. 

It is impractical to provide an exhaustive list of all possible breaking changes, but here are some general guidelines that will minimize the risk of breakages. 
Portions of this section were inspired by [Abseil's Compatibility Guidelines](https://abseil.io/about/compatibility).

- Do not add any declarations to the `thrust::`, `cub::`, `nv::`, or `cuda::` namespaces unless an exception is noted for a specific symbol, e.g., specializing a type trait.
    - **Rationale**: This would cause symbol conflicts if we added a symbol with the same name. 
- Do not take the address of any API in the `thrust::`, `cub::`, `cuda::`, or `nv::` namespaces. 
    - **Rationale**: This would prevent us from adding overloads of these APIs.
- Do not forward declare any API in the `thrust::`, `cub::`, `cuda::`, or `nv::` namespaces.
    - **Rationale**: This would prevent us from adding overloads of these APIs.
- Do not directly reference any symbol prefixed with `_`, `__`, or with `detail` anywhere in its name including a `detail::` namespace or macro
     - **Rationale**: These symbols are for internal use only and may change at any time without warning. 
- Include what you use. For every CCCL symbol that you use, directly `#include` the header file that declares that symbol. In other words, do not rely on headers implicitly included by other headers.
     - **Rationale**: Internal includes may change at any time.


## Deprecation Policy

We will do our best to notify users prior to making any breaking changes to the public API, ABI, or modifying the platforms and compilers we support.

As appropriate, deprecations will come in the form of programmatic warnings which can be disabled.

The deprecation period will depend on the impact of the change, but will usually last at least 2 minor version releases.


## Frequently Asked Questions (FAQs)

**Q: Can I contribute to cccl?**
A: Yes, contributions are welcome! Please refer to the CONTRIBUTING.md for guidelines.

**Q: I'm facing issues or have a question. Who do I contact?**
A: Please [create an issue](https://github.com/NVIDIA/cccl/issues/new/choose) or start a [discussion](https://github.com/NVIDIA/cccl/discussions) and someone from the team will assist you as soon as possible.

**Q: Can I use a newer version of CCCL than I have in my CUDA Toolkit?**
A: Yes! For a complete description of our compatibility guarantees, see #TODO

## Mapping to CTK Versions

// Links to old CCCL mapping tables
// Add new CCCL version to a new table


## CI Pipeline Overview

For a detailed overview of the CI pipeline, see [ci-overview.md](ci-overview.md).