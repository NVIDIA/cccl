[![Open in GitHub Codespaces](https://github.com/codespaces/badge.svg)](https://codespaces.new/NVIDIA/cccl?quickstart=1&devcontainer_path=.devcontainer%2Fdevcontainer.json)

|[Contributor Guide](https://github.com/NVIDIA/cccl/blob/main/CONTRIBUTING.md)|[Dev Containers](https://github.com/NVIDIA/cccl/blob/main/.devcontainer/README.md)|[Discord](https://discord.gg/nvidiadeveloper)|[Godbolt](https://godbolt.org/z/x4G73af9a)|[GitHub Project](https://github.com/orgs/NVIDIA/projects/6)|[Documentation](https://nvidia.github.io/cccl)|
|-|-|-|-|-|-|

# CUDA Core Compute Libraries (CCCL)

Welcome to the CUDA Core Compute Libraries (CCCL) where our mission is to make CUDA more delightful.

This repository unifies three essential CUDA C++ libraries into a single, convenient repository:

- [Thrust](thrust) ([former repo](https://github.com/nvidia/thrust))
- [CUB](cub) ([former repo](https://github.com/nvidia/cub))
- [libcudacxx](libcudacxx) ([former repo](https://github.com/nvidia/libcudacxx))

The goal of CCCL is to provide CUDA C++ developers with building blocks that make it easier to write safe and efficient code.
Bringing these libraries together streamlines your development process and broadens your ability to leverage the power of CUDA C++.
For more information about the decision to unify these projects, see the [announcement here](https://github.com/NVIDIA/cccl/discussions/520).

## Overview

The concept for the CUDA Core Compute Libraries (CCCL) grew organically out of the Thrust, CUB, and libcudacxx projects that were developed independently over the years with a similar goal: to provide high-quality, high-performance, and easy-to-use C++ abstractions for CUDA developers.
Naturally, there was a lot of overlap among the three projects, and it became clear the community would be better served by unifying them into a single repository.

- **Thrust** is the C++ parallel algorithms library which inspired the introduction of parallel algorithms to the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs via configurable backends that allow using multiple parallel programming frameworks (such as CUDA, TBB, and OpenMP).

- **CUB** is a lower-level, CUDA-specific library designed for speed-of-light parallel algorithms across all GPU architectures. In addition to device-wide algorithms, it provides *cooperative algorithms* like block-wide reduction and warp-wide scan, providing CUDA kernel developers with building blocks to create speed-of-light, custom kernels.

- **libcudacxx** is the CUDA C++ Standard Library. It provides an implementation of the C++ Standard Library that works in both host and device code. Additionally, it provides abstractions for CUDA-specific hardware features like synchronization primitives, cache control, atomics, and more.

The main goal of CCCL is to fill a similar role that the Standard C++ Library fills for Standard C++: provide general-purpose, speed-of-light tools to CUDA C++ developers, allowing them to focus on solving the problems that matter.
Unifying these projects is the first step towards realizing that goal.

## Example

This is a simple example demonstrating the use of CCCL functionality from Thrust, CUB, and libcudacxx.

It shows how to use Thrust/CUB/libcudacxx to implement a simple parallel reduction kernel.
Each thread block computes the sum of a subset of the array using `cub::BlockReduce`.
The sum of each block is then reduced to a single value using an atomic add via `cuda::atomic_ref` from libcudacxx.

It then shows how the same reduction can be done using Thrust's `reduce` algorithm and compares the results.

[Try it live on Godbolt!](https://godbolt.org/z/aMx4j9f4T)

```cpp
#include <thrust/execution_policy.h>
#include <thrust/device_vector.h>
#include <cub/block/block_reduce.cuh>
#include <cuda/atomic>
#include <cuda/cmath>
#include <cuda/std/span>
#include <cstdio>

template <int block_size>
__global__ void reduce(cuda::std::span<int const> data, cuda::std::span<int> result) {
  using BlockReduce = cub::BlockReduce<int, block_size>;
  __shared__ typename BlockReduce::TempStorage temp_storage;

  int const index = threadIdx.x + blockIdx.x * blockDim.x;
  int sum = 0;
  if (index < data.size()) {
    sum += data[index];
  }
  sum = BlockReduce(temp_storage).Sum(sum);

  if (threadIdx.x == 0) {
    cuda::atomic_ref<int, cuda::thread_scope_device> atomic_result(result.front());
    atomic_result.fetch_add(sum, cuda::memory_order_relaxed);
  }
}

int main() {

  // Allocate and initialize input data
  int const N = 1000;
  thrust::device_vector<int> data(N);
  thrust::fill(data.begin(), data.end(), 1);

  // Allocate output data
  thrust::device_vector<int> kernel_result(1);

  // Compute the sum reduction of `data` using a custom kernel
  constexpr int block_size = 256;
  int const num_blocks = cuda::ceil_div(N, block_size);
  reduce<block_size><<<num_blocks, block_size>>>(cuda::std::span<int const>(thrust::raw_pointer_cast(data.data()), data.size()),
                                                 cuda::std::span<int>(thrust::raw_pointer_cast(kernel_result.data()), 1));

  auto const err = cudaDeviceSynchronize();
  if (err != cudaSuccess) {
    std::cout << "Error: " << cudaGetErrorString(err) << std::endl;
    return -1;
  }

  int const custom_result = kernel_result[0];

  // Compute the same sum reduction using Thrust
  int const thrust_result = thrust::reduce(thrust::device, data.begin(), data.end(), 0);

  // Ensure the two solutions are identical
  std::printf("Custom kernel sum: %d\n", custom_result);
  std::printf("Thrust reduce sum: %d\n", thrust_result);
  assert(kernel_result[0] == thrust_result);
  return 0;
}
```

## Getting Started

### Users

Everything in CCCL is header-only.
Therefore, users need only concern themselves with how they get the header files and how they incorporate them into their build system.

#### CUDA Toolkit
The easiest way to get started using CCCL is via the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit) which includes the CCCL headers.
When you compile with `nvcc`, it automatically adds CCCL headers to your include path so you can simply `#include` any CCCL header in your code with no additional configuration required.

If compiling with another compiler, you will need to update your build system's include search path to point to the CCCL headers in your CTK install (e.g., `/usr/local/cuda/include`).

```cpp
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
```

#### GitHub

Users who want to stay on the cutting edge of CCCL development are encouraged to use CCCL from GitHub.
Using a newer version of CCCL with an older version of the CUDA Toolkit is supported, but not the other way around.
For complete information on compatibility between CCCL and the CUDA Toolkit, see [our platform support](#platform-support).

Everything in CCCL is header-only, so cloning and including it in a simple project is as easy as the following:
```bash
git clone https://github.com/NVIDIA/cccl.git
nvcc -Icccl/thrust -Icccl/libcudacxx/include -Icccl/cub main.cu -o main
```
> **Note**
> Use `-I` and not `-isystem` to avoid collisions with the CCCL headers implicitly included by `nvcc` from the CUDA Toolkit. All CCCL headers use `#pragma system_header` to ensure warnings will still be silenced as if using `-isystem`, see https://github.com/NVIDIA/cccl/issues/527 for more information.

#### Conda

CCCL also provides conda packages of each release via the `conda-forge` channel:

```bash
conda config --add channels conda-forge
conda install cccl
```

This will install the latest CCCL to the conda environment's `$CONDA_PREFIX/include/` and `$CONDA_PREFIX/lib/cmake/` directories.
It is discoverable by CMake via `find_package(CCCL)` and can be used by any compilers in the conda environment.
For more information, see [this introduction to conda-forge](https://conda-forge.org/docs/user/introduction/).

If you want to use the same CCCL version that shipped with a particular CUDA Toolkit, e.g. CUDA 12.4, you can install CCCL with:

```bash
conda config --add channels conda-forge
conda install cuda-cccl cuda-version=12.4
```

The `cuda-cccl` metapackage installs the `cccl` version that shipped with the CUDA Toolkit corresponding to `cuda-version`.
If you wish to update to the latest `cccl` after installing `cuda-cccl`, uninstall `cuda-cccl` before updating `cccl`:

```bash
conda uninstall cuda-cccl
conda install -c conda-forge cccl
```

> **Note**
> There are also conda packages with names like `cuda-cccl_linux-64`.
> Those packages contain the CCCL versions shipped as part of the CUDA Toolkit, but are designed for internal use by the CUDA Toolkit.
> Install `cccl` or `cuda-cccl` instead, for compatibility with conda compilers.
> For more information, see the [cccl conda-forge recipe](https://github.com/conda-forge/cccl-feedstock/blob/main/recipe/meta.yaml).

##### CMake Integration

CCCL uses [CMake](https://cmake.org/) for all build and installation infrastructure, including tests as well as targets to link against in other CMake projects.
Therefore, CMake is the recommended way to integrate CCCL into another project.

For a complete example of how to do this using CMake Package Manager see [our basic example project](examples/basic).

Other build systems should work, but only CMake is tested.
Contributions to simplify integrating CCCL into other build systems are welcome.

### Contributors

Interested in contributing to making CCCL better? Check out our [Contributing Guide](CONTRIBUTING.md) for a comprehensive overview of everything you need to know to set up your development environment, make changes, run tests, and submit a PR.

## Platform Support

**Objective:** This section describes where users can expect CCCL to compile and run successfully.

In general, CCCL should work everywhere the CUDA Toolkit is supported, however, the devil is in the details.
The sections below describe the details of support and testing for different versions of the CUDA Toolkit, host compilers, and C++ dialects.

### CUDA Toolkit (CTK) Compatibility

**Summary:**
- The latest version of CCCL is backward compatible with the current and preceding CTK major version series
- CCCL is never forward compatible with any version of the CTK. Always use the same or newer than what is included with your CTK.
- Minor version CCCL upgrades won't break existing code, but new features may not support all CTK versions

CCCL users are encouraged to capitalize on the latest enhancements and ["live at head"](https://www.youtube.com/watch?v=tISy7EJQPzI) by always using the newest version of CCCL.
For a seamless experience, you can upgrade CCCL independently of the entire CUDA Toolkit.
This is possible because CCCL maintains backward compatibility with the latest patch release of every minor CTK release from both the current and previous major version series.
In some exceptional cases, the minimum supported minor version of the CUDA Toolkit release may need to be newer than the oldest release within its major version series.
For instance, CCCL requires a minimum supported version of 11.1 from the 11.x series due to an unavoidable compiler issue present in CTK 11.0.

When a new major CTK is released, we drop support for the oldest supported major version.

| CCCL Version | Supports CUDA Toolkit Version                  |
|--------------|------------------------------------------------|
| 2.x          | 11.1 - 11.8, 12.x (only latest patch releases) |
| 3.x (Future) | 12.x, 13.x  (only latest patch releases)       |

[Well-behaved code](#compatibility-guidelines) using the latest CCCL should compile and run successfully with any supported CTK version.
Exceptions may occur for new features that depend on new CTK features, so those features would not work on older versions of the CTK.
For example, C++20 support was not added to `nvcc` until CUDA 12.0, so CCCL features that depend on C++20 would not work with CTK 11.x.

Users can integrate a newer version of CCCL into an older CTK, but not the other way around.
This means an older version of CCCL is not compatible with a newer CTK.
In other words, **CCCL is never forward compatible with the CUDA Toolkit.**

The table below summarizes compatibility of the CTK and CCCL:

| CTK Version | Included CCCL Version |    Desired CCCL     | Supported? |                           Notes                           |
|:-----------:|:---------------------:|:--------------------:|:----------:|:--------------------------------------------------------:|
|  CTK `X.Y`  |  CCCL `MAJOR.MINOR`   | CCCL `MAJOR.MINOR+n` |    ✅     |            Some new features might not work              |
|  CTK `X.Y`  |  CCCL `MAJOR.MINOR`   | CCCL `MAJOR+1.MINOR` |    ✅     | Possible breaks; some new features might not be available|
|  CTK `X.Y`  |  CCCL `MAJOR.MINOR`   | CCCL `MAJOR+2.MINOR` |    ❌     |    CCCL supports only two CTK major versions             |
|  CTK `X.Y`  |  CCCL `MAJOR.MINOR`   | CCCL `MAJOR.MINOR-n` |    ❌     |          CCCL isn't forward compatible                   |
|  CTK `X.Y`  |  CCCL `MAJOR.MINOR`   | CCCL `MAJOR-n.MINOR` |    ❌     |          CCCL isn't forward compatible                   |

For more information on CCCL versioning, API/ABI compatibility, and breaking changes see the [Versioning](#versioning) section below.

### Operating Systems

Unless otherwise specified, CCCL supports all the same operating systems as the CUDA Toolkit, which are documented here:
 - [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#system-requirements)
 - [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)

### Host Compilers

Unless otherwise specified, CCCL supports all the same host compilers as the CUDA Toolkit, which are documented here:
- [Linux](https://docs.nvidia.com/cuda/cuda-installation-guide-linux/index.html#host-compiler-support-policy)
- [Windows](https://docs.nvidia.com/cuda/cuda-installation-guide-microsoft-windows/index.html#system-requirements)

In the spirit of "You only support what you test",  see our [CI Overview](https://github.com/NVIDIA/cccl/blob/main/ci-overview.md) for more information on exactly what we test.

### C++ Dialects
- C++11 (Deprecated in Thrust/CUB, to be removed in next major version)
- C++14 (Deprecated in Thrust/CUB, to be removed in next major version)
- C++17
- C++20

### GPU Architectures

Unless otherwise specified, CCCL supports all the same GPU architectures/Compute Capabilities as the CUDA Toolkit, which are documented here: https://docs.nvidia.com/cuda/cuda-c-programming-guide/index.html#compute-capability

Note that some features may only support certain architectures/Compute Capabilities.

### Testing Strategy

CCCL's testing strategy strikes a balance between testing as many configurations as possible and maintaining reasonable CI times.

For CUDA Toolkit versions, testing is done against both the oldest and the newest supported versions.
For instance, if the latest version of the CUDA Toolkit is 12.3, tests are conducted against 11.1 and 12.3.
For each CUDA version, builds are completed against all supported host compilers with all supported C++ dialects.

The testing strategy and matrix are constantly evolving.
The matrix defined in the [`ci/matrix.yaml`](ci/matrix.yaml) file is the definitive source of truth.
For more information about our CI pipeline, see [here](ci-overview.md).

## Versioning

**Objective:** This section describes how CCCL is versioned, API/ABI stability guarantees, and compatibility guidelines to minimize upgrade headaches.

**Summary**
- The entirety of CCCL's API shares a common semantic version across all components
- Only the most recently released version is supported and fixes are not backported to prior releases
- API breaking changes and incrementing CCCL's major version will only coincide with a new major version release of the CUDA Toolkit
- Not all source breaking changes are considered breaking changes of the public API that warrant bumping the major version number
- Do not rely on ABI stability of entities in the `cub::` or `thrust::` namespaces
- ABI breaking changes for symbols in the `cuda::` namespace may happen at any time, but will be reflected by incrementing the ABI version which is embedded in an inline namespace for all `cuda::` symbols. Multiple ABI versions may be supported concurrently.

**Note:** Prior to merging Thrust, CUB, and libcudacxx into this repository, each library was independently versioned according to semantic versioning.
Starting with the 2.1 release, all three libraries synchronized their release versions in their separate repositories.
Moving forward, CCCL will continue to be released under a single [semantic version](https://semver.org/), with 2.2.0 being the first release from the [nvidia/cccl](www.github.com/nvidia/cccl) repository.

### Breaking Change

A Breaking Change is a change to **explicitly supported** functionality between released versions that would require a user to do work in order to upgrade to the newer version.

In the limit, [_any_ change](https://www.hyrumslaw.com/) has the potential to break someone somewhere.
As a result, not all possible source breaking changes are considered Breaking Changes to the public API that warrant bumping the major semantic version.

The sections below describe the details of breaking changes to CCCL's API and ABI.

### Application Programming Interface (API)

CCCL's public API is the entirety of the functionality _intentionally_ exposed to provide the utility of the library.

In other words, CCCL's public API goes beyond just function signatures and includes (but is not limited to):
- The location and names of headers intended for direct inclusion in user code
- The namespaces intended for direct use in user code
- The declarations and/or definitions of functions, classes, and variables located in headers and intended for direct use in user code
- The semantics of functions, classes, and variables intended for direct use in user code

Moreover, CCCL's public API does **not** include any of the following:
- Any symbol prefixed with `_` or `__`
- Any symbol whose name contains `detail` including the `detail::` namespace or a macro
- Any header file contained in a `detail/` directory or sub-directory thereof
- The header files implicitly included by any header part of the public API

In general, the goal is to avoid breaking anything in the public API.
Such changes are made only if they offer users better performance, easier-to-understand APIs, and/or more consistent APIs.

Any breaking change to the public API will require bumping CCCL's major version number.
In keeping with [CUDA Minor Version Compatibility](https://docs.nvidia.com/deploy/cuda-compatibility/#minor-version-compatibility),
API breaking changes and CCCL major version bumps will only occur coinciding with a new major version release of the CUDA Toolkit.

Anything not part of the public API may change at any time without warning.

#### API Versioning

The public API of all CCCL's components share a unified semantic version of `MAJOR.MINOR.PATCH`.

Only the most recently released version is supported.
As a rule, features and bug fixes are not backported to previously released version or branches.

The preferred method for querying the version is to use `CCCL_[MAJOR/MINOR/PATCH_]VERSION` as described below.
For backwards compatibility, the Thrust/CUB/libcudacxxx version definitions are available and will always be consistent with `CCCL_VERSION`.
Note that Thrust/CUB use a `MMMmmmpp` scheme whereas the CCCL and libcudacxx use `MMMmmmppp`.

|                        | CCCL                                   | libcudacxx                                | Thrust                       | CUB                       |
|------------------------|----------------------------------------|-------------------------------------------|------------------------------|---------------------------|
| Header                 | `<cuda/version>`                       | `<cuda/std/version>`                      | `<thrust/version.h>`         | `<cub/version.h>`         |
| Major Version          | `CCCL_MAJOR_VERSION`                   | `_LIBCUDACXX_CUDA_API_VERSION_MAJOR`      | `THRUST_MAJOR_VERSION`       | `CUB_MAJOR_VERSION`       |
| Minor Version          | `CCCL_MINOR_VERSION`                   | `_LIBCUDACXX_CUDA_API_VERSION_MINOR`      | `THRUST_MINOR_VERSION`       | `CUB_MINOR_VERSION`       |
| Patch/Subminor Version | `CCCL_PATCH_VERSION`                   | `_LIBCUDACXX_CUDA_API_VERSION_PATCH`      | `THRUST_SUBMINOR_VERSION`    | `CUB_SUBMINOR_VERSION`    |
| Concatenated Version   | `CCCL_VERSION (MMMmmmppp)`             | `_LIBCUDACXX_CUDA_API_VERSION (MMMmmmppp)`| `THRUST_VERSION (MMMmmmpp)`  | `CUB_VERSION (MMMmmmpp)`  |

### Application Binary Interface (ABI)

The Application Binary Interface (ABI) is a set of rules for:
- How a library's components are represented in machine code
- How those components interact across different translation units

A library's ABI includes, but is not limited to:
- The mangled names of functions and types
- The size and alignment of objects and types
- The semantics of the bytes in the binary representation of an object

An **ABI Breaking Change** is any change that results in a change to the ABI of a function or type in the public API.
For example, adding a new data member to a struct is an ABI Breaking Change as it changes the size of the type.

In CCCL, the guarantees about ABI are as follows:

- Symbols in the `thrust::` and `cub::` namespaces may break ABI at any time without warning.
- The ABI of `thrust::` and `cub::` [symbols includes the CUDA architectures used for compilation](https://nvidia.github.io/cccl/cub/developer_overview.html#symbols-visibility). Therefore, a `thrust::` or `cub::` symbol may have a different ABI if:
    - compiled with different architectures
    - compiled as a CUDA source file (`-x cu`) vs C++ source (`-x cpp`)
- Symbols in the `cuda::` namespace may also break ABI at any time. However, `cuda::` symbols embed an ABI version number that is incremented whenever an ABI break occurs. Multiple ABI versions may be supported concurrently, and therefore users have the option to revert to a prior ABI version. For more information, see [here](libcudacxx/docs/releases/versioning.md).

**Who should care about ABI?**

In general, CCCL users only need to worry about ABI issues when building or using a binary artifact (like a shared library) whose API directly or indirectly includes types provided by CCCL.

For example, consider if `libA.so` was built using CCCL version `X` and its public API includes a function like:
```c++
void foo(cuda::std::optional<int>);
```

If another library, `libB.so`, is compiled using CCCL version `Y` and uses `foo` from `libA.so`, then this can fail if there was an ABI break between version `X` and `Y`.
Unlike with API breaking changes, ABI breaks usually do not require code changes and only require recompiling everything to use the same ABI version.

To learn more about ABI and why it is important, see [What is ABI, and What Should C++ Do About It?](https://wg21.link/P2028R0).

### Compatibility Guidelines

As mentioned above, not all possible source breaking changes constitute a Breaking Change that would require incrementing CCCL's API major version number.

Users are encouraged to adhere to the following guidelines in order to minimize the risk of disruptions from accidentally depending on parts of CCCL that are not part of the public API:

- Do not add any declarations to, or specialize any template from, the `thrust::`, `cub::`, `nv::`, or `cuda::` namespaces unless an exception is noted for a specific symbol, e.g., specializing `cuda::std::iterator_traits`
    - **Rationale**: This would cause conflicts if a symbol or specialization is added with the same name.
- Do not take the address of any API in the `thrust::`, `cub::`, `cuda::`, or `nv::` namespaces.
    - **Rationale**: This would prevent adding overloads of these APIs.
- Do not forward declare any API in the `thrust::`, `cub::`, `cuda::`, or `nv::` namespaces.
    - **Rationale**: This would prevent adding overloads of these APIs.
- Do not directly reference any symbol prefixed with `_`, `__`, or with `detail` anywhere in its name including a `detail::` namespace or macro
     - **Rationale**: These symbols are for internal use only and may change at any time without warning.
- Include what you use. For every CCCL symbol that you use, directly `#include` the header file that declares that symbol. In other words, do not rely on headers implicitly included by other headers.
     - **Rationale**: Internal includes may change at any time.

Portions of this section were inspired by [Abseil's Compatibility Guidelines](https://abseil.io/about/compatibility).

## Deprecation Policy

We will do our best to notify users prior to making any breaking changes to the public API, ABI, or modifying the supported platforms and compilers.

As appropriate, deprecations will come in the form of programmatic warnings which can be disabled.

The deprecation period will depend on the impact of the change, but will usually last at least 2 minor version releases.


## Mapping to CTK Versions

Coming soon!

## CI Pipeline Overview

For a detailed overview of the CI pipeline, see [ci-overview.md](ci-overview.md).

## Related Projects

Projects that are related to CCCL's mission to make CUDA more delightful:
- [cuCollections](https://github.com/NVIDIA/cuCollections) - GPU accelerated data structures like hash tables
- [NVBench](https://github.com/NVIDIA/nvbench) - Benchmarking library tailored for CUDA applications
- [stdexec](https://github.com/nvidia/stdexec) - Reference implementation for Senders asynchronous programming model

## Projects Using CCCL

Does your project use CCCL? [Open a PR to add your project to this list!](https://github.com/NVIDIA/cccl/edit/main/README.md)

- [AmgX](https://github.com/NVIDIA/AMGX) - Multi-grid linear solver library
- [ColossalAI](https://github.com/hpcaitech/ColossalAI) - Tools for writing distributed deep learning models
- [cuDF](https://github.com/rapidsai/cudf) - Algorithms and file readers for ETL data analytics
- [cuGraph](https://github.com/rapidsai/cugraph) - Algorithms for graph analytics
- [cuML](https://github.com/rapidsai/cuml) - Machine learning algorithms and primitives
- [CuPy](https://cupy.dev) - NumPy & SciPy for GPU
- [cuSOLVER](https://developer.nvidia.com/cusolver) - Dense and sparse linear solvers
- [cuSpatial](https://github.com/rapidsai/cuspatial) - Algorithms for geospatial operations
- [GooFit](https://github.com/GooFit/GooFit) - Library for maximum-likelihood fits
- [HeavyDB](https://github.com/heavyai/heavydb) - SQL database engine
- [HOOMD](https://github.com/glotzerlab/hoomd-blue) - Monte Carlo and molecular dynamics simulations
- [HugeCTR](https://github.com/NVIDIA-Merlin/HugeCTR) - GPU-accelerated recommender framework
- [Hydra](https://github.com/MultithreadCorner/Hydra) - High-energy Physics Data Analysis
- [Hypre](https://github.com/hypre-space/hypre) - Multigrid linear solvers
- [LightSeq](https://github.com/bytedance/lightseq) - Training and inference for sequence processing and generation
- [MatX](https://github.com/NVIDIA/matx) - Numerical computing library using expression templates to provide efficient, Python-like syntax
- [PyTorch](https://github.com/pytorch/pytorch) - Tensor and neural network computations
- [Qiskit](https://github.com/Qiskit/qiskit-aer) - High performance simulator for quantum circuits
- [QUDA](https://github.com/lattice/quda) - Lattice quantum chromodynamics (QCD) computations
- [RAFT](https://github.com/rapidsai/raft) - Algorithms and primitives for machine learning
- [TensorFlow](https://github.com/tensorflow/tensorflow) - End-to-end platform for machine learning
- [TensorRT](https://github.com/NVIDIA/TensorRT) - Deep learning inference
- [tsne-cuda](https://github.com/CannyLab/tsne-cuda) - Stochastic Neighborhood Embedding library
- [Visualization Toolkit (VTK)](https://gitlab.kitware.com/vtk/vtk) - Rendering and visualization library
- [XGBoost](https://github.com/dmlc/xgboost) - Gradient boosting machine learning algorithms
