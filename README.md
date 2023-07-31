# CUDA C++ Core Libraries (CCCL)

Welcome to the CUDA C++ Core Libraries (CCCL) where our mission is to make CUDA C++ more delightful.

This repository unifies three essential CUDA C++ libraries into a single, convenient repository: 

1. [Thrust](https://github.com/nvidia/thrust)
2. [libcudacxx](https://github.com/nvidia/libcudacxx)
3. [CUB](https://github.com/nvidia/cub)

Our goal is to provide CUDA C++ developers with the building blocks they need to make it easier to write safe and efficient code. By bringing these libraries together, we hope to streamline your development process and broaden your ability to leverage the power of CUDA C++.

## Overview

The CUDA C++ Core Libraries (CCCL) grew organically out of the Thrust, CUB, and libcudacxx projects that were developed independently over the years with a similar goal: to provide a high-quality, high-performance, and easy-to-use C++ abstractions for CUDA developers.
Naturally, there was a lot of overlap among the three projects, and we realized that we could better serve the community by unifying them into a single repository.

- **Thrust** is the C++ parallel algorithms library which inspired the introduction of parallel algorithms to the C++ Standard Library. Thrust's high-level interface greatly enhances programmer productivity while enabling performance portability between GPUs and multicore CPUs via configurable backends that allow using multiple parallel programming frameworks (such as CUDA, TBB, and OpenMP).

- **libcudacxx** is the CUDA C++ Standard Library. It provides an implementation of the C++ Standard Library that works in both host and device code. Additionally, it provides abstractions for CUDA-specific hardware features like synchronization primitives, cache control, atomics, and more. 

- **CUB** is a lower-level, CUDA-specific library designed for speed-of-light parallel algorithms across all GPU architectures. In addition to device-wide algorithms, it provides *cooperative algorithms *like block-wide reduce and warp-wide scan, providing CUDA kernel developers with building blocks to create speed-of-light, custom kernels. 

Bringing these libraries together enables us to better deliver the features you need as well as provide a more cohesive experience where everything “just works” out of the box.  
Our long-term goal is to offer a single, unified, modern C++ library that merges and extends the functionality of our existing libraries and more. 
We envisage the CUDA C++ Core Libraries to fill a similar role that the Standard C++ Library fills for Standard C++ – equipping you with general-purpose, speed-of-light tools to focus on solving the problems you care about, not boilerplate.

## Example

This is a simple example demonstrating the use of CCCL functionality from Thrust, CUB, and libcudacxx.

It shows how to use Thrust/CUB/libcudacxx to implement a simple parallel reduction kernel. Each thread block
computes the sum of a subset of the array using `cub::BlockRecuce`. The sum of each block is then reduced 
to a single value using an atomic add via `cuda::atomic_ref` from libcudacxx.

It then shows how the same reduction can be done using Thrust's `reduce` algorithm and compares to the results.

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
  thrust::device_vector<int> data(N, 1);
  thrust::device_vector<int> kernel_result(1);

  int num_blocks = (N + block_size - 1) / block_size;

  // Compute the sum reduction of `data` using a custom kernel
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
  assert(kernel_result[0] == thrust_result]);
  return 0;
}
```

## Getting Started

### CUDA Toolkit 
The easiest way to get started using CCCL is if you already have the [CUDA Toolkit](https://developer.nvidia.com/cuda-toolkit), then you already have CCCL installed on your system and when compiling with `nvcc` you can simply include the desired headers in your code with no additional configuration required.
If compiling with another compiler, you will need to add the include path (e.g., `/usr/local/cuda/include`) to the CCCL headers to your build system.

```cpp
#include <thrust/device_vector.h>
#include <cub/cub.cuh>
#include <cuda/std/atomic>
```

### GitHub

For users that do not have the CUDA Toolkit installed, or if you want to stay on the cutting edge of CCCL development, you can clone the CCCL repository from GitHub and include it in your build system.

```bash
git clone https://github.com/NVIDIA/cccl.git
nvcc -I/path/to/cloned/cccl main.cu -o main # Note, must use -I and not -isystem or else nvcc will use the version of the headers in the CUDA Toolkit
```


#### CMake Integration

CCCL uses [CMake](https://cmake.org/) for all of our build and installation infrastructure, including our tests as well as targets for users to link against their own projects.
As a result, we recommend anyone using CCCL from GitHub to use CMake to integrate CCCL into your project. For a complete example of how to do this using CMake Package Manager see [our example project](examples/example_project). 


## Frequently Asked Questions (FAQs)

**Q: Can I contribute to cccl?**
A: Yes, contributions are welcome! Please refer to the CONTRIBUTING.md for guidelines.

**Q: I'm facing issues or have a question. Who do I contact?**
A: Please [create an issue](https://github.com/NVIDIA/cccl/issues/new/choose) or start a [discussion](https://github.com/NVIDIA/cccl/discussions) and someone from the team will assist you as soon as possible.

**Q: Can I use a newer version of CCCL than I have in my CUDA Toolkit?**
A: Yes! For a complete description of our compatibility guarantees, see #TODO


## CI Pipeline Overview

For a detailed overview of the CI pipeline, see [ci-overview.md](ci-overview.md).