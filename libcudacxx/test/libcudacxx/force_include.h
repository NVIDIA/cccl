//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// We use <stdio.h> instead of <iostream> to avoid relying on the host system's
// C++ standard library.
#ifndef _LIBCUDACXX_FORCE_INCLUDE_H
#define _LIBCUDACXX_FORCE_INCLUDE_H

#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>

void list_devices()
{
  int device_count;
  cudaGetDeviceCount(&device_count);
  printf("CUDA devices found: %d\n", device_count);

  int selected_device;
  cudaGetDevice(&selected_device);

  for (int dev = 0; dev < device_count; ++dev)
  {
    cudaDeviceProp device_prop;
    cudaGetDeviceProperties(&device_prop, dev);

    printf("Device %d: \"%s\", ", dev, device_prop.name);
    if (dev == selected_device)
    {
      printf("Selected, ");
    }
    else
    {
      printf("Unused, ");
    }

    printf("SM%d%d, %zu [bytes]\n", device_prop.major, device_prop.minor, device_prop.totalGlobalMem);
  }
}

#ifdef __CUDACC_TILE__
__tile__
#endif // __CUDACC_TILE__
  __host__ __device__ int
  fake_main(int, char**);

int cuda_thread_count = 1;
int cuda_cluster_size = 1;

__device__ int fake_main_kernel_ret = 0;

#ifdef __CUDACC_TILE__
__tile_global__
#else // ^^^ __CUDACC_TILE__ ^^^ / vvv !__CUDACC_TILE__ vvv
__global__
#endif // !__CUDACC_TILE__
  void
  fake_main_kernel()
{
  int this_ret = fake_main(0, nullptr);

  // There may be multiple threads trying to write the test return value at the same time. We need to make sure they
  // don't overwrite a previous failed result.
  atomicCAS(&fake_main_kernel_ret, 0, this_ret);
}

#define CUDA_CALL(err, ...)                                                                              \
  do                                                                                                     \
  {                                                                                                      \
    err = __VA_ARGS__;                                                                                   \
    if (err != cudaSuccess)                                                                              \
    {                                                                                                    \
      printf("CUDA ERROR, line %d: %s: %s\n", __LINE__, cudaGetErrorName(err), cudaGetErrorString(err)); \
      exit(1);                                                                                           \
    }                                                                                                    \
  } while (false)

int main(int argc, char** argv)
{
  cudaError_t err;
  const cudaStream_t stream = nullptr;

  // Launch the test kernel.
  if (cuda_cluster_size > 1)
  {
    cudaLaunchAttribute attributes[1];
    attributes[0].id               = cudaLaunchAttributeClusterDimension;
    attributes[0].val.clusterDim.x = cuda_cluster_size; // Cluster size in X-dimension
    attributes[0].val.clusterDim.y = 1;
    attributes[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {
      dim3(cuda_cluster_size), // grid dim
      dim3(cuda_thread_count), // block dim
      0, // dynamic smem bytes
      stream, // stream
      attributes, // attributes
      1, // number of attributes
    };
    CUDA_CALL(err, cudaLaunchKernelEx(&config, fake_main_kernel));
  }
  else
  {
    fake_main_kernel<<<1, cuda_thread_count, 0, stream>>>();
    CUDA_CALL(err, cudaGetLastError());
  }

  // Allocate pinned memory for the device run return value.
  int* host_ret;
  CUDA_CALL(err, cudaHostAlloc(&host_ret, sizeof(int), cudaHostAllocDefault));

  // Copy the device result on host.
  CUDA_CALL(err,
            cudaMemcpyFromSymbolAsync(host_ret, fake_main_kernel_ret, sizeof(int), 0, cudaMemcpyDeviceToHost, stream));

  // Execute host testing, while device testing is running.
  printf("Testing on host:\n");
  fflush(stdout);
  int ret = fake_main(argc, argv);
  if (ret != 0)
  {
    printf("Host testing returned failure\n");
    fflush(stdout);
    return ret;
  }

  list_devices();
  printf("Testing on device:\n");
  fflush(stdout);

  // Wait for device testing to end and check the return value.
  CUDA_CALL(err, cudaStreamSynchronize(stream));
  if (*host_ret != 0)
  {
    printf("Device testing returned failure\n");
    return *host_ret;
  }

  // Leak host_ret.
}

#ifdef __CUDACC_TILE__
#  define main(...) __tile__ __host__ __device__ fake_main(__VA_ARGS__)
#else // ^^^ __CUDACC_TILE__ ^^^ / vvv !__CUDACC_TILE__ vvv
#  define main(...) __host__ __device__ fake_main(__VA_ARGS__)
#endif // !__CUDACC_TILE__

#endif
