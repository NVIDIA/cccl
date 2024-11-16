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

#define VERSION2_COMPARE_GTE(MAJ1, MIN1, MAJ2, MIN2) ((MAJ1 == MAJ2 && MIN1 >= MIN2) || (MAJ1 > MAJ2))

#if defined(__CUDACC__)
#  define CUDACC_COMPARE_GTE(MAJ, MIN) VERSION2_COMPARE_GTE(__CUDACC_VER_MAJOR__, __CUDACC_VER_MINOR__, MAJ, MIN)
#else
#  define CUDACC_COMPARE_GTE(MAJ, MIN) 0
#endif

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

__host__ __device__ int fake_main(int, char**);

int cuda_thread_count = 1;
int cuda_cluster_size = 1;

__global__ void fake_main_kernel(int* ret)
{
  *ret = fake_main(0, nullptr);
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
  // Check if the CUDA driver/runtime are installed and working for sanity.
  cudaError_t err;
  CUDA_CALL(err, cudaDeviceSynchronize());

  list_devices();

  int ret = fake_main(argc, argv);
  if (ret != 0)
  {
    return ret;
  }

  int* cuda_ret = 0;
  CUDA_CALL(err, cudaMalloc(&cuda_ret, sizeof(int)));
// cudaLaunchKernelEx is supported from 11.8 onwards
#if CUDACC_COMPARE_GTE(11, 8)
  if (cuda_cluster_size > 1)
  {
    cudaLaunchAttribute attribute[1];
    attribute[0].id               = cudaLaunchAttributeClusterDimension;
    attribute[0].val.clusterDim.x = cuda_cluster_size; // Cluster size in X-dimension
    attribute[0].val.clusterDim.y = 1;
    attribute[0].val.clusterDim.z = 1;

    cudaLaunchConfig_t config = {
      dim3(cuda_cluster_size), // grid dim
      dim3(cuda_thread_count), // block dim
      0, // dynamic smem bytes
      0, // stream
      attribute, // attributes
      1, // number of attributes
    };
    CUDA_CALL(err, cudaLaunchKernelEx(&config, fake_main_kernel, cuda_ret));
  }
  else
#endif // CTK <= 11.7
  {
    (void) cuda_cluster_size;
    fake_main_kernel<<<1, cuda_thread_count>>>(cuda_ret);
  }
  CUDA_CALL(err, cudaGetLastError());
  CUDA_CALL(err, cudaDeviceSynchronize());
  CUDA_CALL(err, cudaMemcpy(&ret, cuda_ret, sizeof(int), cudaMemcpyDeviceToHost));
  CUDA_CALL(err, cudaFree(cuda_ret));

  return ret;
}

#define main __host__ __device__ fake_main

#endif
