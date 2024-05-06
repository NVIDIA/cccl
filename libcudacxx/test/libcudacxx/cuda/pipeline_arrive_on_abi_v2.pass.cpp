//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: pre-sm-70
// UNSUPPORTED: clang && (!nvcc)

#define _LIBCUDACXX_CUDA_ABI_VERSION 2

#include <cuda_pipeline.h>

#include <cuda/barrier>

#include "cuda_space_selector.h"
#include "test_macros.h"

// TODO: Remove pointless comparison suppression when compiler fixes short-circuiting
TEST_NV_DIAG_SUPPRESS(186) // pointless comparison of unsigned integer with zero

TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)
TEST_NV_DIAG_SUPPRESS(declared_but_not_referenced)

using nvcuda::experimental::pipeline;

__host__ __device__ bool operator==(int2 a, int2 b)
{
  return a.x == b.x && a.y == b.y;
}

__host__ __device__ bool operator==(int4 a, int4 b)
{
  return a.x == b.x && a.y == b.y && a.z == b.z && a.w == b.w;
}

template <typename T>
__device__ void arrive_on_device_copy(
  T* global_array, T* shared_array, unsigned copy_count, cuda::barrier<cuda::thread_scope_block>* barrier)
{
  pipeline pipe;

  for (unsigned i = 0; i < copy_count; ++i)
  {
    const int idx = i * blockDim.x + threadIdx.x;
    memcpy_async(shared_array[idx], global_array[idx], pipe);
  }

  pipe.arrive_on(*barrier);

  barrier->arrive_and_wait();

  for (unsigned i = 0; i < copy_count; ++i)
  {
    // Rotate thread indexes for check
    const int tid = (threadIdx.x + (blockDim.x / 2)) % blockDim.x;
    const int idx = i * blockDim.x + tid;
    assert(global_array[idx] == shared_array[idx]);
  }
}

template <typename T>
__device__ bool arrive_on_test(char* global_buffer, size_t buffer_size)
{
  assert(blockDim.y == 1 && blockDim.z == 1);

  extern __shared__ char shared_buffer[];
  shared_memory_selector<cuda::barrier<cuda::thread_scope_block>, constructor_initializer> sel;
  SHARED cuda::barrier<cuda::thread_scope_block>* barrier;
  barrier = sel.construct();

  if (threadIdx.x == 0)
  {
    init(barrier, blockDim.x);
  }
  __syncthreads();

  T* global_array               = reinterpret_cast<T*>(global_buffer);
  T* shared_array               = reinterpret_cast<T*>(shared_buffer);
  const size_t array_size       = buffer_size / sizeof(T);
  const unsigned max_copy_count = array_size / blockDim.x;
  const unsigned min_copy_count = 1;

  arrive_on_device_copy<T>(global_array, shared_array, min_copy_count, barrier);
  for (unsigned copy_count = 2; copy_count <= max_copy_count; copy_count += copy_count - 1)
  {
    arrive_on_device_copy<T>(global_array, shared_array, copy_count, barrier);
  }
  arrive_on_device_copy<T>(global_array, shared_array, max_copy_count, barrier);

  return true;
}

#ifdef TEST_COMPILER_NVRTC
__device__ void arrive_on_nvrtc(size_t buffer_size)
{
  auto scramble_buffer = [](char* buffer, size_t buffer_size, size_t base_value) {
    if (threadIdx.x == 0)
    {
      for (int i = 0; i < buffer_size; ++i)
      {
        buffer[i] = (base_value + i) % CHAR_MAX;
      }
    }
    __syncthreads();
  };
  __shared__ char* global_buffer;
  if (threadIdx.x == 0)
  {
    global_buffer = new char[buffer_size];
  }
  __syncthreads();

  scramble_buffer(global_buffer, buffer_size, 0);
  assert(arrive_on_test<char>(global_buffer, buffer_size));
  scramble_buffer(global_buffer, buffer_size, 43);
  assert(arrive_on_test<short>(global_buffer, buffer_size));
  scramble_buffer(global_buffer, buffer_size, 17);
  assert(arrive_on_test<int>(global_buffer, buffer_size));
  scramble_buffer(global_buffer, buffer_size, 13);
  assert(arrive_on_test<int2>(global_buffer, buffer_size));
  scramble_buffer(global_buffer, buffer_size, 127);
  assert(arrive_on_test<int4>(global_buffer, buffer_size));
}
#else
template <typename T>
__global__ void arrive_on_kernel(char* global_buffer, size_t buffer_size, bool* success)
{
  *success = arrive_on_test<T>(global_buffer, buffer_size);
}

template <typename T>
void arrive_on_launch(
  char* global_buffer, size_t buffer_size, bool* success_dptr, volatile bool* success_hptr, unsigned block_size)
{
  *success_hptr = false;
  printf("arrive_on_kernel<%2zu><<<%u, %2u, %zu>>> ", sizeof(T), 1, block_size, buffer_size);
  arrive_on_kernel<T><<<1, block_size, buffer_size>>>(global_buffer, buffer_size, success_dptr);
  cudaError_t result;
  CUDA_CALL(result, cudaDeviceSynchronize());
  CUDA_CALL(result, cudaGetLastError());
  printf("%s\n", *success_hptr ? "[ OK ]" : "[FAIL]");
  assert(*success_hptr);
}

void arrive_on_run(
  char* global_buffer, size_t buffer_size, bool* success_dptr, volatile bool* success_hptr, unsigned block_size)
{
  arrive_on_launch<char>(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  arrive_on_launch<short>(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  arrive_on_launch<int>(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  arrive_on_launch<int2>(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  arrive_on_launch<int4>(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
}

void arrive_on()
{
  volatile bool* success_hptr;
  bool* success_dptr;
  int lanes_per_warp;
  int max_shmem;
  size_t buffer_size;
  char* global_buffer;
  char* host_buffer;

  cudaError_t result;
  CUDA_CALL(result, cudaHostAlloc(&success_hptr, sizeof(*success_hptr), cudaHostAllocMapped));
  CUDA_CALL(result, cudaHostGetDevicePointer(&success_dptr, (void*) success_hptr, 0));
  CUDA_CALL(result, cudaDeviceGetAttribute(&lanes_per_warp, cudaDevAttrWarpSize, 0));
  CUDA_CALL(result, cudaDeviceGetAttribute(&max_shmem, cudaDevAttrMaxSharedMemoryPerBlock, 0));
  cudaFuncAttributes attrib;
  CUDA_CALL(result, cudaFuncGetAttributes(&attrib, arrive_on_kernel<int4>));
  buffer_size = max_shmem - attrib.sharedSizeBytes;
  CUDA_CALL(result, cudaMalloc(&global_buffer, buffer_size));
  host_buffer = new char[buffer_size];

  for (unsigned i = 0; i < buffer_size; ++i)
  {
    host_buffer[i] = rand() % CHAR_MAX;
  }
  cudaMemcpy(global_buffer, host_buffer, buffer_size, cudaMemcpyHostToDevice);

  // 1 Thread
  {
    const unsigned block_size = 1;
    arrive_on_run(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  }

  // 1 Warp
  {
    const unsigned block_size = 1 * lanes_per_warp;
    arrive_on_run(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  }

  // 1 CTA
  {
    const unsigned block_size = 2 * lanes_per_warp;
    arrive_on_run(global_buffer, buffer_size, success_dptr, success_hptr, block_size);
  }

  delete[] host_buffer;
  CUDA_CALL(result, cudaFree(global_buffer));
  CUDA_CALL(result, cudaFreeHost((void*) success_hptr));
}
#endif

int main(int argc, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, arrive_on();)

#ifdef TEST_COMPILER_NVRTC
  int cuda_thread_count     = 64;
  int cuda_block_shmem_size = 40000;

  arrive_on_nvrtc(cuda_block_shmem_size - sizeof(cuda::barrier<cuda::thread_scope_block>) - sizeof(char*));
#endif // TEST_COMPILER_NVRTC

  return 0;
}
