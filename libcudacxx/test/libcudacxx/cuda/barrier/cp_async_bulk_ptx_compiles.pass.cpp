//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: libcpp-has-no-threads
// UNSUPPORTED: pre-sm-90

// <cuda/barrier>

#include <cuda/barrier>
#include <cuda/ptx>

#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

struct CUtensorMap_st;
typedef struct CUtensorMap_st CUtensorMap;

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;

// Kernels below are intended to be compiled, but not run. This is to check if
// all generated PTX is valid.
__global__ void test_bulk_tensor(CUtensorMap* map)
{
  __shared__ int smem;
#if _CCCL_CUDA_COMPILER(CLANG)
  __shared__ char barrier_data[sizeof(barrier)];
  barrier& bar = cuda::std::bit_cast<barrier>(barrier_data);
#else // ^^^ _CCCL_CUDA_COMPILER(CLANG) ^^^ / vvv !_CCCL_CUDA_COMPILER(CLANG)
  __shared__ barrier bar;
#endif // !_CCCL_CUDA_COMPILER(CLANG)
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  const int coords_1d[1]{};
  const int coords_2d[2]{};
  const int coords_3d[3]{};
  const int coords_4d[4]{};
  const int coords_5d[5]{};

  auto native_bar = cuda::device::barrier_native_handle(bar);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, &smem, map, coords_1d, native_bar);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, &smem, map, coords_2d, native_bar);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, &smem, map, coords_3d, native_bar);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, &smem, map, coords_4d, native_bar);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_cluster, cuda::ptx::space_global, &smem, map, coords_5d, native_bar);

  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global, cuda::ptx::space_shared, map, coords_1d, &smem);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global, cuda::ptx::space_shared, map, coords_2d, &smem);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global, cuda::ptx::space_shared, map, coords_3d, &smem);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global, cuda::ptx::space_shared, map, coords_4d, &smem);
  cuda::ptx::cp_async_bulk_tensor(cuda::ptx::space_global, cuda::ptx::space_shared, map, coords_5d, &smem);
}

__global__ void test_bulk(void* gmem)
{
  __shared__ int smem;
  __shared__ char barrier_data[sizeof(barrier)];
  barrier& bar = *reinterpret_cast<barrier*>(&barrier_data);
  if (threadIdx.x == 0)
  {
    init(&bar, blockDim.x);
  }
  __syncthreads();

  cuda::ptx::cp_async_bulk(
    cuda::ptx::space_cluster, cuda::ptx::space_global, &smem, gmem, 1024, cuda::device::barrier_native_handle(bar));
  cuda::ptx::cp_async_bulk(cuda::ptx::space_global, cuda::ptx::space_shared, gmem, &smem, 1024);
}

__global__ void test_fences_async_group(void*)
{
  cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);

  cuda::ptx::cp_async_bulk_commit_group();
  // Wait for up to 8 groups
  cuda::ptx::cp_async_bulk_wait_group_read<0>({});
  cuda::ptx::cp_async_bulk_wait_group_read<1>({});
  cuda::ptx::cp_async_bulk_wait_group_read<2>({});
  cuda::ptx::cp_async_bulk_wait_group_read<3>({});
  cuda::ptx::cp_async_bulk_wait_group_read<4>({});
  cuda::ptx::cp_async_bulk_wait_group_read<5>({});
  cuda::ptx::cp_async_bulk_wait_group_read<6>({});
  cuda::ptx::cp_async_bulk_wait_group_read<7>({});
  cuda::ptx::cp_async_bulk_wait_group_read<8>({});
}

int main(int, char**)
{
  return 0;
}
