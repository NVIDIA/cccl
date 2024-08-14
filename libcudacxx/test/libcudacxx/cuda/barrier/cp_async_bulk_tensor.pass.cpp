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
// UNSUPPORTED: nvcc-11
// UNSUPPORTED: nvrtc
// NVRTC_SKIP_KERNEL_RUN // This will have effect once PR 433 is merged (line above should be removed.)

// <cuda/barrier>

#include <cuda/barrier>
#include <cuda/std/utility> // cuda::std::move

#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// NVRTC does not support cuda.h (due to import of stdlib.h)
#ifndef TEST_COMPILER_NVRTC
#  include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap
#endif // !TEST_COMPILER_NVRTC

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

constexpr size_t GMEM_WIDTH  = 1024; // Width of tensor (in # elements)
constexpr size_t GMEM_HEIGHT = 1024; // Height of tensor (in # elements)
constexpr size_t gmem_len    = GMEM_WIDTH * GMEM_HEIGHT;

constexpr int SMEM_WIDTH  = 32; // Width of shared memory buffer (in # elements)
constexpr int SMEM_HEIGHT = 8; // Height of shared memory buffer (in # elements)

static constexpr int buf_len = SMEM_HEIGHT * SMEM_WIDTH;
__device__ int gmem_tensor[gmem_len];

// We need a type with a size. On NVRTC, cuda.h cannot be imported, so we don't
// have access to the definition of CUTensorMap (only to the declaration of CUtensorMap inside
// cuda/barrier). So we use this type instead and reinterpret_cast in the
// kernel.
struct fake_cutensormap
{
  alignas(64) uint64_t opaque[16];
};
__constant__ fake_cutensormap global_fake_tensor_map;

__device__ void test(int base_i, int base_j)
{
  CUtensorMap* global_tensor_map = reinterpret_cast<CUtensorMap*>(&global_fake_tensor_map);

  // SETUP: fill global memory buffer
  for (int i = threadIdx.x; i < static_cast<int>(gmem_len); i += blockDim.x)
  {
    gmem_tensor[i] = i;
  }
  // Ensure that writes to global memory are visible to others, including
  // those in the async proxy.
  __threadfence();
  __syncthreads();

  // TEST: Add i to buffer[i]
  alignas(128) __shared__ int smem_buffer[buf_len];
  __shared__ barrier* bar;
  if (threadIdx.x == 0)
  {
    init(bar, blockDim.x);
  }
  __syncthreads();

  // Load data:
  uint64_t token;
  if (threadIdx.x == 0)
  {
    // Fastest moving coordinate first.
    cde::cp_async_bulk_tensor_2d_global_to_shared(smem_buffer, global_tensor_map, base_j, base_i, *bar);
    token = cuda::device::barrier_arrive_tx(*bar, 1, sizeof(smem_buffer));
  }
  else
  {
    token = bar->arrive();
  }
  bar->wait(cuda::std::move(token));

  // Check smem
  for (int i = 0; i < SMEM_HEIGHT; ++i)
  {
    for (int j = 0; j < SMEM_HEIGHT; ++j)
    {
      const int gmem_lin_idx = (base_i + i) * GMEM_WIDTH + base_j + j;
      const int smem_lin_idx = i * SMEM_WIDTH + j;

      assert(smem_buffer[smem_lin_idx] == gmem_lin_idx);
    }
  }

  __syncthreads();

  // Update smem
  for (int i = threadIdx.x; i < buf_len; i += blockDim.x)
  {
    smem_buffer[i] = 2 * smem_buffer[i] + 1;
  }
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  // Write back to global memory:
  if (threadIdx.x == 0)
  {
    cde::cp_async_bulk_tensor_2d_shared_to_global(global_tensor_map, base_j, base_i, smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  __threadfence();
  __syncthreads();

  // TEAR-DOWN: check that global memory is correct
  for (int i = 0; i < SMEM_HEIGHT; ++i)
  {
    for (int j = 0; j < SMEM_HEIGHT; ++j)
    {
      int gmem_lin_idx = (base_i + i) * GMEM_WIDTH + base_j + j;

      assert(gmem_tensor[gmem_lin_idx] == 2 * gmem_lin_idx + 1);
    }
  }
  __syncthreads();
}

#ifndef TEST_COMPILER_NVRTC
PFN_cuTensorMapEncodeTiled get_cuTensorMapEncodeTiled()
{
  void* driver_ptr = nullptr;
  cudaDriverEntryPointQueryResult driver_status;
  auto code = cudaGetDriverEntryPoint("cuTensorMapEncodeTiled", &driver_ptr, cudaEnableDefault, &driver_status);
  assert(code == cudaSuccess && "Could not get driver API");
  return reinterpret_cast<PFN_cuTensorMapEncodeTiled>(driver_ptr);
}
#endif // ! TEST_COMPILER_NVRTC

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      // Required by concurrent_agents_launch to know how many we're launching
      cuda_thread_count = 512;

      int* tensor_ptr = nullptr;
      auto code       = cudaGetSymbolAddress((void**) &tensor_ptr, gmem_tensor);
      assert(code == cudaSuccess && "getsymboladdress failed.");

      // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
      CUtensorMap local_tensor_map{};
      // rank is the number of dimensions of the array.
      constexpr uint32_t rank = 2;
      uint64_t size[rank]     = {GMEM_WIDTH, GMEM_HEIGHT};
      // The stride is the number of bytes to traverse from the first element of one row to the next.
      // It must be a multiple of 16.
      uint64_t stride[rank - 1] = {GMEM_WIDTH * sizeof(int)};
      // The box_size is the size of the shared memory buffer that is used as the
      // destination of a TMA transfer.
      uint32_t box_size[rank] = {SMEM_WIDTH, SMEM_HEIGHT};
      // The distance between elements in units of sizeof(element). A stride of 2
      // can be used to load only the real component of a complex-valued tensor, for instance.
      uint32_t elem_stride[rank] = {1, 1};

      // Get a function pointer to the cuTensorMapEncodeTiled driver API.
      auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

      // Create the tensor descriptor.
      CUresult res = cuTensorMapEncodeTiled(
        &local_tensor_map, // CUtensorMap *tensorMap,
        CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
        rank, // cuuint32_t tensorRank,
        tensor_ptr, // void *globalAddress,
        size, // const cuuint64_t *globalDim,
        stride, // const cuuint64_t *globalStrides,
        box_size, // const cuuint32_t *boxDim,
        elem_stride, // const cuuint32_t *elementStrides,
        CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
        CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
        CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
        CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

      assert(res == CUDA_SUCCESS && "tensormap creation failed.");
      code = cudaMemcpyToSymbol(global_fake_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
      assert(code == cudaSuccess && "memcpytosymbol failed.");));

  NV_DISPATCH_TARGET(NV_IS_DEVICE, (test(0, 0); test(4, 0); test(4, 4);));
  return 0;
}
