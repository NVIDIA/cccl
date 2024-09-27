//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/barrier>

#ifndef TEST_CP_ASYNC_BULK_TENSOR_GENERIC_H_
#define TEST_CP_ASYNC_BULK_TENSOR_GENERIC_H_

#include <cuda/barrier>
#include <cuda/std/array>
#include <cuda/std/utility> // cuda::std::move

#include "test_macros.h" // TEST_NV_DIAG_SUPPRESS

// NVRTC does not support cuda.h (due to import of stdlib.h)
#ifndef TEST_COMPILER_NVRTC
#  include <cudaTypedefs.h> // PFN_cuTensorMapEncodeTiled, CUtensorMap

#  include <cstdio>
#endif // ! TEST_COMPILER_NVRTC

// Suppress warning about barrier in shared memory
TEST_NV_DIAG_SUPPRESS(static_var_with_dynamic_init)

using barrier = cuda::barrier<cuda::thread_scope_block>;
namespace cde = cuda::device::experimental;

/*
 * This header supports the 1d, 2d, ..., 5d test of the TMA PTX wrappers.
 *
 * The functions below help convert Nd coordinates into something useful.
 *
 */

// Compute the total number of elements in a tensor
template <class T, size_t num_dims>
constexpr __host__ __device__ int tensor_len(cuda::std::array<T, num_dims> dims)
{
  T len = 1;
  for (T d : dims)
  {
    len *= d;
  }
  return static_cast<int>(len);
}

// Function to convert:
// a linear index into a shared memory tensor
// into
// a linear index into a global memory tensor.
template <size_t num_dims>
inline __device__ int smem_lin_idx_to_gmem_lin_idx(
  int smem_lin_idx,
  cuda::std::array<uint32_t, num_dims> smem_coord,
  cuda::std::array<uint32_t, num_dims> smem_dims,
  cuda::std::array<uint64_t, num_dims> gmem_dims)
{
  assert(smem_coord.size() == smem_dims.size());
  assert(smem_coord.size() == gmem_dims.size());

  int gmem_lin_idx = 0;
  int gmem_stride  = 1;
  for (int i = 0; i < (int) smem_coord.size(); ++i)
  {
    int smem_i_idx = smem_lin_idx % smem_dims.begin()[i];
    gmem_lin_idx += (smem_coord.begin()[i] + smem_i_idx) * gmem_stride;

    smem_lin_idx /= smem_dims.begin()[i];
    gmem_stride *= gmem_dims.begin()[i];
  }
  return gmem_lin_idx;
}

template <size_t num_dims>
__device__ inline void cp_tensor_global_to_shared(
  CUtensorMap* tensor_map, cuda::std::array<uint32_t, num_dims> indices, void* smem, barrier& bar)
{
  switch (indices.size())
  {
    case 1:
      cde::cp_async_bulk_tensor_1d_global_to_shared(smem, tensor_map, indices[0], bar);
      break;
    case 2:
      cde::cp_async_bulk_tensor_2d_global_to_shared(smem, tensor_map, indices[0], indices[1], bar);
      break;
    case 3:
      cde::cp_async_bulk_tensor_3d_global_to_shared(smem, tensor_map, indices[0], indices[1], indices[2], bar);
      break;
    case 4:
      cde::cp_async_bulk_tensor_4d_global_to_shared(
        smem, tensor_map, indices[0], indices[1], indices[2], indices[3], bar);
      break;
    case 5:
      cde::cp_async_bulk_tensor_5d_global_to_shared(
        smem, tensor_map, indices[0], indices[1], indices[2], indices[3], indices[4], bar);
      break;
    default:
      assert(false && "Wrong number of dimensions.");
  }
}

template <size_t num_dims>
__device__ inline void
cp_tensor_shared_to_global(CUtensorMap* tensor_map, cuda::std::array<uint32_t, num_dims> indices, void* smem)
{
  switch (indices.size())
  {
    case 1:
      cde::cp_async_bulk_tensor_1d_shared_to_global(tensor_map, indices[0], smem);
      break;
    case 2:
      cde::cp_async_bulk_tensor_2d_shared_to_global(tensor_map, indices[0], indices[1], smem);
      break;
    case 3:
      cde::cp_async_bulk_tensor_3d_shared_to_global(tensor_map, indices[0], indices[1], indices[2], smem);
      break;
    case 4:
      cde::cp_async_bulk_tensor_4d_shared_to_global(tensor_map, indices[0], indices[1], indices[2], indices[3], smem);
      break;
    case 5:
      cde::cp_async_bulk_tensor_5d_shared_to_global(
        tensor_map, indices[0], indices[1], indices[2], indices[3], indices[4], smem);
      break;
    default:
      assert(false && "Wrong number of dimensions.");
  }
}

// To define a tensor map in constant memory, we need a type with a size. On
// NVRTC, cuda.h cannot be imported, so we don't have access to the definition
// of CUTensorMap (only to the declaration of CUtensorMap inside cuda/barrier).
// So we use this type instead and reinterpret_cast in the kernel.
struct fake_cutensormap
{
  alignas(64) uint64_t opaque[16];
};
__constant__ fake_cutensormap global_fake_tensor_map;

/*
 * This test has as primary purpose to make sure that the indices in the mapping
 * from C++ to PTX didn't get mixed up.
 *
 * How does it test this?
 *
 * 1. It fills a global memory tensor with linear coordinates 0, 1, ...
 * 2. It loads a tile into shared memory at some coordinate (x, y, ... )
 * 3. It checks that the coordinates that were received in shared memory match the expected.
 * 4. It modifies the coordinates (c = 2 * c + 1)
 * 5. It writes the tile back to global memory
 * 6. It checks that all the values in global are properly modified.
 */
template <size_t smem_len, size_t num_dims>
__device__ void
test(cuda::std::array<uint32_t, num_dims> smem_coord,
     cuda::std::array<uint32_t, num_dims> smem_dims,
     cuda::std::array<uint64_t, num_dims> gmem_dims,
     int* gmem_tensor,
     int gmem_len)
{
  CUtensorMap* global_tensor_map = reinterpret_cast<CUtensorMap*>(&global_fake_tensor_map);

  // SETUP: fill global memory buffer
  for (int i = threadIdx.x; i < gmem_len; i += blockDim.x)
  {
    gmem_tensor[i] = i;
  }
  // Ensure that writes to global memory are visible to others, including
  // those in the async proxy.
  __threadfence();
  __syncthreads();

  // TEST: Add i to buffer[i]
  alignas(128) __shared__ int smem_buffer[smem_len];
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
    cp_tensor_global_to_shared(global_tensor_map, smem_coord, smem_buffer, *bar);
    token = cuda::device::barrier_arrive_tx(*bar, 1, sizeof(smem_buffer));
  }
  else
  {
    token = bar->arrive();
  }
  bar->wait(cuda::std::move(token));

  // Check smem
  for (int i = threadIdx.x; i < static_cast<int>(smem_len); i += blockDim.x)
  {
    int gmem_lin_idx = smem_lin_idx_to_gmem_lin_idx(i, smem_coord, smem_dims, gmem_dims);
    assert(smem_buffer[i] == gmem_lin_idx);
  }

  __syncthreads();

  // Update smem
  for (int i = threadIdx.x; i < static_cast<int>(smem_len); i += blockDim.x)
  {
    smem_buffer[i] = 2 * smem_buffer[i] + 1;
  }
  cde::fence_proxy_async_shared_cta();
  __syncthreads();

  // Write back to global memory:
  if (threadIdx.x == 0)
  {
    cp_tensor_shared_to_global(global_tensor_map, smem_coord, smem_buffer);
    cde::cp_async_bulk_commit_group();
    cde::cp_async_bulk_wait_group_read<0>();
  }
  __threadfence();
  __syncthreads();

  // // TEAR-DOWN: check that global memory is correct
  for (int i = threadIdx.x; i < static_cast<int>(smem_len); i += blockDim.x)
  {
    int gmem_lin_idx = smem_lin_idx_to_gmem_lin_idx(i, smem_coord, smem_dims, gmem_dims);

    assert(gmem_tensor[gmem_lin_idx] == 2 * gmem_lin_idx + 1);
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
#endif

#ifndef TEST_COMPILER_NVRTC
template <typename T, size_t num_dims>
CUtensorMap map_encode(T* tensor_ptr,
                       const cuda::std::array<uint64_t, num_dims>& gmem_dims,
                       const cuda::std::array<uint32_t, num_dims>& smem_dims)
{
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html
  CUtensorMap tensor_map{};

  // The stride is the number of bytes to traverse from the first element of one row to the next.
  // It must be a multiple of 16.
  // cuTensorMapEncodeTiled requies that the stride array is a valid pointer, so we add one superfluous element
  // This is necessary for num_dims == 1
  cuda::std::array<uint64_t, num_dims> stride;
  uint64_t base_stride = sizeof(T);
  for (size_t i = 0; i < stride.size() - 1; ++i)
  {
    base_stride *= gmem_dims[i];
    stride[i] = base_stride;
  }

  // The distance between elements in units of sizeof(element). A stride of 2
  // can be used to load only the real component of a complex-valued tensor, for instance.
  cuda::std::array<uint32_t, num_dims> elem_stride; // = {1, .., 1};
  for (size_t i = 0; i < elem_stride.size(); ++i)
  {
    elem_stride[i] = 1;
  }

  // Get a function pointer to the cuTensorMapEncodeTiled driver API.
  auto cuTensorMapEncodeTiled = get_cuTensorMapEncodeTiled();

  // Create the tensor descriptor.
  CUresult res = cuTensorMapEncodeTiled(
    &tensor_map, // CUtensorMap *tensorMap,
    CUtensorMapDataType::CU_TENSOR_MAP_DATA_TYPE_INT32,
    num_dims, // cuuint32_t tensorRank,
    tensor_ptr, // void *globalAddress,
    gmem_dims.data(), // const cuuint64_t *globalDim,
    stride.data(), // const cuuint64_t *globalStrides,
    smem_dims.data(), // const cuuint32_t *boxDim,
    elem_stride.data(), // const cuuint32_t *elementStrides,
    CUtensorMapInterleave::CU_TENSOR_MAP_INTERLEAVE_NONE,
    CUtensorMapSwizzle::CU_TENSOR_MAP_SWIZZLE_NONE,
    CUtensorMapL2promotion::CU_TENSOR_MAP_L2_PROMOTION_NONE,
    CUtensorMapFloatOOBfill::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  assert(res == CUDA_SUCCESS && "tensormap creation failed.");

  return tensor_map;
}

template <typename T, size_t num_dims>
void init_tensor_map(const T& gmem_tensor_symbol,
                     const cuda::std::array<uint64_t, num_dims>& gmem_dims,
                     const cuda::std::array<uint32_t, num_dims>& smem_dims)
{
  // Get pointer to gmem_tensor to create tensor map.
  int* tensor_ptr = nullptr;
  auto code       = cudaGetSymbolAddress((void**) &tensor_ptr, gmem_tensor_symbol);
  assert(code == cudaSuccess && "Could not get symbol address.");

  // Create tensor map
  CUtensorMap local_tensor_map = map_encode(tensor_ptr, gmem_dims, smem_dims);

  // Copy it to device
  code = cudaMemcpyToSymbol(global_fake_tensor_map, &local_tensor_map, sizeof(CUtensorMap));
  assert(code == cudaSuccess && "Could not copy symbol to device.");
}
#endif // ! TEST_COMPILER_NVRTC

#endif // TEST_CP_ASYNC_BULK_TENSOR_GENERIC_H_
