//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-90
#include <dlpack/dlpack.h> // to include before the make_from_dlpack.h
//
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>
#include <cuda/tma>

#include <cuda_runtime_api.h>

#include "test_macros.h"

constexpr auto no_interleave = cuda::tma_interleave_layout::none;

bool test_ranks()
{
  float* data = nullptr;
  assert(cudaMalloc(&data, 64 * sizeof(float)) == cudaSuccess);
  constexpr int64_t shape_storage[5]   = {128, 128, 128, 128, 128};
  constexpr int64_t strides_storage[5] = {128 * 128 * 128 * 128, 128 * 128 * 128, 128 * 128, 128, 1};
  int box_sizes_storage[5]             = {4, 4, 4, 4, 4};
  cuda::std::span<const int, 5> box_sizes{box_sizes_storage};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 5;
  tensor.dtype.code  = static_cast<uint8_t>(kDLInt);
  tensor.dtype.lanes = 1;
  tensor.dtype.bits  = 32;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;
  // test 5D tensor
  unused(cuda::make_tma_descriptor(tensor, box_sizes));

  tensor.ndim    = 3;
  tensor.strides = const_cast<int64_t*>(strides_storage + 2);
  cuda::std::span<const int, 3> box_sizes_3D{box_sizes_storage};
  // test 3D tensor + interleave layout 16B
  unused(cuda::make_tma_descriptor(tensor, box_sizes_3D, cuda::tma_interleave_layout::bytes16));
  // test 3D tensor + interleave layout 32B + swizzle 32B
  unused(
    cuda::make_tma_descriptor(tensor, box_sizes_3D, cuda::tma_interleave_layout::bytes32, cuda::tma_swizzle::bytes32));
  assert(cudaFree(data) == cudaSuccess);
  return true;
}

bool test_address_alignment()
{
  float* data_32B = nullptr; // aligned to 32B
  assert(cudaMalloc(&data_32B, 64 * sizeof(float)) == cudaSuccess);
  float* data_16B = data_32B + 4; // aligned to 16B

  constexpr int64_t shape_storage[]   = {128, 128, 128, 128};
  constexpr int64_t strides_storage[] = {128 * 128 * 128, 128 * 128, 128, 1};
  DLTensor tensor{};
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 4;
  tensor.dtype.code  = static_cast<uint8_t>(kDLInt);
  tensor.dtype.lanes = 1;
  tensor.dtype.bits  = 32;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;

  int box_sizes_storage[4] = {4, 4, 4, 4};
  cuda::std::span<const int, 4> box_sizes{box_sizes_storage};
  // test 16B alignment
  tensor.data = data_16B;
  for (auto interleave_layout : {cuda::tma_interleave_layout::none, cuda::tma_interleave_layout::bytes16})
  {
    unused(cuda::make_tma_descriptor(tensor, box_sizes, interleave_layout));
  }
  // test 32B alignment
  tensor.data = data_32B;
  unused(
    cuda::make_tma_descriptor(tensor, box_sizes, cuda::tma_interleave_layout::bytes32, cuda::tma_swizzle::bytes32));
  assert(cudaFree(data_32B) == cudaSuccess);
  return true;
}

bool test_sizes()
{
  float* data = nullptr;
  assert(cudaMalloc(&data, 64 * sizeof(float)) == cudaSuccess);
  constexpr int64_t shape_storage[2]   = {16, int64_t{1} << 32};
  constexpr int64_t strides_storage[2] = {int64_t{1} << 32, 1};

  DLTensor tensor{};
  tensor.data             = data;
  tensor.device           = {kDLCUDA, 0};
  tensor.ndim             = 2;
  tensor.dtype.code       = static_cast<uint8_t>(kDLInt);
  tensor.dtype.lanes      = 1;
  tensor.dtype.bits       = 32;
  tensor.shape            = const_cast<int64_t*>(shape_storage);
  tensor.strides          = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset      = 0;
  int box_sizes_storage[] = {16, 16};
  cuda::std::span<const int, 2> box_sizes{box_sizes_storage};
  // test largest tensor size
  unused(cuda::make_tma_descriptor(tensor, box_sizes));
  assert(cudaFree(data) == cudaSuccess);
  return true;
}

bool test_strides()
{
  float* data = nullptr;
  assert(cudaMalloc(&data, 64 * sizeof(float)) == cudaSuccess);
  constexpr int64_t shape_storage[2] = {16, 128};
  int64_t strides_storage[2]         = {(int64_t{1} << 38) - 4, 1};
  int box_sizes_storage[]            = {16, 16};
  cuda::std::span<const int, 2> box_sizes{box_sizes_storage};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 2;
  tensor.dtype.code  = static_cast<uint8_t>(kDLInt);
  tensor.dtype.lanes = 1;
  tensor.dtype.bits  = 32;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;
  // normal case
  unused(cuda::make_tma_descriptor(tensor, box_sizes));
  // stride is 0
  strides_storage[0] = 0;
  unused(cuda::make_tma_descriptor(tensor, box_sizes));
  assert(cudaFree(data) == cudaSuccess);
  return true;
}

bool test_box_sizes()
{
  float* data = nullptr;
  assert(cudaMalloc(&data, 64 * sizeof(float)) == cudaSuccess);
  constexpr int64_t shape_storage[1]   = {256};
  constexpr int64_t strides_storage[1] = {1};
  int box_sizes_storage[1]             = {256};
  cuda::std::span<const int, 1> box_sizes{box_sizes_storage};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDAManaged, 0}; // test also managed memory
  tensor.ndim        = 1;
  tensor.dtype.lanes = 1;
  tensor.dtype.code  = static_cast<uint8_t>(kDLInt);
  tensor.dtype.bits  = 32;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;
  // test largest box size
  unused(cuda::make_tma_descriptor(tensor, box_sizes));
  assert(cudaFree(data) == cudaSuccess);
  return true;
}

bool test_elem_strides()
{
  float* data = nullptr;
  assert(cudaMalloc(&data, 64 * sizeof(float)) == cudaSuccess);
  constexpr int64_t shape_storage[]   = {128, 128, 128};
  constexpr int64_t strides_storage[] = {128 * 128, 128, 1};
  int box_sizes_storage[]             = {16, 16, 16};
  int elem_strides_storage[]          = {8, 8, 8};
  cuda::std::span<const int, 3> box_sizes{box_sizes_storage};
  cuda::std::span<const int, 3> elem_strides{elem_strides_storage};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 3;
  tensor.dtype.lanes = 1;
  tensor.dtype.code  = static_cast<uint8_t>(kDLInt);
  tensor.dtype.bits  = 32;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;
  unused(cuda::make_tma_descriptor(tensor, box_sizes, elem_strides, no_interleave));
  unused(cuda::make_tma_descriptor(tensor, box_sizes, elem_strides, cuda::tma_interleave_layout::bytes16));
  assert(cudaFree(data) == cudaSuccess);
  return true;
}

bool test_enums()
{
  constexpr cuda::tma_oob_fill tma_oob_fill_array[]           = {cuda::tma_oob_fill::none, cuda::tma_oob_fill::nan};
  constexpr cuda::tma_l2_fetch_size tma_l2_fetch_size_array[] = {
    cuda::tma_l2_fetch_size::none,
    cuda::tma_l2_fetch_size::bytes64,
    cuda::tma_l2_fetch_size::bytes128,
    cuda::tma_l2_fetch_size::bytes256};
  constexpr cuda::tma_swizzle tma_swizzle_array[] = {
    cuda::tma_swizzle::none,
    cuda::tma_swizzle::bytes32,
    cuda::tma_swizzle::bytes64,
    cuda::tma_swizzle::bytes128,
#if _CCCL_CTK_AT_LEAST(12, 8)
    cuda::tma_swizzle::bytes128_atom_32B,
    cuda::tma_swizzle::bytes128_atom_32B_flip_8B,
    cuda::tma_swizzle::bytes128_atom_64B
#endif // _CCCL_CTK_AT_LEAST(12, 8)
  };
  constexpr cuda::tma_interleave_layout tma_interleave_layout_array[] = {
    cuda::tma_interleave_layout::none, cuda::tma_interleave_layout::bytes16, cuda::tma_interleave_layout::bytes32};

  int computeCapabilityMajor;
  assert(cudaDeviceGetAttribute(&computeCapabilityMajor, cudaDevAttrComputeCapabilityMajor, 0) == cudaSuccess);

  float* data = nullptr;
  assert(cudaMalloc(&data, 64 * sizeof(float)) == cudaSuccess);
  constexpr int64_t shape_storage[3]   = {128, 128, 128};
  constexpr int64_t strides_storage[3] = {128 * 128, 128, 1};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 3;
  tensor.dtype.lanes = 1;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;

  int box_sizes_storage[3] = {16, 16, 16};
  cuda::std::span<const int, 3> box_sizes{box_sizes_storage};

  auto exec_make_tma_descriptor =
    [&](int bits,
        cuda::tma_interleave_layout layout,
        cuda::tma_swizzle swizzle,
        cuda::tma_l2_fetch_size l2_fetch_size,
        cuda::tma_oob_fill oobfill) {
      tensor.dtype.bits    = static_cast<uint8_t>(bits);
      box_sizes_storage[0] = /*min_align=*/16 * /*bits=*/8 / tensor.dtype.bits;
      box_sizes_storage[1] = /*min_align=*/16 * /*bits=*/8 / tensor.dtype.bits;
      box_sizes_storage[2] = /*min_align=*/16 * /*bits=*/8 / tensor.dtype.bits;
      unused(cuda::make_tma_descriptor(tensor, box_sizes, layout, swizzle, l2_fetch_size, oobfill));
    };

  for (auto oobfill : tma_oob_fill_array)
  {
    for (auto l2_fetch_size : tma_l2_fetch_size_array)
    {
      for (auto swizzle : tma_swizzle_array)
      {
        if (computeCapabilityMajor < 10
            && (swizzle == cuda::tma_swizzle::bytes128_atom_32B
                || swizzle == cuda::tma_swizzle::bytes128_atom_32B_flip_8B
                || swizzle == cuda::tma_swizzle::bytes128_atom_64B))
        {
          continue;
        }
        for (auto layout : tma_interleave_layout_array)
        {
          if (layout == cuda::tma_interleave_layout::bytes32 && swizzle != cuda::tma_swizzle::bytes32)
          {
            continue;
          }
          if (oobfill != cuda::tma_oob_fill::nan)
          {
            tensor.dtype.code = static_cast<uint8_t>(kDLInt);
            // INT32, INT64
            for (auto bits : {32, 64})
            {
              exec_make_tma_descriptor(bits, layout, swizzle, l2_fetch_size, oobfill);
            }
            // UINT8, UINT16, UINT32, UINT64
            tensor.dtype.code = static_cast<uint8_t>(kDLUInt);
            for (auto bits : {8, 16, 32, 64})
            {
              exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
            }
            // TYPES that can be mapped to UINT8
            for (auto code :
                 {kDLBool,
                  kDLFloat8_e3m4,
                  kDLFloat8_e4m3,
                  kDLFloat8_e4m3b11fnuz,
                  kDLFloat8_e4m3fn,
                  kDLFloat8_e4m3fnuz,
                  kDLFloat8_e5m2,
                  kDLFloat8_e5m2fnuz,
                  kDLFloat8_e8m0fnu})
            {
              tensor.dtype.code  = static_cast<uint8_t>(code);
              constexpr int bits = 8;
              exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
            }
#if _CCCL_CTK_AT_LEAST(12, 8)
            // U4 x 16
            if (computeCapabilityMajor >= 10)
            {
              constexpr int bits = 4;
              tensor.dtype.lanes = 16;
              exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
              tensor.dtype.code = static_cast<uint8_t>(kDLFloat4_e2m1fn);
              exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
              tensor.dtype.lanes = 1;
            }
#endif // _CCCL_CTK_AT_LEAST(12, 8)
          }
          tensor.dtype.code = static_cast<uint8_t>(kDLFloat);
          // FLOAT16, FLOAT32, FLOAT64
          for (auto bits : {16, 32, 64})
          {
            exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
          }
          // BFLOAT16
          tensor.dtype.code = static_cast<uint8_t>(kDLBfloat);
          exec_make_tma_descriptor(16, no_interleave, swizzle, l2_fetch_size, oobfill);
        }
      }
    }
  }
  assert(cudaFree(data) == cudaSuccess);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_enums());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_ranks());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_address_alignment());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_sizes());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_strides());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_box_sizes());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_elem_strides());));
  return 0;
}
