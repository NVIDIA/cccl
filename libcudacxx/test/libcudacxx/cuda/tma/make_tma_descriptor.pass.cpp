//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc
// UNSfUPPORTED: pre-sm-90

#include <dlpack/dlpack.h> // to include before the make_from_dlpack.h
//
#include <cuda/__tma/make_tma_descriptor.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include "test_macros.h"

constexpr auto no_interleave = cuda::tma_interleave_layout::none;
// constexpr auto no_swizzle       = cuda::tma_swizzle::none;
// constexpr auto no_l2_fetch_size = cuda::tma_l2_fetch_size::none;
// constexpr auto no_oobfill       = cuda::tma_oob_fill::none;

__host__ bool test_ranks()
{
  alignas(128) float data[64]{};
  constexpr int64_t shape_storage[5]   = {128, 128, 128, 128, 128};
  constexpr int64_t strides_storage[5] = {128 * 128 * 128 * 128, 128 * 128 * 128, 128 * 128, 128, 1};

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

  int box_sizes_storage[5] = {4, 4, 4, 4, 4};
  cuda::std::span<const int, 5> box_sizes{box_sizes_storage};

  unused(cuda::make_tma_descriptor(tensor, box_sizes));

  tensor.ndim    = 3;
  tensor.strides = const_cast<int64_t*>(strides_storage + 2);
  cuda::std::span<const int, 3> box_sizes_3D{box_sizes_storage};
  unused(cuda::make_tma_descriptor(tensor, box_sizes_3D, cuda::tma_interleave_layout::bytes16));
  unused(
    cuda::make_tma_descriptor(tensor, box_sizes_3D, cuda::tma_interleave_layout::bytes16, cuda::tma_swizzle::bytes32));
  return true;
}

__host__ bool test_address()
{
  alignas(16) float data_16B[64]{};
  alignas(32) float data_32B[64]{};
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

  tensor.data = data_16B;
  for (auto interleave_layout : {cuda::tma_interleave_layout::none, cuda::tma_interleave_layout::bytes16})
  {
    unused(cuda::make_tma_descriptor(tensor, box_sizes, interleave_layout));
  }
  tensor.data = data_32B;
  unused(
    cuda::make_tma_descriptor(tensor, box_sizes, cuda::tma_interleave_layout::bytes32, cuda::tma_swizzle::bytes32));
  return true;
}

__host__ bool test_sizes()
{
  alignas(128) float data[64]{};
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
  unused(cuda::make_tma_descriptor(tensor, box_sizes));
  return true;
}

__host__ bool test_strides()
{
  alignas(128) float data[64]{};
  constexpr int64_t shape_storage[2] = {16, 128};
  int64_t strides_storage[2]         = {(int64_t{1} << 38) - 4, 1};

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

  int box_sizes_storage[] = {16, 16};
  cuda::std::span<const int, 2> box_sizes{box_sizes_storage};
  unused(cuda::make_tma_descriptor(tensor, box_sizes));

  strides_storage[0] = 0;
  unused(cuda::make_tma_descriptor(tensor, box_sizes));

  tensor.strides = nullptr;
  unused(cuda::make_tma_descriptor(tensor, box_sizes));
  return true;
}

__host__ bool test_box_sizes()
{
  alignas(128) float data[64]{};
  constexpr int64_t shape_storage[2]   = {256, 256};
  constexpr int64_t strides_storage[2] = {256, 1};
  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 2;
  tensor.dtype.lanes = 1;
  tensor.dtype.code  = static_cast<uint8_t>(kDLInt);
  tensor.dtype.bits  = 32;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;

  int box_sizes_storage[2] = {256, 256};
  cuda::std::span<const int, 2> box_sizes{box_sizes_storage};
  unused(cuda::make_tma_descriptor(tensor, box_sizes, cuda::tma_interleave_layout::bytes32));
  return true;
}

__host__ bool test_enums()
{
  constexpr cuda::tma_oob_fill tma_oob_fill_array[] = {cuda::tma_oob_fill::none, cuda::tma_oob_fill::nan};

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

  alignas(128) float data[64]{};
  constexpr int64_t shape_storage[2]   = {128, 128};
  constexpr int64_t strides_storage[2] = {128, 1};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 2;
  tensor.dtype.lanes = 1;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;

  int box_sizes_storage[2] = {16, 16};
  cuda::std::span<const int, 2> box_sizes{box_sizes_storage};

  auto exec_make_tma_descriptor =
    [&](int bits,
        cuda::tma_interleave_layout no_interleave,
        cuda::tma_swizzle swizzle,
        cuda::tma_l2_fetch_size l2_fetch_size,
        cuda::tma_oob_fill oobfill) {
      tensor.dtype.bits    = bits;
      box_sizes_storage[0] = /*min_align=*/16 * /*bits=*/8 / tensor.dtype.bits;
      box_sizes_storage[1] = /*min_align=*/16 * /*bits=*/8 / tensor.dtype.bits;
      unused(cuda::make_tma_descriptor(tensor, box_sizes, no_interleave, swizzle, l2_fetch_size, oobfill));
    };

  for (auto oobfill : tma_oob_fill_array)
  {
    for (auto l2_fetch_size : tma_l2_fetch_size_array)
    {
      for (auto swizzle : tma_swizzle_array)
      {
        if (oobfill != cuda::tma_oob_fill::nan)
        {
          tensor.dtype.code = static_cast<uint8_t>(kDLInt);
          // INT32, INT64
          for (auto bits : {32, 64})
          {
            exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
          }
          // UINT8, UINT16, UINT32, UINT64
          tensor.dtype.code = static_cast<uint8_t>(kDLUInt);
          for (auto bits : {8, 16, 32, 64})
          {
            exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
          }
#if _CCCL_CTK_AT_LEAST(12, 8)
          // U4 x 16
          {
            tensor.dtype.lanes = 16;
            exec_make_tma_descriptor(4, no_interleave, swizzle, l2_fetch_size, oobfill);
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
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_enums());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_ranks());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_address());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_sizes());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_strides());));
  NV_IF_TARGET(NV_IS_HOST, (assert(test_box_sizes());));
  return 0;
}
