//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "dlpack/dlpack.h" // to include before the make_from_dlpack.h
//
#include <cuda/__tma/make_from_dlpack.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include "test_macros.h"

#if _CCCL_HOST_COMPILATION()

__host__ bool test()
{
  alignas(16) float data[64]{};

  constexpr int64_t shape_storage[2]   = {8, 8};
  constexpr int64_t strides_storage[2] = {1, 8};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 2;
  tensor.dtype.code  = static_cast<uint8_t>(kDLFloat);
  tensor.dtype.bits  = 32;
  tensor.dtype.lanes = 1;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;

  int box_sizes_storage[2]    = {8, 8};
  int elem_strides_storage[2] = {1, 1};

  cuda::std::span<int, 2> box_sizes{box_sizes_storage};
  cuda::std::span<int, 2> elem_strides{elem_strides_storage};

  auto descriptor = cuda::make_tma_descriptor(tensor, box_sizes, elem_strides);

  return true;
}

#endif // _CCCL_HOST_COMPILATION()
/*
TEST_CASE("make_tma_descriptor with bfloat16 and swizzle")
{
  alignas(32) unsigned short data[128]{};

  constexpr int64_t shape_storage[3]   = {2, 4, 16};
  constexpr int64_t strides_storage[3] = {1, 2, 8};

  DLTensor tensor{};
  tensor.data        = data;
  tensor.device      = {kDLCUDA, 0};
  tensor.ndim        = 3;
  tensor.dtype.code  = static_cast<uint8_t>(kDLBfloat);
  tensor.dtype.bits  = 16;
  tensor.dtype.lanes = 1;
  tensor.shape       = const_cast<int64_t*>(shape_storage);
  tensor.strides     = const_cast<int64_t*>(strides_storage);
  tensor.byte_offset = 0;

  int box_sizes_storage[3]    = {16, 4, 2};
  int elem_strides_storage[3] = {1, 1, 1};

  cuda::std::span<int, 3> box_sizes{box_sizes_storage};
  cuda::std::span<int, 3> elem_strides{elem_strides_storage};

  auto descriptor = cuda::make_tma_descriptor(
    tensor,
    box_sizes,
    elem_strides,
    cuda::TmaInterleaveLayout::bytes32,
    cuda::TmaSwizzle::bytes32,
    cuda::TmaL2FetchSize::bytes64,
    cuda::TmaOOBfill::nan);

}*/

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test());));
  return 0;
}
