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

#include "dlpack/dlpack.h" // to include before the make_from_dlpack.h
//
#include <cuda/__tma/make_tma_descriptor.h>
#include <cuda/std/array>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include "test_macros.h"

__host__ bool enum_test()
{
  constexpr cuda::tma_oob_fill tma_oob_fill_array[] = {cuda::tma_oob_fill::none, cuda::tma_oob_fill::nan};

  constexpr cuda::tma_l2_fetch_size tma_l2_fetch_size_array[] = {
    cuda::tma_l2_fetch_size::none,
    cuda::tma_l2_fetch_size::bytes64,
    cuda::tma_l2_fetch_size::bytes128,
    cuda::tma_l2_fetch_size::bytes256};

  // constexpr cuda::tma_interleave_layout tma_interleave_layout_array[] = {
  // cuda::tma_interleave_layout::none
  //, cuda::tma_interleave_layout::bytes16//, cuda::tma_interleave_layout::bytes32
  //};

  auto no_interleave = cuda::tma_interleave_layout::none;

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
          {
            tensor.dtype.bits  = 4;
            tensor.dtype.lanes = 16;
            unused(cuda::make_tma_descriptor(tensor, box_sizes, no_interleave, swizzle, l2_fetch_size, oobfill));
            tensor.dtype.lanes = 1;
          }
#endif // _CCCL_CTK_AT_LEAST(12, 8)
        }
        tensor.dtype.code = static_cast<uint8_t>(kDLFloat);
        // FLOAT16, FLOAT32
        for (auto bits : {16, 32})
        {
          exec_make_tma_descriptor(bits, no_interleave, swizzle, l2_fetch_size, oobfill);
        }
        // FLOAT64
        exec_make_tma_descriptor(64, no_interleave, swizzle, l2_fetch_size, oobfill);
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
  NV_IF_TARGET(NV_IS_HOST, (assert(enum_test());));
  return 0;
}
