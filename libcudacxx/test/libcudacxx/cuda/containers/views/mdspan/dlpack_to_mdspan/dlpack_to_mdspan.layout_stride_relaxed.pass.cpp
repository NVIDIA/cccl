//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// XFAIL: enable-tile
// nvbug6081171: error: "call to non-tile function not supported!"

// UNSUPPORTED: nvrtc

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <nv/target>

#include "test_macros.h"
#include <dlpack/dlpack.h>

template <size_t Rank>
using dlpack_array = cuda::std::array<int64_t, Rank>;

//----------------------------------------------------------------------------------------------------------------------
// Rank-1 with positive stride

bool test_rank1()
{
  cuda::std::array<int, 5> data = {1, 2, 3, 4, 5};
  dlpack_array<1> shape         = {5};
  dlpack_array<1> strides       = {1};
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 1;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 1, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.rank() == 1);
  assert(host_mdspan.extent(0) == 5);
  assert(host_mdspan.mapping().stride(0) == 1);
  assert(host_mdspan.mapping().offset() == 0);
  for (int i = 0; i < 5; ++i)
  {
    assert(host_mdspan(i) == data[i]);
  }
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-2 with padded positive strides

bool test_rank2()
{
  cuda::std::array<int, 8> data = {1, 2, 3, 0, 4, 5, 6, 0};
  dlpack_array<2> shape         = {2, 3};
  dlpack_array<2> strides       = {4, 1};
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<int>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.rank() == 2);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.mapping().stride(0) == 4);
  assert(host_mdspan.mapping().stride(1) == 1);
  assert(host_mdspan.mapping().offset() == 0);
  assert(host_mdspan(0, 0) == 1);
  assert(host_mdspan(0, 1) == 2);
  assert(host_mdspan(0, 2) == 3);
  assert(host_mdspan(1, 0) == 4);
  assert(host_mdspan(1, 1) == 5);
  assert(host_mdspan(1, 2) == 6);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-2 with zero stride (broadcasting)

bool test_rank2_zero_stride()
{
  cuda::std::array<int, 4> data = {10, 20, 30, 40};
  dlpack_array<2> shape         = {3, 4};
  dlpack_array<2> strides       = {0, 1};
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<int>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.rank() == 2);
  assert(host_mdspan.extent(0) == 3);
  assert(host_mdspan.extent(1) == 4);
  assert(host_mdspan.mapping().stride(0) == 0);
  assert(host_mdspan.mapping().stride(1) == 1);
  for (int i = 0; i < 3; ++i)
  {
    assert(host_mdspan(i, 0) == 10);
    assert(host_mdspan(i, 1) == 20);
    assert(host_mdspan(i, 2) == 30);
    assert(host_mdspan(i, 3) == 40);
  }
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-1 with negative stride (reverse iteration)
// byte_offset is mapped to the mapping's element offset, not a pointer adjustment.

bool test_rank1_negative_stride()
{
  cuda::std::array<int, 5> data = {1, 2, 3, 4, 5};
  dlpack_array<1> shape         = {5};
  dlpack_array<1> strides       = {-1};
  DLTensor tensor{};
  tensor.data        = data.data();
  tensor.device      = DLDevice{kDLCPU, 0};
  tensor.ndim        = 1;
  tensor.dtype       = cuda::__data_type_to_dlpack<int>();
  tensor.shape       = shape.data();
  tensor.strides     = strides.data();
  tensor.byte_offset = 4 * sizeof(int);
  auto host_mdspan   = cuda::to_host_mdspan<int, 1, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.rank() == 1);
  assert(host_mdspan.extent(0) == 5);
  assert(host_mdspan.mapping().stride(0) == -1);
  assert(host_mdspan.mapping().offset() == 4);
  assert(host_mdspan.data_handle() == data.data());
  // mapping(i) = 4 + i*(-1) => indices 4, 3, 2, 1, 0
  assert(host_mdspan(0) == 5);
  assert(host_mdspan(1) == 4);
  assert(host_mdspan(2) == 3);
  assert(host_mdspan(3) == 2);
  assert(host_mdspan(4) == 1);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-2 with negative row stride (row reversal)

bool test_rank2_negative_stride()
{
  cuda::std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  dlpack_array<2> strides         = {-3, 1};
  DLTensor tensor{};
  tensor.data        = data.data();
  tensor.device      = DLDevice{kDLCPU, 0};
  tensor.ndim        = 2;
  tensor.dtype       = cuda::__data_type_to_dlpack<float>();
  tensor.shape       = shape.data();
  tensor.strides     = strides.data();
  tensor.byte_offset = 3 * sizeof(float);
  auto host_mdspan   = cuda::to_host_mdspan<float, 2, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.rank() == 2);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.mapping().stride(0) == -3);
  assert(host_mdspan.mapping().stride(1) == 1);
  assert(host_mdspan.mapping().offset() == 3);
  assert(host_mdspan.data_handle() == data.data());
  // mapping(0,j) = 3 + 0*(-3) + j*1 = 3+j  => data[3]=4, data[4]=5, data[5]=6
  // mapping(1,j) = 3 + 1*(-3) + j*1 = j    => data[0]=1, data[1]=2, data[2]=3
  assert(host_mdspan(0, 0) == 4.0f);
  assert(host_mdspan(0, 1) == 5.0f);
  assert(host_mdspan(0, 2) == 6.0f);
  assert(host_mdspan(1, 0) == 1.0f);
  assert(host_mdspan(1, 1) == 2.0f);
  assert(host_mdspan(1, 2) == 3.0f);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-3 with padded positive strides

bool test_rank3()
{
  cuda::std::array<int, 32> data{};
  for (int i = 0; i < 2; ++i)
  {
    for (int j = 0; j < 3; ++j)
    {
      for (int k = 0; k < 4; ++k)
      {
        data[i * 16 + j * 5 + k] = i * 12 + j * 4 + k + 1;
      }
    }
  }
  dlpack_array<3> shape   = {2, 3, 4};
  dlpack_array<3> strides = {16, 5, 1};
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 3;
  tensor.dtype     = cuda::__data_type_to_dlpack<int>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 3, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.rank() == 3);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.extent(2) == 4);
  assert(host_mdspan.mapping().stride(0) == 16);
  assert(host_mdspan.mapping().stride(1) == 5);
  assert(host_mdspan.mapping().stride(2) == 1);
  assert(host_mdspan.mapping().offset() == 0);
  assert(host_mdspan(0, 0, 0) == 1);
  assert(host_mdspan(0, 0, 3) == 4);
  assert(host_mdspan(0, 1, 0) == 5);
  assert(host_mdspan(0, 2, 3) == 12);
  assert(host_mdspan(1, 0, 0) == 13);
  assert(host_mdspan(1, 2, 3) == 24);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Empty tensor (zero extent)

bool test_empty_tensor()
{
  int dummy               = 0;
  dlpack_array<2> shape   = {0, 5};
  dlpack_array<2> strides = {5, 1};
  DLTensor tensor{};
  tensor.data      = &dummy;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.extent(0) == 0);
  assert(host_mdspan.extent(1) == 5);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Return type verification

bool test_return_type()
{
  cuda::std::array<float, 4> data{};
  dlpack_array<1> shape   = {4};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 1;
  tensor.dtype   = cuda::__data_type_to_dlpack<float>();
  tensor.shape   = shape.data();
  tensor.strides = strides.data();
  auto host_ms   = cuda::to_host_mdspan<float, 1, cuda::layout_stride_relaxed>(tensor);

  static_assert(
    cuda::std::is_same_v<decltype(host_ms),
                         cuda::host_mdspan<float, cuda::std::dextents<int64_t, 1>, cuda::layout_stride_relaxed>>);
  assert(host_ms.extent(0) == 4);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Null strides fallback (DLPack < 1.2 only)

bool test_null_strides()
{
  cuda::std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = nullptr;
  auto host_mdspan = cuda::to_host_mdspan<float, 2, cuda::layout_stride_relaxed>(tensor);

  assert(host_mdspan.mapping().stride(0) == 3);
  assert(host_mdspan.mapping().stride(1) == 1);
  assert(host_mdspan(0, 0) == 1.0f);
  assert(host_mdspan(0, 2) == 3.0f);
  assert(host_mdspan(1, 0) == 4.0f);
  assert(host_mdspan(1, 2) == 6.0f);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (assert(test_rank1()); //
     assert(test_rank2());
     // Zero / negative stride tests
     assert(test_rank2_zero_stride());
     assert(test_rank1_negative_stride());
     assert(test_rank2_negative_stride());
     // Rank-3 test
     assert(test_rank3());
     // Edge cases
     assert(test_empty_tensor());
     assert(test_return_type());))
#if !(DLPACK_MAJOR_VERSION > 1 || (DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION >= 2))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_null_strides());))
#endif
  return 0;
}
