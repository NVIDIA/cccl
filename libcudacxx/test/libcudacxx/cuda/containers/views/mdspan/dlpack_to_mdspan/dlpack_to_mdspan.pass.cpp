//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
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
// Rank-0 mdspan conversion

bool test_rank0()
{
  float data = 42.0f;
  DLTensor tensor{};
  tensor.data      = &data;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 0;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLFloat, 32, 1};
  auto host_mdspan = cuda::to_host_mdspan<float, 0>(tensor);

  assert(host_mdspan.rank() == 0);
  assert(host_mdspan.size() == 1);
  assert(host_mdspan.data_handle() == &data);
  assert(host_mdspan() == 42.0f);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Empty tensor (zero in one dimension)

bool test_empty_tensor_layout_right_first_dim_zero()
{
  int dummy               = 0; // Non-null but won't be accessed
  dlpack_array<2> shape   = {0, 5};
  dlpack_array<2> strides = {5, 1}; // row-major
  DLTensor tensor{};
  tensor.data      = &dummy;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_right>(tensor);

  assert(host_mdspan.extent(0) == 0);
  assert(host_mdspan.extent(1) == 5);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

bool test_empty_tensor_layout_right_second_dim_zero()
{
  int dummy               = 0; // Non-null but won't be accessed
  dlpack_array<2> shape   = {2, 0};
  dlpack_array<2> strides = {0, 1}; // row-major: stride[0] = 0 * 1 = 0, stride[1] = 1
  DLTensor tensor{};
  tensor.data      = &dummy;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_right>(tensor);

  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 0);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

bool test_empty_tensor_layout_left_first_dim_zero()
{
  int dummy               = 0; // Non-null but won't be accessed
  dlpack_array<2> shape   = {0, 5};
  dlpack_array<2> strides = {1, 0}; // column-major: stride[0] = 1, stride[1] = 0 * 1 = 0
  DLTensor tensor{};
  tensor.data      = &dummy;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_left>(tensor);

  assert(host_mdspan.extent(0) == 0);
  assert(host_mdspan.extent(1) == 5);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

bool test_empty_tensor_layout_stride_explicit_strides()
{
  int dummy               = 0; // Non-null but won't be accessed
  dlpack_array<2> shape   = {0, 5};
  dlpack_array<2> strides = {5, 1}; // explicit strides
  DLTensor tensor{};
  tensor.data      = &dummy;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_stride>(tensor);

  assert(host_mdspan.extent(0) == 0);
  assert(host_mdspan.extent(1) == 5);
  assert(host_mdspan.stride(0) == 5);
  assert(host_mdspan.stride(1) == 1);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

bool test_empty_tensor_layout_stride_null_strides()
{
  int dummy             = 0; // Non-null but won't be accessed
  dlpack_array<2> shape = {0, 5};
  DLTensor tensor{};
  tensor.data      = &dummy;
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape     = shape.data();
  tensor.strides   = nullptr; // null strides (only valid for DLPack < 1.2)
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_stride>(tensor);

  assert(host_mdspan.extent(0) == 0);
  assert(host_mdspan.extent(1) == 5);
  // Should use row-major strides by default
  assert(host_mdspan.stride(0) == 5);
  assert(host_mdspan.stride(1) == 1);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-1 mdspan with layout_right (row-major)

bool test_rank1()
{
  cuda::std::array<int, 5> data = {1, 2, 3, 4, 5};
  dlpack_array<1> shape         = {5};
  dlpack_array<1> strides       = {1};
  DLTensor tensor{};
  tensor.data             = data.data();
  tensor.device           = DLDevice{kDLCPU, 0};
  tensor.ndim             = 1;
  tensor.dtype            = ::DLDataType{::kDLInt, 32, 1};
  tensor.shape            = shape.data();
  tensor.strides          = strides.data();
  auto host_mdspan_right  = cuda::to_host_mdspan<int, 1, cuda::std::layout_right>(tensor);
  auto host_mdspan_left   = cuda::to_host_mdspan<int, 1, cuda::std::layout_left>(tensor);
  auto host_mdspan_stride = cuda::to_host_mdspan<int, 1, cuda::std::layout_stride>(tensor);

  assert(host_mdspan_right.rank() == 1);
  assert(host_mdspan_right.extent(0) == 5);
  assert(host_mdspan_right.stride(0) == 1);
  for (int i = 0; i < 5; ++i)
  {
    assert(host_mdspan_right(i) == data[i]);
  }
  assert(host_mdspan_left.rank() == 1);
  assert(host_mdspan_left.extent(0) == 5);
  assert(host_mdspan_left.stride(0) == 1);
  for (int i = 0; i < 5; ++i)
  {
    assert(host_mdspan_left(i) == data[i]);
  }
  assert(host_mdspan_stride.rank() == 1);
  assert(host_mdspan_stride.extent(0) == 5);
  assert(host_mdspan_stride.stride(0) == 1);
  for (int i = 0; i < 5; ++i)
  {
    assert(host_mdspan_stride(i) == data[i]);
  }
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-2 mdspan with layout_right (row-major)

bool test_rank2_layout_right()
{
  // 2x3 matrix in row-major order
  cuda::std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  dlpack_array<2> strides         = {3, 1}; // row-major
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<float, 2, cuda::std::layout_right>(tensor);

  assert(host_mdspan.rank() == 2);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.stride(0) == 3); // row stride
  assert(host_mdspan.stride(1) == 1); // column stride
  // Check values: row-major layout
  assert(host_mdspan(0, 0) == 1.0f);
  assert(host_mdspan(0, 1) == 2.0f);
  assert(host_mdspan(0, 2) == 3.0f);
  assert(host_mdspan(1, 0) == 4.0f);
  assert(host_mdspan(1, 1) == 5.0f);
  assert(host_mdspan(1, 2) == 6.0f);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-2 mdspan with layout_left (column-major)

bool test_rank2_layout_left()
{
  // 2x3 matrix in column-major order
  cuda::std::array<float, 6> data = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  dlpack_array<2> strides         = {1, 2}; // column-major
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<float, 2, cuda::std::layout_left>(tensor);

  assert(host_mdspan.rank() == 2);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.stride(0) == 1); // row stride
  assert(host_mdspan.stride(1) == 2); // column stride
  // Check values: column-major layout
  assert(host_mdspan(0, 0) == 1.0f);
  assert(host_mdspan(0, 1) == 2.0f);
  assert(host_mdspan(0, 2) == 3.0f);
  assert(host_mdspan(1, 0) == 4.0f);
  assert(host_mdspan(1, 1) == 5.0f);
  assert(host_mdspan(1, 2) == 6.0f);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-2 mdspan with layout_stride (arbitrary strides)

bool test_rank2_layout_stride()
{
  // 2x3 matrix with custom strides (e.g., padded)
  cuda::std::array<int, 8> data = {1, 2, 3, 0, 4, 5, 6, 0}; // Each row padded to 4 elements
  dlpack_array<2> shape         = {2, 3};
  dlpack_array<2> strides       = {4, 1}; // Row stride = 4 (padded), col stride = 1
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<int>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_stride>(tensor);

  assert(host_mdspan.rank() == 2);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.stride(0) == 4);
  assert(host_mdspan.stride(1) == 1);
  assert(host_mdspan(0, 0) == 1);
  assert(host_mdspan(0, 1) == 2);
  assert(host_mdspan(0, 2) == 3);
  assert(host_mdspan(1, 0) == 4);
  assert(host_mdspan(1, 1) == 5);
  assert(host_mdspan(1, 2) == 6);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-3 mdspan with layout_right (row-major)

bool test_rank3_layout_right()
{
  // 2x3x4 tensor in row-major order
  cuda::std::array<float, 24> data = {
    1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  dlpack_array<3> shape   = {2, 3, 4};
  dlpack_array<3> strides = {12, 4, 1}; // row-major: stride[i] = product of shape[i+1:]
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 3;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<float, 3, cuda::std::layout_right>(tensor);

  assert(host_mdspan.rank() == 3);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.extent(2) == 4);
  assert(host_mdspan.stride(0) == 12);
  assert(host_mdspan.stride(1) == 4);
  assert(host_mdspan.stride(2) == 1);
  // Check values
  assert(host_mdspan(0, 0, 0) == 1.0f);
  assert(host_mdspan(0, 0, 3) == 4.0f);
  assert(host_mdspan(0, 1, 0) == 5.0f);
  assert(host_mdspan(0, 2, 3) == 12.0f);
  assert(host_mdspan(1, 0, 0) == 13.0f);
  assert(host_mdspan(1, 2, 3) == 24.0f);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-3 mdspan with layout_left (column-major)

bool test_rank3_layout_left()
{
  // 2x3x4 tensor in column-major order
  // In column-major, elements are stored with the first index varying fastest
  cuda::std::array<float, 24> data = {
    1.0f,  2.0f,  3.0f,  4.0f,  5.0f,  6.0f,  7.0f,  8.0f,  9.0f,  10.0f, 11.0f, 12.0f,
    13.0f, 14.0f, 15.0f, 16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f, 24.0f};
  dlpack_array<3> shape   = {2, 3, 4};
  dlpack_array<3> strides = {1, 2, 6}; // column-major: stride[i] = product of shape[:i]
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 3;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<float, 3, cuda::std::layout_left>(tensor);

  assert(host_mdspan.rank() == 3);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.extent(2) == 4);
  assert(host_mdspan.stride(0) == 1);
  assert(host_mdspan.stride(1) == 2);
  assert(host_mdspan.stride(2) == 6);
  // Check values: element at (i,j,k) is at index i*1 + j*2 + k*6 + 1 (1-indexed value)
  assert(host_mdspan(0, 0, 0) == 1.0f);
  assert(host_mdspan(1, 0, 0) == 2.0f);
  assert(host_mdspan(0, 1, 0) == 3.0f);
  assert(host_mdspan(1, 1, 0) == 4.0f);
  assert(host_mdspan(0, 0, 1) == 7.0f);
  assert(host_mdspan(1, 2, 3) == 24.0f);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Rank-3 mdspan with layout_stride

bool test_rank3_layout_stride()
{
  // 2x3x4 tensor with custom strides (padded)
  cuda::std::array<int, 32> data{}; // Extra space for padding
  // Fill with sequential values at the expected positions
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
  dlpack_array<3> strides = {16, 5, 1}; // Custom strides with padding
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 3;
  tensor.dtype     = cuda::__data_type_to_dlpack<int>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<int, 3, cuda::std::layout_stride>(tensor);

  assert(host_mdspan.rank() == 3);
  assert(host_mdspan.extent(0) == 2);
  assert(host_mdspan.extent(1) == 3);
  assert(host_mdspan.extent(2) == 4);
  assert(host_mdspan.stride(0) == 16);
  assert(host_mdspan.stride(1) == 5);
  assert(host_mdspan.stride(2) == 1);
  // Check values
  assert(host_mdspan(0, 0, 0) == 1);
  assert(host_mdspan(0, 0, 3) == 4);
  assert(host_mdspan(0, 1, 0) == 5);
  assert(host_mdspan(0, 2, 3) == 12);
  assert(host_mdspan(1, 0, 0) == 13);
  assert(host_mdspan(1, 2, 3) == 24);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// const element types

bool test_const_element_type_rank1()
{
  const cuda::std::array<float, 5> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f};
  dlpack_array<1> shape                 = {5};
  dlpack_array<1> strides               = {1};
  DLTensor tensor{};
  tensor.data      = const_cast<float*>(data.data()); // DLPack uses void*, need const_cast
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 1;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = strides.data();
  auto host_mdspan = cuda::to_host_mdspan<const float, 1>(tensor);

  static_assert(cuda::std::is_same_v<typename decltype(host_mdspan)::element_type, const float>);
  assert(host_mdspan.rank() == 1);
  assert(host_mdspan.extent(0) == 5);
  for (int i = 0; i < 5; ++i)
  {
    assert(host_mdspan(i) == data[i]);
  }
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// layout_stride with default (layout_right) strides when strides is nullptr
// Note: This tests the fallback behavior for DLPack < 1.2

bool test_layout_stride_null_strides()
{
  cuda::std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  DLTensor tensor{};
  tensor.data      = data.data();
  tensor.device    = DLDevice{kDLCPU, 0};
  tensor.ndim      = 2;
  tensor.dtype     = cuda::__data_type_to_dlpack<float>();
  tensor.shape     = shape.data();
  tensor.strides   = nullptr; // null strides
  auto host_mdspan = cuda::to_host_mdspan<float, 2, cuda::std::layout_stride>(tensor);
  // Should use row-major strides by default
  assert(host_mdspan.stride(0) == 3);
  assert(host_mdspan.stride(1) == 1);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// byte_offset support

bool test_byte_offset()
{
  cuda::std::array<int, 8> data = {0, 0, 1, 2, 3, 4, 5, 6};
  // Skip first 2 ints (8 bytes)
  dlpack_array<1> shape   = {6};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data        = data.data();
  tensor.device      = DLDevice{kDLCPU, 0};
  tensor.ndim        = 1;
  tensor.dtype       = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape       = shape.data();
  tensor.strides     = strides.data();
  tensor.byte_offset = sizeof(int) * 2;
  auto host_mdspan   = cuda::to_host_mdspan<int, 1>(tensor);

  assert(host_mdspan.extent(0) == 6);
  assert(host_mdspan(0) == 1);
  assert(host_mdspan(5) == 6);
  return true;
}

//----------------------------------------------------------------------------------------------------------------------
// Return type checking

bool test_return_types()
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
  // Check return type of to_host_mdspan
  auto host_ms = cuda::to_host_mdspan<float, 1>(tensor);

  static_assert(
    cuda::std::is_same_v<decltype(host_ms),
                         cuda::host_mdspan<float, cuda::std::dextents<int64_t, 1>, cuda::std::layout_stride>>);
  assert(host_ms.extent(0) == 4);

  auto host_ms_right = cuda::to_host_mdspan<float, 1, cuda::std::layout_right>(tensor);
  static_assert(
    cuda::std::is_same_v<decltype(host_ms_right),
                         cuda::host_mdspan<float, cuda::std::dextents<int64_t, 1>, cuda::std::layout_right>>);
  assert(host_ms_right.extent(0) == 4);
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (assert(test_rank0()); //
                           // Empty tensor tests
     assert(test_empty_tensor_layout_right_first_dim_zero());
     assert(test_empty_tensor_layout_right_second_dim_zero());
     assert(test_empty_tensor_layout_left_first_dim_zero());
     assert(test_empty_tensor_layout_stride_explicit_strides());
     // Rank-1 and Rank-2 tests
     assert(test_rank1());
     assert(test_rank2_layout_right());
     assert(test_rank2_layout_left());
     assert(test_rank2_layout_stride());
     // Rank-3 tests
     assert(test_rank3_layout_right());
     assert(test_rank3_layout_left());
     assert(test_rank3_layout_stride());
     // Const element type tests
     assert(test_const_element_type_rank1());
     // Other tests
     assert(test_byte_offset());
     assert(test_return_types());))
#if !(DLPACK_MAJOR_VERSION > 1 || (DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION >= 2))
  NV_IF_TARGET(NV_IS_HOST,
               (assert(test_layout_stride_null_strides()); //
                assert(test_empty_tensor_layout_stride_null_strides());))
#endif
  return 0;
}
