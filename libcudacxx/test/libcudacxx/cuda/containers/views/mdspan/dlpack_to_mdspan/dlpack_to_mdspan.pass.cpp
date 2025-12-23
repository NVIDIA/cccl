//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <dlpack/dlpack.h>

#include <cuda/mdspan>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>

#include <nv/target>

#include "test_macros.h"

template <size_t Rank>
using dlpack_array = cuda::std::array<int64_t, Rank>;

//==============================================================================
// Test: Rank-0 mdspan conversion
//==============================================================================

bool test_rank0()
{
  float data = 42.0f;
  DLTensor tensor{};
  tensor.data   = &data;
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim   = 0;
  tensor.dtype  = DLDataType{DLDataTypeCode::kDLFloat, 32, 1};

  auto host_mdspan = cuda::to_host_mdspan<float, 0>(tensor);

  assert(host_mdspan.rank() == 0);
  assert(host_mdspan.size() == 1);
  assert(host_mdspan.data_handle() == &data);
  assert(host_mdspan() == 42.0f);
  return true;
}

//==============================================================================
// Test: Empty tensor (zero in one dimension)
//==============================================================================

bool test_empty_tensor()
{
  int dummy               = 0; // Non-null but won't be accessed
  dlpack_array<2> shape   = {0, 5};
  dlpack_array<2> strides = {5, 1}; // row-major
  DLTensor tensor{};
  tensor.data    = &dummy;
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  auto host_mdspan = cuda::to_host_mdspan<int, 2, cuda::std::layout_right>(tensor);

  assert(host_mdspan.extent(0) == 0);
  assert(host_mdspan.extent(1) == 5);
  assert(host_mdspan.size() == 0);
  assert(host_mdspan.empty());
  return true;
}

//==============================================================================
// Test: Rank-1 mdspan with layout_right (row-major)
//==============================================================================

bool test_rank1()
{
  cuda::std::array<int, 5> data = {1, 2, 3, 4, 5};
  dlpack_array<1> shape         = {5};
  dlpack_array<1> strides       = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 1;
  tensor.dtype   = ::DLDataType{::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

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

//==============================================================================
// Test: Rank-2 mdspan with layout_right (row-major)
//==============================================================================

bool test_rank2_layout_right()
{
  // 2x3 matrix in row-major order
  cuda::std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  dlpack_array<2> strides         = {3, 1}; // row-major
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = cuda::__data_type_to_dlpack<float>();
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

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

//==============================================================================
// Test: Rank-2 mdspan with layout_left (column-major)
//==============================================================================

bool test_rank2_layout_left()
{
  // 2x3 matrix in column-major order
  cuda::std::array<float, 6> data = {1.0f, 4.0f, 2.0f, 5.0f, 3.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  dlpack_array<2> strides         = {1, 2}; // column-major
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = cuda::__data_type_to_dlpack<float>();
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

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

//==============================================================================
// Test: Rank-2 mdspan with layout_stride (arbitrary strides)
//==============================================================================

bool test_rank2_layout_stride()
{
  // 2x3 matrix with custom strides (e.g., padded)
  cuda::std::array<int, 8> data = {1, 2, 3, 0, 4, 5, 6, 0}; // Each row padded to 4 elements
  dlpack_array<2> shape         = {2, 3};
  dlpack_array<2> strides       = {4, 1}; // Row stride = 4 (padded), col stride = 1
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = cuda::__data_type_to_dlpack<int>();
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

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

//==============================================================================
// Test: layout_stride with default (layout_right) strides when strides is nullptr
// Note: This tests the fallback behavior for DLPack < 1.2
//==============================================================================
#if !(DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION >= 2)
bool test_layout_stride_null_strides()
{
  cuda::std::array<float, 6> data = {1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f};
  dlpack_array<2> shape           = {2, 3};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = cuda::__data_type_to_dlpack<float>();
  tensor.shape   = shape.data();
  tensor.strides = nullptr; // null strides

  auto host_mdspan = cuda::to_host_mdspan<float, 2, cuda::std::layout_stride>(tensor);

  // Should use row-major strides by default
  assert(host_mdspan.stride(0) == 3);
  assert(host_mdspan.stride(1) == 1);
  return true;
}
#endif

//==============================================================================
// Test: byte_offset support
//==============================================================================

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

  auto host_mdspan = cuda::to_host_mdspan<int, 1>(tensor);

  assert(host_mdspan.extent(0) == 6);
  assert(host_mdspan(0) == 1);
  assert(host_mdspan(5) == 6);
  return true;
}

//==============================================================================
// Exception tests
//==============================================================================

void test_exception_wrong_rank()
{
  cuda::std::array<int, 6> data{};
  dlpack_array<2> shape   = {2, 3};
  dlpack_array<2> strides = {3, 1}; // row-major
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    // Try to convert rank-2 tensor to rank-1 mdspan
    unused(cuda::to_host_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_wrong_dtype()
{
  cuda::std::array<int, 4> data{};
  dlpack_array<1> shape   = {4};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1}; // dtype is int
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    // Try to convert int tensor to float mdspan
    unused(cuda::to_host_mdspan<float, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_null_data()
{
  dlpack_array<1> shape   = {4};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = nullptr;
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_null_shape()
{
  cuda::std::array<int, 4> data{};
  DLTensor tensor{};
  tensor.data   = data.data();
  tensor.device = DLDevice{kDLCPU, 0};
  tensor.ndim   = 1;
  tensor.dtype  = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape  = nullptr; // null shape

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_negative_shape()
{
  cuda::std::array<int, 4> data{};
  dlpack_array<1> shape   = {-3}; // negative shape
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_wrong_device_type_host()
{
  cuda::std::array<int, 4> data{};
  dlpack_array<1> shape   = {4};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{::kDLCUDA, 0}; // CUDA device, not CPU
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_wrong_device_type_device()
{
  cuda::std::array<int, 4> data{};
  dlpack_array<1> shape   = {4};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0}; // CPU device, not CUDA
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_device_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_wrong_device_type_managed()
{
  cuda::std::array<int, 4> data{};
  dlpack_array<1> shape   = {4};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0}; // CPU device, not CUDA managed
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_managed_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_stride_mismatch_layout_right()
{
  cuda::std::array<float, 6> data{};
  dlpack_array<2> shape   = {2, 3};
  dlpack_array<2> strides = {1, 2}; // Column-major, not row-major
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLFloat, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<float, 2, cuda::std::layout_right>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_stride_mismatch_layout_left()
{
  cuda::std::array<float, 6> data{};
  dlpack_array<2> shape   = {2, 3};
  dlpack_array<2> strides = {3, 1}; // Row-major, not column-major
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLFloat, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<float, 2, cuda::std::layout_left>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_zero_stride_layout_stride()
{
  cuda::std::array<int, 6> data{};
  dlpack_array<2> shape   = {2, 3};
  dlpack_array<2> strides = {0, 1}; // Zero stride is invalid
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<int, 2, cuda::std::layout_stride>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_null_strides_dlpack_v12()
{
  cuda::std::array<float, 6> data{};
  dlpack_array<2> shape = {2, 3};
  DLTensor tensor{};
  tensor.data    = data.data();
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 2;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLFloat, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = nullptr; // null strides not allowed in DLPack v1.2+

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<float, 2>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

void test_exception_misaligned_data()
{
  // Create a buffer that allows us to get a misaligned pointer
  alignas(16) cuda::std::array<char, 20> buffer{};
  // Get a pointer that's 1 byte into the buffer (misaligned for int)
  auto misaligned_ptr     = reinterpret_cast<int*>(buffer.data() + 1);
  dlpack_array<1> shape   = {3};
  dlpack_array<1> strides = {1};
  DLTensor tensor{};
  tensor.data    = misaligned_ptr;
  tensor.device  = DLDevice{kDLCPU, 0};
  tensor.ndim    = 1;
  tensor.dtype   = DLDataType{DLDataTypeCode::kDLInt, 32, 1};
  tensor.shape   = shape.data();
  tensor.strides = strides.data();

  bool caught = false;
  try
  {
    unused(cuda::to_host_mdspan<int, 1>(tensor));
  }
  catch (const std::invalid_argument&)
  {
    caught = true;
  }
  assert(caught);
}

bool test_exceptions()
{
  test_exception_wrong_rank();
  test_exception_wrong_dtype();
  test_exception_null_data();
  test_exception_null_shape();
  test_exception_negative_shape();
  test_exception_wrong_device_type_host();
  test_exception_wrong_device_type_device();
  test_exception_wrong_device_type_managed();
  test_exception_stride_mismatch_layout_right();
  test_exception_stride_mismatch_layout_left();
  test_exception_zero_stride_layout_stride();
#if DLPACK_MAJOR_VERSION > 1 || (DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION >= 2)
  test_exception_null_strides_dlpack_v12();
#endif
  test_exception_misaligned_data();
  return true;
}

//==============================================================================
// Test: Return type checking
//==============================================================================

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
     assert(test_rank1());
     assert(test_rank2_layout_right());
     assert(test_rank2_layout_left());
     assert(test_rank2_layout_stride());
     assert(test_element_types());
     assert(test_byte_offset());
     assert(test_empty_tensor());
     assert(test_return_types());
     assert(test_exceptions());))
#if !(DLPACK_MAJOR_VERSION > 1 || (DLPACK_MAJOR_VERSION == 1 && DLPACK_MINOR_VERSION >= 2))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_layout_stride_null_strides());))
#endif
  return 0;
}
