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

#include <nv/target>

#include "test_macros.h"
#include <dlpack/dlpack.h>

template <size_t Rank>
using dlpack_array = cuda::std::array<int64_t, Rank>;

//----------------------------------------------------------------------------------------------------------------------
// Exception tests

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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught   = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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
  bool caught    = false;
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

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_exceptions());))
  return 0;
}
