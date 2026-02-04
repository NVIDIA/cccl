//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/std/utility>
#include <cuda/utility>

#include "test_macros.h"

void check_datatype(const DLDataType& dt, uint8_t code, uint8_t bits, uint16_t lanes)
{
  assert(dt.code == code);
  assert(dt.bits == bits);
  assert(dt.lanes == lanes);
}

bool test_mdspan_to_dlpack_wrapper_default_ctor()
{
  cuda::__dlpack_tensor<3> dlpack_wrapper{};
  DLDataType default_dtype = {};
  DLDevice default_device  = {};
  auto tensor              = dlpack_wrapper.get();
  assert(tensor.device.device_type == default_device.device_type);
  assert(tensor.device.device_id == default_device.device_id);
  check_datatype(tensor.dtype, default_dtype.code, default_dtype.bits, default_dtype.lanes);
  assert(tensor.shape != nullptr);
  assert(tensor.strides != nullptr);
  return true;
}

bool test_dlpack_wrapper_copy_ctor()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data[6]     = {0, 1, 2, 3, 4, 5};
  cuda::host_mdspan<int, extents_t> md{data, extents_t{}};
  auto w            = cuda::to_dlpack_tensor(md);
  auto t            = w.get();
  auto* shape_ptr   = t.shape;
  auto* strides_ptr = t.strides;

  auto w2 = w; // copy construct
  // Copy must not alias the source wrapper's shape/stride storage.
  auto t2 = w2.get();
  assert(t2.shape != nullptr);
  assert(t2.strides != nullptr);
  assert(t2.shape != shape_ptr);
  assert(t2.strides != strides_ptr);

  // Source wrapper must remain intact.
  assert(t.shape == shape_ptr);
  assert(t.strides == strides_ptr);

  // Sanity-check copied tensor metadata and values.
  assert(t2.device.device_type == kDLCPU);
  assert(t2.device.device_id == 0);
  assert(t2.ndim == 2);
  check_datatype(t2.dtype, kDLInt, 32, 1);
  assert(t2.shape[0] == 2);
  assert(t2.shape[1] == 3);
  assert(t2.strides[0] == 3);
  assert(t2.strides[1] == 1);
  assert(t2.byte_offset == 0);
  assert(t2.data == data);
  return true;
}

bool test_dlpack_wrapper_get()
{
  using wrapper_t = cuda::__dlpack_tensor<2>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const wrapper_t&>().get()), ::DLTensor>);
  return true;
}

void test_dlpack_wrapper_deleted_operations()
{
  using wrapper_t = cuda::__dlpack_tensor<2>;
  static_assert(cuda::std::is_copy_constructible_v<wrapper_t>);
  static_assert(cuda::std::is_move_constructible_v<wrapper_t>);
  static_assert(cuda::std::is_copy_assignable_v<wrapper_t>);
  static_assert(cuda::std::is_move_assignable_v<wrapper_t>);
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST,
               (assert(test_mdspan_to_dlpack_wrapper_default_ctor()); //
                assert(test_dlpack_wrapper_copy_ctor());
                assert(test_dlpack_wrapper_get());))
  return 0;
}
