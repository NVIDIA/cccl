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
  auto& tensor             = dlpack_wrapper.get();
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
  auto w            = cuda::to_dlpack(md);
  auto& t           = w.get();
  auto* shape_ptr   = t.shape;
  auto* strides_ptr = t.strides;

  auto w2 = w; // copy construct
  // Copy must not alias the source wrapper's shape/stride storage.
  auto& t2 = w2.get();
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

bool test_dlpack_wrapper_move_ctor()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data[6]     = {0, 1, 2, 3, 4, 5};
  cuda::host_mdspan<int, extents_t> md{data, extents_t{}};
  auto w            = cuda::to_dlpack(md);
  auto& t           = w.get();
  auto* shape_ptr   = t.shape;
  auto* strides_ptr = t.strides;
  auto moved        = cuda::std::move(w); // move construct

  // Moved-to wrapper must not keep pointers to moved-from storage.
  auto& tm = moved.get();
  assert(tm.shape != nullptr);
  assert(tm.strides != nullptr);
  assert(tm.shape != shape_ptr);
  assert(tm.strides != strides_ptr);

  // Moved-from wrapper is explicitly reset to a default/empty DLTensor.
  assert(t.shape == nullptr);
  assert(t.strides == nullptr);
  assert(t.data == nullptr);
  assert(t.ndim == 0);

  // Sanity-check moved-to tensor metadata and values.
  assert(tm.device.device_type == kDLCPU);
  assert(tm.device.device_id == 0);
  assert(tm.ndim == 2);
  check_datatype(tm.dtype, kDLInt, 32, 1);
  assert(tm.shape[0] == 2);
  assert(tm.shape[1] == 3);
  assert(tm.strides[0] == 3);
  assert(tm.strides[1] == 1);
  assert(tm.byte_offset == 0);
  assert(tm.data == data);
  return true;
}

bool test_dlpack_wrapper_copy_assignment()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data_a[6]   = {0, 1, 2, 3, 4, 5};
  int data_b[6]   = {6, 7, 8, 9, 10, 11};
  cuda::host_mdspan<int, extents_t> md_a{data_a, extents_t{}};
  cuda::host_mdspan<int, extents_t> md_b{data_b, extents_t{}};
  auto a              = cuda::to_dlpack(md_a);
  auto b              = cuda::to_dlpack(md_b);
  auto& ta            = a.get();
  auto& tb            = b.get();
  auto* b_shape_ptr   = tb.shape;
  auto* b_strides_ptr = tb.strides;

  b = a; // copy assign
  // Destination must keep pointing to its own member arrays (not to `a`).
  assert(tb.shape == b_shape_ptr);
  assert(tb.strides == b_strides_ptr);
  assert(tb.shape != ta.shape);
  assert(tb.strides != ta.strides);

  // Values must be copied correctly.
  assert(tb.data == data_a);
  assert(tb.ndim == 2);
  assert(tb.shape[0] == 2);
  assert(tb.shape[1] == 3);
  assert(tb.strides[0] == 3);
  assert(tb.strides[1] == 1);
  return true;
}

bool test_dlpack_wrapper_move_assignment()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data_a[6]   = {0, 1, 2, 3, 4, 5};
  int data_b[6]   = {6, 7, 8, 9, 10, 11};
  cuda::host_mdspan<int, extents_t> md_a{data_a, extents_t{}};
  cuda::host_mdspan<int, extents_t> md_b{data_b, extents_t{}};
  auto a              = cuda::to_dlpack(md_a);
  auto b              = cuda::to_dlpack(md_b);
  auto& ta            = a.get();
  auto& tb            = b.get();
  auto* a_shape_ptr   = ta.shape;
  auto* a_strides_ptr = ta.strides;
  auto* b_shape_ptr   = tb.shape;
  auto* b_strides_ptr = tb.strides;

  b = cuda::std::move(a); // move assign
  // Destination must keep pointing to its own member arrays, not the source's.
  assert(tb.shape == b_shape_ptr);
  assert(tb.strides == b_strides_ptr);
  assert(tb.shape != a_shape_ptr);
  assert(tb.strides != a_strides_ptr);

  // Source must be reset.
  assert(ta.shape == nullptr);
  assert(ta.strides == nullptr);
  assert(ta.data == nullptr);
  assert(ta.ndim == 0);

  // Values must be moved correctly.
  assert(tb.data == data_a);
  assert(tb.ndim == 2);
  assert(tb.shape[0] == 2);
  assert(tb.shape[1] == 3);
  assert(tb.strides[0] == 3);
  assert(tb.strides[1] == 1);
  return true;
}

bool test_dlpack_wrapper_get()
{
  using wrapper_t = cuda::__dlpack_tensor<2>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<wrapper_t&>().get()), ::DLTensor&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const wrapper_t&>().get()), const ::DLTensor&>);

  wrapper_t w{};
  // Mutating through the reference returned by `get()` must be observable.
  auto& t = w.get();
  t.ndim  = 123;
  assert(w.get().ndim == 123);

  // Const overload should also alias the same underlying object.
  const wrapper_t& cw = w;
  assert(&cw.get() == &w.get());
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(
    NV_IS_HOST,
    (assert(test_mdspan_to_dlpack_wrapper_default_ctor()); assert(test_dlpack_wrapper_copy_ctor());
     assert(test_dlpack_wrapper_move_ctor());
     assert(test_dlpack_wrapper_copy_assignment());
     assert(test_dlpack_wrapper_move_assignment());
     assert(test_dlpack_wrapper_get());))
  return 0;
}
