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
  cuda::DLPackWrapper<3> dlpack_wrapper{};
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
  auto w            = cuda::mdspan_to_dlpack(md);
  auto* shape_ptr   = w->shape;
  auto* strides_ptr = w->strides;

  auto w2 = w; // copy construct
  // Copy must not alias the source wrapper's shape/stride storage.
  assert(w2->shape != nullptr);
  assert(w2->strides != nullptr);
  assert(w2->shape != shape_ptr);
  assert(w2->strides != strides_ptr);

  // Source wrapper must remain intact.
  assert(w->shape == shape_ptr);
  assert(w->strides == strides_ptr);

  // Sanity-check copied tensor metadata and values.
  assert(w2->device.device_type == kDLCPU);
  assert(w2->device.device_id == 0);
  assert(w2->ndim == 2);
  check_datatype(w2->dtype, kDLInt, 32, 1);
  assert(w2->shape[0] == 2);
  assert(w2->shape[1] == 3);
  assert(w2->strides[0] == 3);
  assert(w2->strides[1] == 1);
  assert(w2->byte_offset == 0);
  assert(w2->data == data);
  return true;
}

bool test_dlpack_wrapper_move_ctor()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data[6]     = {0, 1, 2, 3, 4, 5};
  cuda::host_mdspan<int, extents_t> md{data, extents_t{}};
  auto w            = cuda::mdspan_to_dlpack(md);
  auto* shape_ptr   = w->shape;
  auto* strides_ptr = w->strides;
  auto moved        = cuda::std::move(w); // move construct

  // Moved-to wrapper must not keep pointers to moved-from storage.
  assert(moved->shape != nullptr);
  assert(moved->strides != nullptr);
  assert(moved->shape != shape_ptr);
  assert(moved->strides != strides_ptr);

  // Moved-from wrapper is explicitly reset to a default/empty DLTensor.
  assert(w->shape == nullptr);
  assert(w->strides == nullptr);
  assert(w->data == nullptr);
  assert(w->ndim == 0);

  // Sanity-check moved-to tensor metadata and values.
  assert(moved->device.device_type == kDLCPU);
  assert(moved->device.device_id == 0);
  assert(moved->ndim == 2);
  check_datatype(moved->dtype, kDLInt, 32, 1);
  assert(moved->shape[0] == 2);
  assert(moved->shape[1] == 3);
  assert(moved->strides[0] == 3);
  assert(moved->strides[1] == 1);
  assert(moved->byte_offset == 0);
  assert(moved->data == data);
  return true;
}

bool test_dlpack_wrapper_copy_assignment()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data_a[6]   = {0, 1, 2, 3, 4, 5};
  int data_b[6]   = {6, 7, 8, 9, 10, 11};
  cuda::host_mdspan<int, extents_t> md_a{data_a, extents_t{}};
  cuda::host_mdspan<int, extents_t> md_b{data_b, extents_t{}};
  auto a              = cuda::mdspan_to_dlpack(md_a);
  auto b              = cuda::mdspan_to_dlpack(md_b);
  auto* b_shape_ptr   = b->shape;
  auto* b_strides_ptr = b->strides;

  b = a; // copy assign
  // Destination must keep pointing to its own member arrays (not to `a`).
  assert(b->shape == b_shape_ptr);
  assert(b->strides == b_strides_ptr);
  assert(b->shape != a->shape);
  assert(b->strides != a->strides);

  // Values must be copied correctly.
  assert(b->data == data_a);
  assert(b->ndim == 2);
  assert(b->shape[0] == 2);
  assert(b->shape[1] == 3);
  assert(b->strides[0] == 3);
  assert(b->strides[1] == 1);
  return true;
}

bool test_dlpack_wrapper_move_assignment()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data_a[6]   = {0, 1, 2, 3, 4, 5};
  int data_b[6]   = {6, 7, 8, 9, 10, 11};
  cuda::host_mdspan<int, extents_t> md_a{data_a, extents_t{}};
  cuda::host_mdspan<int, extents_t> md_b{data_b, extents_t{}};
  auto a              = cuda::mdspan_to_dlpack(md_a);
  auto b              = cuda::mdspan_to_dlpack(md_b);
  auto* a_shape_ptr   = a->shape;
  auto* a_strides_ptr = a->strides;
  auto* b_shape_ptr   = b->shape;
  auto* b_strides_ptr = b->strides;

  b = cuda::std::move(a); // move assign
  // Destination must keep pointing to its own member arrays, not the source's.
  assert(b->shape == b_shape_ptr);
  assert(b->strides == b_strides_ptr);
  assert(b->shape != a_shape_ptr);
  assert(b->strides != a_strides_ptr);

  // Source must be reset.
  assert(a->shape == nullptr);
  assert(a->strides == nullptr);
  assert(a->data == nullptr);
  assert(a->ndim == 0);

  // Values must be moved correctly.
  assert(b->data == data_a);
  assert(b->ndim == 2);
  assert(b->shape[0] == 2);
  assert(b->shape[1] == 3);
  assert(b->strides[0] == 3);
  assert(b->strides[1] == 1);
  return true;
}

bool test_dlpack_wrapper_get()
{
  using wrapper_t = cuda::DLPackWrapper<2>;
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<wrapper_t&>().get()), ::DLTensor&>);
  static_assert(cuda::std::is_same_v<decltype(cuda::std::declval<const wrapper_t&>().get()), const ::DLTensor&>);

  wrapper_t w{};
  // `get()` must return a reference to the same underlying `DLTensor` as `operator->()`.
  assert(&w.get() == w.operator->());

  // Mutating through the reference returned by `get()` must be observable through `operator->()`.
  auto& t = w.get();
  t.ndim  = 123;
  assert(w->ndim == 123);

  // Const overload should also alias the same underlying object.
  const wrapper_t& cw = w;
  assert(&cw.get() == cw.operator->());
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
