//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc

// UNSUPPORTED: force-tile
// error: dynamic allocation is not supported in tile mode

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "host_device_types.h"
#include "test_macros.h"

// Covers the functions that have their execution space checks suppressed, so that a host only type
// can be used with them: default_delete<T>::operator(), default_delete<T[]>::operator(),
// make_unique<T>() and make_unique_for_overwrite<T>().
//
// The two array factories, make_unique<T[]>() and make_unique_for_overwrite<T[]>(), are not
// exercised here. They allocate with new T[n], which nvcc lowers to an unnamed __host__ __device__
// element construction loop, and _CCCL_EXEC_CHECK_DISABLE on the enclosing function does not extend
// to that generated helper:
//
//   error #20011-D: calling a __host__ function("host_only_type::host_only_type(int)")
//   from a __host__ __device__ function("<unknown>") is not allowed
//
// device_only_types.pass.cpp does exercise both, because its helpers are __device__ rather than
// __host__ __device__, so no cross execution space call arises.

template <class T>
void test_single()
{
  { // make_unique<T> forwards its arguments to the single object overload
    cuda::std::unique_ptr<T> ptr = cuda::std::make_unique<T>(42);
    assert(ptr);
    assert(*ptr == T{42});
  } // default_delete<T>::operator() runs here

  { // make_unique<T> value initializes without arguments
    cuda::std::unique_ptr<T> ptr = cuda::std::make_unique<T>();
    assert(ptr);
    assert(*ptr == T{0});
  }

  { // make_unique_for_overwrite<T> default initializes, so the value is only known after a write
    cuda::std::unique_ptr<T> ptr = cuda::std::make_unique_for_overwrite<T>();
    assert(ptr);
    *ptr = T{1337};
    assert(*ptr == T{1337});
  }

  { // default_delete<T> also runs for a pointer that the unique_ptr did not allocate itself
    cuda::std::unique_ptr<T> ptr{new T{42}};
    assert(ptr);
    assert(*ptr == T{42});
  }
}

template <class T>
void test_array()
{
  { // default_delete<T[]>::operator() runs for an array the unique_ptr took ownership of
    cuda::std::unique_ptr<T[]> arr{new T[4]};
    assert(arr);
    for (int i = 0; i < 4; ++i)
    {
      assert(arr[i] == T{0});
    }

    arr[2] = T{1337};
    assert(arr[2] == T{1337});
  } // default_delete<T[]>::operator() runs here

  { // and again after a reset, which re-enters the deleter
    cuda::std::unique_ptr<T[]> arr{new T[2]};
    arr[0] = T{42};
    assert(arr[0] == T{42});
    arr.reset(new T[3]);
    assert(arr);
    assert(arr[0] == T{0});
  }
}

void test()
{
  test_single<host_only_type>();
  test_array<host_only_type>();
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
