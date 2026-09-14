//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: dynamic allocation is not supported in tile mode

#include <cuda/std/__memory_>
#include <cuda/std/cassert>

#include "host_device_types.h"
#include "test_macros.h"

#if _CCCL_DEVICE_COMPILATION()
template <class T>
TEST_DEVICE_FUNC void test_single()
{
  {
    cuda::std::unique_ptr<T> ptr = cuda::std::make_unique<T>(42);
    assert(ptr);
    assert(*ptr == T{42});
  }

  {
    cuda::std::unique_ptr<T> ptr = cuda::std::make_unique<T>();
    assert(ptr);
    assert(*ptr == T{0});
  }

  {
    cuda::std::unique_ptr<T> ptr = cuda::std::make_unique_for_overwrite<T>();
    assert(ptr);
    *ptr = T{1337};
    assert(*ptr == T{1337});
  }

  {
    cuda::std::unique_ptr<T> ptr{new T{42}};
    assert(ptr);
    assert(*ptr == T{42});
  }
}

template <class T>
TEST_DEVICE_FUNC void test_array()
{
  {
    cuda::std::unique_ptr<T[]> arr = cuda::std::make_unique<T[]>(4);
    assert(arr);
    for (int i = 0; i < 4; ++i)
    {
      assert(arr[i] == T{0});
    }

    arr[2] = T{1337};
    assert(arr[2] == T{1337});
  }

  {
    cuda::std::unique_ptr<T[]> arr = cuda::std::make_unique_for_overwrite<T[]>(4);
    assert(arr);
    for (int i = 0; i < 4; ++i)
    {
      arr[i] = T{i};
    }

    for (int i = 0; i < 4; ++i)
    {
      assert(arr[i] == T{i});
    }
  }

  {
    cuda::std::unique_ptr<T[]> arr{new T[2]};
    assert(arr);
    arr[0] = T{42};
    assert(arr[0] == T{42});
  }
}

TEST_DEVICE_FUNC void test()
{
  test_single<device_only_type>();
  test_array<device_only_type>();
}
#endif // _CCCL_DEVICE_COMPILATION()

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_DEVICE, test();)
  return 0;
}
