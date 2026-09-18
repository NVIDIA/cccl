//===----------------------------------------------------------------------===//
//
// Part of the CUDA Toolkit, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: force-tile
// error: dynamic allocations are not supported in tile mode

#include <cuda/__container/simple_vector.h>
#include <cuda/std/__new_>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_GLOBAL_VARIABLE int DestroyCounted_count = 0;

struct DestroyCounted
{
  int value;

  DestroyCounted() = delete;
  TEST_HOST_DEVICE_FUNC explicit DestroyCounted(int v)
      : value(v)
  {
    ++DestroyCounted_count;
  }

  TEST_HOST_DEVICE_FUNC ~DestroyCounted()
  {
    assert(DestroyCounted_count > 0);
    --DestroyCounted_count;
  }
};

static_assert(!cuda::std::is_trivially_destructible_v<DestroyCounted>, "");
static_assert(cuda::std::is_trivially_destructible_v<int>, "");

TEST_HOST_DEVICE_FUNC void test_trivial_destructor()
{
  cuda::__simple_vector<int> vec(4, cuda::no_init);
  assert(vec.size() == 4);
  for (int i = 0; i < 4; ++i)
  {
    ::new (static_cast<void*>(vec.data() + i)) int{i};
  }
}

TEST_HOST_DEVICE_FUNC void test_non_trivial_destructor()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> vec(3, cuda::no_init);
    for (int i = 0; i < 3; ++i)
    {
      ::new (static_cast<void*>(vec.data() + i)) DestroyCounted{i};
    }
    assert(DestroyCounted_count == 3);
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test_empty_non_trivial_destructor()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> vec(0, cuda::no_init);
    assert(vec.empty());
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_trivial_destructor();
  test_non_trivial_destructor();
  test_empty_non_trivial_destructor();
}

int main(int, char**)
{
  test();
  return 0;
}
