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
#include <cuda/memory>
#include <cuda/std/__new_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>

#include "test_macros.h"

TEST_DIAG_SUPPRESS_MSVC(4324) // structure was padded due to alignment specifier

struct NoDefault
{
  int value;

  NoDefault() = delete;
  TEST_HOST_DEVICE_FUNC explicit NoDefault(int v)
      : value(v)
  {}
};

struct alignas(32) OverAligned
{
  int value;

  OverAligned() = delete;
  TEST_HOST_DEVICE_FUNC explicit OverAligned(int v)
      : value(v)
  {}
};

static_assert(!cuda::std::is_default_constructible_v<NoDefault>, "");
static_assert(!cuda::std::is_default_constructible_v<cuda::__simple_vector<int>>, "");
static_assert(cuda::std::is_constructible_v<cuda::__simple_vector<int>, cuda::std::size_t, cuda::no_init_t>, "");
static_assert(!cuda::std::is_constructible_v<cuda::__simple_vector<int>, cuda::std::size_t>, "");

template <class T>
TEST_HOST_DEVICE_FUNC void test_empty()
{
  cuda::__simple_vector<T> vec(0, cuda::no_init);
  assert(vec.size() == 0);
  assert(vec.empty());
  assert(vec.data() == nullptr);
  assert(vec.begin() == vec.end());
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_uninitialized_storage(cuda::std::size_t n)
{
  cuda::__simple_vector<T> vec(n, cuda::no_init);
  assert(vec.size() == n);
  assert(!vec.empty());
  assert(vec.data() != nullptr);
  assert(vec.begin() == vec.data());
  assert(vec.end() == vec.data() + n);
  assert(cuda::is_aligned(vec.data(), alignof(T)));

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    ::new (static_cast<void*>(vec.data() + i)) T{static_cast<int>(i + 1)};
  }

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    assert(vec.data()[i].value == static_cast<int>(i + 1));
  }
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_empty<int>();
  test_empty<NoDefault>();
  test_empty<OverAligned>();

  test_uninitialized_storage<NoDefault>(1);
  test_uninitialized_storage<NoDefault>(5);
  test_uninitialized_storage<OverAligned>(3);
}

int main(int, char**)
{
  test();
  return 0;
}
