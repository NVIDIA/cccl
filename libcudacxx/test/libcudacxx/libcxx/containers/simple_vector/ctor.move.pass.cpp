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
#include <cuda/std/cstddef>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

struct NoDefault
{
  int value;

  NoDefault() = delete;
  TEST_HOST_DEVICE_FUNC explicit NoDefault(int v)
      : value(v)
  {}
};

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

static_assert(cuda::std::is_nothrow_move_constructible_v<cuda::__simple_vector<int>>, "");
static_assert(cuda::std::is_nothrow_move_constructible_v<cuda::__simple_vector<NoDefault>>, "");
static_assert(cuda::std::is_nothrow_move_constructible_v<cuda::__simple_vector<DestroyCounted>>, "");
static_assert(!cuda::std::is_copy_constructible_v<cuda::__simple_vector<int>>, "");

template <class T>
TEST_HOST_DEVICE_FUNC void test_move_empty()
{
  cuda::__simple_vector<T> src(0, cuda::no_init);
  cuda::__simple_vector<T> dst(cuda::std::move(src));

  assert(src.size() == 0);
  assert(src.empty());
  assert(src.data() == nullptr);
  assert(src.begin() == src.end());

  assert(dst.size() == 0);
  assert(dst.empty());
  assert(dst.data() == nullptr);
  assert(dst.begin() == dst.end());
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_move_non_empty()
{
  cuda::std::size_t n = 4;
  cuda::__simple_vector<T> src(n, cuda::no_init);
  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    ::new (static_cast<void*>(src.data() + i)) T{static_cast<int>(i + 1)};
  }

  T* original = src.data();
  cuda::__simple_vector<T> dst(cuda::std::move(src));

  assert(src.size() == 0);
  assert(src.empty());
  assert(src.data() == nullptr);
  assert(src.begin() == src.end());

  assert(dst.size() == n);
  assert(!dst.empty());
  assert(dst.data() == original);
  assert(dst.begin() == original);
  assert(dst.end() == original + n);

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    assert(dst.data()[i].value == static_cast<int>(i + 1));
  }
}

TEST_HOST_DEVICE_FUNC cuda::__simple_vector<DestroyCounted> make_destroy_counted(int n)
{
  cuda::__simple_vector<DestroyCounted> src(static_cast<cuda::std::size_t>(n), cuda::no_init);
  for (int i = 0; i < n; ++i)
  {
    ::new (static_cast<void*>(src.data() + i)) DestroyCounted{i};
  }
  // Construct the returned vector explicitly, otherwise copy elision would avoid the move we want to test
  return cuda::__simple_vector<DestroyCounted>(cuda::std::move(src));
}

TEST_HOST_DEVICE_FUNC void test_move_does_not_destroy_elements()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> src(3, cuda::no_init);
    for (int i = 0; i < 3; ++i)
    {
      ::new (static_cast<void*>(src.data() + i)) DestroyCounted{i};
    }
    assert(DestroyCounted_count == 3);

    cuda::__simple_vector<DestroyCounted> dst(cuda::std::move(src));
    assert(DestroyCounted_count == 3);
    assert(src.empty());
    assert(dst.size() == 3);
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test_moved_from_source_does_not_destroy()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> dst = make_destroy_counted(2);
    assert(DestroyCounted_count == 2);
    assert(dst.size() == 2);
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_move_empty<int>();
  test_move_empty<NoDefault>();
  test_move_empty<DestroyCounted>();

  test_move_non_empty<NoDefault>();
  test_move_does_not_destroy_elements();
  test_moved_from_source_does_not_destroy();
}

int main(int, char**)
{
  test();
  return 0;
}
