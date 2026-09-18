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

// Self assignment post-conditions are tested.
// ADDITIONAL_COMPILE_OPTIONS_HOST: -Wno-self-move

#include <cuda/__container/simple_vector.h>
#include <cuda/std/__new_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/memory>
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

static_assert(cuda::std::is_nothrow_move_assignable_v<cuda::__simple_vector<int>>, "");
static_assert(cuda::std::is_nothrow_move_assignable_v<cuda::__simple_vector<NoDefault>>, "");
static_assert(cuda::std::is_nothrow_move_assignable_v<cuda::__simple_vector<DestroyCounted>>, "");
static_assert(!cuda::std::is_copy_assignable_v<cuda::__simple_vector<int>>, "");

template <class T>
TEST_HOST_DEVICE_FUNC void construct_values(cuda::__simple_vector<T>& vec)
{
  for (cuda::std::size_t i = 0; i < vec.size(); ++i)
  {
    ::new (static_cast<void*>(vec.data() + i)) T{static_cast<int>(i + 1)};
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_assign_empty_to_empty()
{
  cuda::__simple_vector<T> src(0, cuda::no_init);
  cuda::__simple_vector<T> dst(0, cuda::no_init);

  cuda::__simple_vector<T>& ret = (dst = cuda::std::move(src));
  assert(cuda::std::addressof(ret) == cuda::std::addressof(dst));

  assert(src.size() == 0);
  assert(src.data() == nullptr);
  assert(dst.size() == 0);
  assert(dst.data() == nullptr);
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_assign_non_empty_to_empty()
{
  cuda::std::size_t n = 4;
  cuda::__simple_vector<T> src(n, cuda::no_init);
  construct_values(src);
  T* original = src.data();

  cuda::__simple_vector<T> dst(0, cuda::no_init);
  dst = cuda::std::move(src);

  assert(src.size() == 0);
  assert(src.empty());
  assert(src.data() == nullptr);

  assert(dst.size() == n);
  assert(!dst.empty());
  assert(dst.data() == original);
  assert(dst.end() == original + n);

  for (cuda::std::size_t i = 0; i < n; ++i)
  {
    assert(dst.data()[i].value == static_cast<int>(i + 1));
  }
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_assign_empty_to_non_empty()
{
  cuda::__simple_vector<T> src(0, cuda::no_init);
  cuda::__simple_vector<T> dst(3, cuda::no_init);
  construct_values(dst);

  dst = cuda::std::move(src);

  assert(src.size() == 0);
  assert(src.data() == nullptr);
  assert(dst.size() == 0);
  assert(dst.empty());
  assert(dst.data() == nullptr);
}

template <class T>
TEST_HOST_DEVICE_FUNC void test_assign_non_empty_to_non_empty()
{
  cuda::__simple_vector<T> src(4, cuda::no_init);
  construct_values(src);
  T* original = src.data();

  cuda::__simple_vector<T> dst(2, cuda::no_init);
  construct_values(dst);

  dst = cuda::std::move(src);

  assert(src.size() == 0);
  assert(src.data() == nullptr);
  assert(dst.size() == 4);
  assert(dst.data() == original);

  for (cuda::std::size_t i = 0; i < 4; ++i)
  {
    assert(dst.data()[i].value == static_cast<int>(i + 1));
  }
}

TEST_HOST_DEVICE_FUNC void test_assign_destroys_destination_elements()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> src(3, cuda::no_init);
    construct_values(src);
    cuda::__simple_vector<DestroyCounted> dst(2, cuda::no_init);
    construct_values(dst);
    assert(DestroyCounted_count == 5);

    dst = cuda::std::move(src);
    assert(DestroyCounted_count == 3);
    assert(src.empty());
    assert(dst.size() == 3);
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test_assign_empty_destroys_destination_elements()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> src(0, cuda::no_init);
    cuda::__simple_vector<DestroyCounted> dst(2, cuda::no_init);
    construct_values(dst);
    assert(DestroyCounted_count == 2);

    dst = cuda::std::move(src);
    assert(DestroyCounted_count == 0);
    assert(dst.empty());
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test_self_move_assignment()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> vec(3, cuda::no_init);
    construct_values(vec);
    DestroyCounted* original = vec.data();

    vec = cuda::std::move(vec);

    assert(vec.size() == 3);
    assert(vec.data() == original);
    assert(DestroyCounted_count == 3);
  }
  assert(DestroyCounted_count == 0);
}

TEST_HOST_DEVICE_FUNC void test()
{
  test_assign_empty_to_empty<int>();
  test_assign_empty_to_empty<NoDefault>();

  test_assign_non_empty_to_empty<NoDefault>();
  test_assign_empty_to_non_empty<NoDefault>();
  test_assign_non_empty_to_non_empty<NoDefault>();

  test_assign_destroys_destination_elements();
  test_assign_empty_destroys_destination_elements();
  test_self_move_assignment();
}

int main(int, char**)
{
  test();
  return 0;
}
