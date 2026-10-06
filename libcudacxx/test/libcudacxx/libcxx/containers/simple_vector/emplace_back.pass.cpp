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
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/memory>
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

struct FromRvalue
{
  int value;
  bool moved;

  FromRvalue() = delete;
  TEST_HOST_DEVICE_FUNC explicit FromRvalue(int v)
      : value(v)
      , moved(false)
  {}

  FromRvalue(const FromRvalue&) = delete;
  TEST_HOST_DEVICE_FUNC FromRvalue(FromRvalue&& other)
      : value(other.value)
      , moved(true)
  {
    other.value = -1;
  }
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

TEST_HOST_DEVICE_FUNC void test_emplace_returns_inserted_element()
{
  cuda::__simple_vector<NoDefault> vec(2, cuda::no_init);
  NoDefault& first = vec.emplace_back(11);
  assert(cuda::std::addressof(first) == vec.data());
  assert(first.value == 11);
  assert(vec.size() == 1);
  assert(vec.end() == vec.data() + 1);

  NoDefault& second = vec.emplace_back(22);
  assert(cuda::std::addressof(second) == vec.data() + 1);
  assert(second.value == 22);
  assert(vec.size() == 2);
  assert(vec.end() == vec.data() + 2);
}

TEST_HOST_DEVICE_FUNC void test_emplace_forwards_rvalue()
{
  cuda::__simple_vector<FromRvalue> vec(1, cuda::no_init);
  FromRvalue src(42);
  FromRvalue& inserted = vec.emplace_back(cuda::std::move(src));
  assert(inserted.moved);
  assert(inserted.value == 42);
  assert(src.value == -1);
  assert(vec.size() == 1);
}

TEST_HOST_DEVICE_FUNC void test_emplace_int()
{
  cuda::__simple_vector<int> vec(3, cuda::no_init);
  int& ref = vec.emplace_back(5);
  assert(ref == 5);
  vec.emplace_back(6);
  vec.emplace_back(7);
  assert(vec.size() == 3);
  assert(vec.data()[0] == 5);
  assert(vec.data()[1] == 6);
  assert(vec.data()[2] == 7);
}

TEST_HOST_DEVICE_FUNC void test_emplace_constructs_in_order()
{
  DestroyCounted_count = 0;
  {
    cuda::__simple_vector<DestroyCounted> vec(3, cuda::no_init);
    vec.emplace_back(1);
    vec.emplace_back(2);
    assert(DestroyCounted_count == 2);
    assert(vec.size() == 2);
    assert(vec.data()[0].value == 1);
    assert(vec.data()[1].value == 2);
  }
  assert(DestroyCounted_count == 0);
}

#if TEST_HAS_EXCEPTIONS()
static int ThrowsCounted_count       = 0;
static int ThrowsCounted_constructed = 0;
static int ThrowsCounted_throw_after = 0;

struct ThrowsCounted
{
  static void reset()
  {
    ThrowsCounted_count = ThrowsCounted_constructed = ThrowsCounted_throw_after = 0;
  }

  explicit ThrowsCounted(int)
  {
    ++ThrowsCounted_constructed;
    if (ThrowsCounted_throw_after > 0 && --ThrowsCounted_throw_after == 0)
    {
      TEST_THROW(1);
    }
    ++ThrowsCounted_count;
  }

  ~ThrowsCounted()
  {
    assert(ThrowsCounted_count > 0);
    --ThrowsCounted_count;
  }
};

void test_emplace_throw_does_not_destroy_unconstructed()
{
  ThrowsCounted::reset();
  {
    cuda::__simple_vector<ThrowsCounted> vec(3, cuda::no_init);
    ThrowsCounted_throw_after = 3;
    vec.emplace_back(1);
    vec.emplace_back(2);
    assert(vec.size() == 2);
    assert(ThrowsCounted_count == 2);

    try
    {
      vec.emplace_back(3);
      assert(false);
    }
    catch (...)
    {}

    assert(vec.size() == 2);
    assert(ThrowsCounted_count == 2);
    assert(ThrowsCounted_constructed == 3);
  }
  assert(ThrowsCounted_count == 0);
}

void test_emplace_throw_on_first_element()
{
  ThrowsCounted::reset();
  {
    cuda::__simple_vector<ThrowsCounted> vec(2, cuda::no_init);
    ThrowsCounted_throw_after = 1;
    try
    {
      vec.emplace_back(1);
      assert(false);
    }
    catch (...)
    {}

    assert(vec.empty());
    assert(ThrowsCounted_count == 0);
    assert(ThrowsCounted_constructed == 1);
  }
  assert(ThrowsCounted_count == 0);
}

void test_exceptions()
{
  test_emplace_throw_does_not_destroy_unconstructed();
  test_emplace_throw_on_first_element();
}
#endif // TEST_HAS_EXCEPTIONS()

TEST_HOST_DEVICE_FUNC void test()
{
  test_emplace_returns_inserted_element();
  test_emplace_forwards_rvalue();
  test_emplace_int();
  test_emplace_constructs_in_order();
}

int main(int, char**)
{
  test();
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
