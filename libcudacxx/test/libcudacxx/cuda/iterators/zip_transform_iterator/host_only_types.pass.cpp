//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: enable-tile

// UNSUPPORTED: nvrtc

#include <cuda/iterator>
#include <cuda/std/cassert>

#include "host_device_types.h"
#include "test_macros.h"

struct host_plus_value
{
  int value_;

  host_plus_value(const int value = 42) noexcept
      : value_(value)
  {}

  host_plus_value(const host_plus_value& other) noexcept
      : value_(other.value_)
  {}

  host_plus_value(host_plus_value&& other) noexcept
      : value_(other.value_)
  {}

  host_plus_value& operator=(const host_plus_value& other) noexcept
  {
    value_ = other.value_;
    return *this;
  }

  host_plus_value& operator=(host_plus_value&& other) noexcept
  {
    value_ = other.value_;
    return *this;
  }

  int operator()(const int& val) const noexcept
  {
    return val + value_;
  }

  ~host_plus_value() noexcept {}
};

void test()
{
  host_only_container vec{};

  {
    using Iter                   = typename host_only_container::iterator;
    using zip_transform_iterator = cuda::zip_transform_iterator<host_plus_value, Iter>;

    const zip_transform_iterator default_constructed{};
    zip_transform_iterator value_constructed{host_plus_value{42}, vec.begin()};

    zip_transform_iterator copy_constructed{default_constructed};
    zip_transform_iterator move_constructed{::cuda::std::move(value_constructed)};

    [[maybe_unused]] zip_transform_iterator copy_assigned{};
    copy_assigned = copy_constructed;

    [[maybe_unused]] zip_transform_iterator move_assigned{};
    move_assigned = ::cuda::std::move(move_constructed);
  }

  cuda::zip_transform_iterator iter1{host_plus_value{42}, vec.begin()};
  const cuda::zip_transform_iterator iter2{host_plus_value{42}, vec.begin() + 1};
  assert(iter1 != iter2);

  {
    assert(++iter1 == iter2);
    assert(--iter1 != iter2);
  }

  {
    assert(iter1++ != iter2);
    assert(iter1-- == iter2);
  }

  {
    assert(iter1 + 1 == iter2);
    assert(1 + iter1 == iter2);
    assert(iter1 - 1 != iter2);
    assert(iter2 - iter1 == 1);
  }

  {
    iter1 += 1;
    assert(iter1 == iter2);
    iter1 -= 1;
    assert(iter1 != iter2);
  }

  {
    assert((iter1[1] == vec[1] + 42));
    assert((iter2[1] == vec[2] + 42));

    assert((*iter1 == vec[0] + 42));
    assert((*iter2 == vec[1] + 42));
  }
}

int main(int arg, char** argv)
{
  NV_IF_TARGET(NV_IS_HOST, (test();))
  return 0;
}
