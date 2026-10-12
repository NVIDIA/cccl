//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_ALLOC_CONSTEXPR_TYPES_H
#define TEST_ALLOC_CONSTEXPR_TYPES_H

#include <cuda/std/tuple>

#include "allocators.h"
#include "test_macros.h"

struct constexpr_alloc_arg
{
  using allocator_type = A1<int>;

  int value;

  TEST_FUNC constexpr constexpr_alloc_arg()
      : value(0)
  {}

  // Ordinary constructors satisfy [tuple.cnstr] is_constructible constraints.
  // Their values differ from the uses-allocator constructors, which are the ones the tuple must call.
  TEST_FUNC constexpr explicit constexpr_alloc_arg(int)
      : value(-1)
  {}

  TEST_FUNC constexpr constexpr_alloc_arg(const constexpr_alloc_arg&)
      : value(-1)
  {}

  TEST_FUNC constexpr constexpr_alloc_arg(constexpr_alloc_arg&&)
      : value(-1)
  {}

  TEST_FUNC constexpr constexpr_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&)
      : value(1)
  {}

  TEST_FUNC constexpr constexpr_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, int v)
      : value(v)
  {}

  TEST_FUNC constexpr constexpr_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, const constexpr_alloc_arg& other)
      : value(other.value)
  {}

  TEST_FUNC constexpr constexpr_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, constexpr_alloc_arg&& other)
      : value(other.value)
  {}
};

struct constexpr_alloc_last
{
  using allocator_type = A1<int>;

  int value;

  TEST_FUNC constexpr constexpr_alloc_last()
      : value(0)
  {}

  // Ordinary constructors satisfy [tuple.cnstr] is_constructible constraints.
  // Their values differ from the uses-allocator constructors, which are the ones the tuple must call.
  TEST_FUNC constexpr explicit constexpr_alloc_last(int)
      : value(-1)
  {}

  TEST_FUNC constexpr constexpr_alloc_last(const constexpr_alloc_last&)
      : value(-1)
  {}

  TEST_FUNC constexpr constexpr_alloc_last(constexpr_alloc_last&&)
      : value(-1)
  {}

  TEST_FUNC constexpr constexpr_alloc_last(const A1<int>&)
      : value(2)
  {}

  TEST_FUNC constexpr constexpr_alloc_last(int v, const A1<int>&)
      : value(v)
  {}

  TEST_FUNC constexpr constexpr_alloc_last(const constexpr_alloc_last& other, const A1<int>&)
      : value(other.value)
  {}

  TEST_FUNC constexpr constexpr_alloc_last(constexpr_alloc_last&& other, const A1<int>&)
      : value(other.value)
  {}
};

struct nothrow_alloc_arg
{
  using allocator_type = A1<int>;

  TEST_FUNC nothrow_alloc_arg() noexcept {}
  TEST_FUNC nothrow_alloc_arg(int) noexcept {}
  TEST_FUNC nothrow_alloc_arg(const nothrow_alloc_arg&) noexcept {}
  TEST_FUNC nothrow_alloc_arg(nothrow_alloc_arg&&) noexcept {}
  TEST_FUNC nothrow_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&) noexcept {}
  TEST_FUNC nothrow_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, int) noexcept {}
  TEST_FUNC nothrow_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, const nothrow_alloc_arg&) noexcept {}
  TEST_FUNC nothrow_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, nothrow_alloc_arg&&) noexcept {}
};

// Ordinary constructors are non-throwing. The uses-allocator constructors are not,
// so an allocator-extended tuple constructor must not be noexcept.
struct throw_on_alloc_arg
{
  using allocator_type = A1<int>;

  TEST_FUNC throw_on_alloc_arg() noexcept {}
  TEST_FUNC throw_on_alloc_arg(int) noexcept {}
  TEST_FUNC throw_on_alloc_arg(float) noexcept {}
  TEST_FUNC throw_on_alloc_arg(const throw_on_alloc_arg&) noexcept {}
  TEST_FUNC throw_on_alloc_arg(throw_on_alloc_arg&&) noexcept {}

  TEST_FUNC throw_on_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(1);))
  }
  TEST_FUNC throw_on_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, int)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(1);))
  }
  TEST_FUNC throw_on_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, float)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(1);))
  }
  TEST_FUNC throw_on_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, const throw_on_alloc_arg&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(1);))
  }
  TEST_FUNC throw_on_alloc_arg(cuda::std::allocator_arg_t, const A1<int>&, throw_on_alloc_arg&&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(1);))
  }
};

struct throw_on_alloc_last
{
  using allocator_type = A1<int>;

  TEST_FUNC throw_on_alloc_last() noexcept {}
  TEST_FUNC throw_on_alloc_last(int) noexcept {}
  TEST_FUNC throw_on_alloc_last(float) noexcept {}
  TEST_FUNC throw_on_alloc_last(const throw_on_alloc_last&) noexcept {}
  TEST_FUNC throw_on_alloc_last(throw_on_alloc_last&&) noexcept {}

  TEST_FUNC throw_on_alloc_last(const A1<int>&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(2);))
  }
  TEST_FUNC throw_on_alloc_last(int, const A1<int>&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(2);))
  }
  TEST_FUNC throw_on_alloc_last(float, const A1<int>&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(2);))
  }
  TEST_FUNC throw_on_alloc_last(const throw_on_alloc_last&, const A1<int>&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(2);))
  }
  TEST_FUNC throw_on_alloc_last(throw_on_alloc_last&&, const A1<int>&)
  {
    NV_IF_TARGET(NV_IS_HOST, (TEST_THROW(2);))
  }
};

#endif // TEST_ALLOC_CONSTEXPR_TYPES_H
