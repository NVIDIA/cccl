//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//

// <cuda/std/concepts>

// cuda::__nothrow_list_initializable_from
// cuda::__nothrow_emplaceable_from

#include <cuda/std/concepts>

#include "test_macros.h"

struct NothrowFromInt
{
  TEST_FUNC explicit NothrowFromInt(int) noexcept {}
};

struct ThrowsFromInt
{
  TEST_FUNC explicit ThrowsFromInt(int) {}
};

struct NothrowDefault
{
  TEST_FUNC NothrowDefault() noexcept {}
};

struct ThrowsDefault
{
  TEST_FUNC ThrowsDefault() {}
};

struct Aggregate
{
  int x;
};

struct NothrowFn
{
  TEST_FUNC int operator()() noexcept
  {
    return 1;
  }
};

struct ThrowFn
{
  TEST_FUNC int operator()()
  {
    return 1;
  }
};

struct NothrowAddFn
{
  TEST_FUNC int operator()(int a, int b) noexcept
  {
    return a + b;
  }
};

struct Immovable
{
  int x;

  TEST_FUNC explicit Immovable(int v) noexcept
      : x(v)
  {}

  Immovable(const Immovable&) = delete;
  Immovable(Immovable&&)      = delete;
};

struct ThrowsImmovable
{
  int x;

  TEST_FUNC explicit ThrowsImmovable(int v)
      : x(v)
  {}

  ThrowsImmovable(const ThrowsImmovable&) = delete;
  ThrowsImmovable(ThrowsImmovable&&)      = delete;
};

static_assert(cuda::__list_initializable_from<NothrowFromInt, int>);
static_assert(cuda::__nothrow_list_initializable_from<NothrowFromInt, int>);
static_assert(cuda::__list_initializable_from<ThrowsFromInt, int>);
static_assert(!cuda::__nothrow_list_initializable_from<ThrowsFromInt, int>);
static_assert(!cuda::__list_initializable_from<ThrowsFromInt, void*>);
static_assert(!cuda::__nothrow_list_initializable_from<ThrowsFromInt, void*>);

static_assert(cuda::__list_initializable_from<NothrowDefault>);
static_assert(cuda::__nothrow_list_initializable_from<NothrowDefault>);
static_assert(cuda::__list_initializable_from<ThrowsDefault>);
static_assert(!cuda::__nothrow_list_initializable_from<ThrowsDefault>);

static_assert(cuda::__list_initializable_from<Aggregate, int>);
static_assert(cuda::__nothrow_list_initializable_from<Aggregate, int>);
static_assert(cuda::__nothrow_initializable_from<Aggregate, int>);

// Floating-to-integer is a narrowing conversion, so list initialization is ill-formed.
static_assert(!cuda::__list_initializable_from<NothrowFromInt, double>);
static_assert(!cuda::__nothrow_list_initializable_from<NothrowFromInt, double>);

static_assert(cuda::__emplaceable_from<Immovable, NothrowFn>);
static_assert(cuda::__nothrow_emplaceable_from<Immovable, NothrowFn>);
static_assert(cuda::__emplaceable_from<Immovable, ThrowFn>);
static_assert(!cuda::__nothrow_emplaceable_from<Immovable, ThrowFn>);
static_assert(cuda::__emplaceable_from<Immovable, NothrowAddFn, int, int>);
static_assert(cuda::__nothrow_emplaceable_from<Immovable, NothrowAddFn, int, int>);
static_assert(!cuda::__emplaceable_from<Immovable, NothrowFn, int>);
static_assert(!cuda::__nothrow_emplaceable_from<Immovable, NothrowFn, int>);
static_assert(!cuda::__nothrow_emplaceable_from<Immovable, void*>);

static_assert(cuda::__emplaceable_from<ThrowsImmovable, NothrowFn>);
// MSVC and NVCC < 12.9 cannot test construction from a function result, so they only
// check that calling the function is non-throwing.
#if !TEST_COMPILER(MSVC) && !TEST_CUDA_COMPILER(NVCC, <, 12, 9)
static_assert(!cuda::__nothrow_emplaceable_from<ThrowsImmovable, NothrowFn>);
#endif // !TEST_COMPILER(MSVC) && !TEST_CUDA_COMPILER(NVCC, <, 12, 9)

int main(int, char**)
{
  return 0;
}
