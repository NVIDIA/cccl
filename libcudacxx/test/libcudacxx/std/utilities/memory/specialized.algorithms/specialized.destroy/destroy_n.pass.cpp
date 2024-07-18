//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: gcc-6

// <memory>

// template <class ForwardIt, class Size>
// constexpr ForwardIt destroy_n(ForwardIt, Size s);

// #include <cuda/std/memory>
#include <cuda/std/cassert>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

struct Counted
{
  int* counter_ = nullptr;
  __host__ __device__ TEST_CONSTEXPR Counted(int* counter)
      : counter_(counter)
  {
    ++*counter_;
  }
  __host__ __device__ TEST_CONSTEXPR Counted(Counted const& other)
      : counter_(other.counter_)
  {
    ++*counter_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~Counted()
  {
    --*counter_;
  }
  __host__ __device__ friend void operator&(Counted) = delete;
};

#if TEST_STD_VER > 2017
__host__ __device__ constexpr bool test_arrays()
{
  {
    int counter     = 0;
    Counted pool[3] = {{&counter}, {&counter}, {&counter}};
    assert(counter == 3);

    Counted* p = cuda::std::destroy_n(pool, 3);
    ASSERT_SAME_TYPE(decltype(cuda::std::destroy_n(pool, 3)), Counted*);
    assert(p == pool + 3);
    assert(counter == 0);
  }
  {
    using Array   = Counted[2];
    int counter   = 0;
    Array pool[3] = {{{&counter}, {&counter}}, {{&counter}, {&counter}}, {{&counter}, {&counter}}};
    assert(counter == 3 * 2);

    Array* p = cuda::std::destroy_n(pool, 3);
    ASSERT_SAME_TYPE(decltype(cuda::std::destroy_n(pool, 3)), Array*);
    assert(p == pool + 3);
    assert(counter == 0);
  }

  return true;
}
#endif

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX20 void test()
{
  int counter = 0;

  Counted pool[5] = {{&counter}, {&counter}, {&counter}, {&counter}, {&counter}};
  assert(counter == 5);

  It it = cuda::std::destroy_n(It(pool), 5);
  ASSERT_SAME_TYPE(decltype(cuda::std::destroy_n(It(pool), 5)), It);
  assert(it == It(pool + 5));
  assert(counter == 0);

  // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
  for (int i = 0; i < 5; ++i)
  {
    cuda::std::__construct_at(pool + i, &counter);
  }
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool tests()
{
  test<Counted*>();
  test<forward_iterator<Counted*>>();
  return true;
}

int main(int, char**)
{
  tests();
#if TEST_STD_VER > 2017
  test_arrays();
#  if !defined(TEST_COMPILER_NVRTC)
#    if (defined(TEST_COMPILER_CLANG) && __clang_major__ > 10) || (defined(TEST_COMPILER_GCC) && __GNUC__ > 9) \
      || defined(TEST_COMPILER_MSVC_2022) || defined(TEST_COMPILER_NVHPC)
  static_assert(tests());
  // TODO: Until cuda::std::__construct_at has support for arrays, it's impossible to test this
  //       in a constexpr context (see https://reviews.llvm.org/D114903).
  // static_assert(test_arrays());
#    endif
#  endif // TEST_COMPILER_NVRTC
#endif // TEST_STD_VER > 2017
  return 0;
}
