//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: gcc-6

// <memory>

// template <class T>
// constexpr void destroy_at(T*);

// #include <cuda/std/memory>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/type_traits>

#include "test_macros.h"

struct Counted
{
  int* counter_;
  __host__ __device__ constexpr Counted(int* counter)
      : counter_(counter)
  {
    ++*counter_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~Counted()
  {
    --*counter_;
  }
  __host__ __device__ friend void operator&(Counted) = delete;
};

struct VirtualCounted
{
  int* counter_;
  __host__ __device__ constexpr VirtualCounted(int* counter)
      : counter_(counter)
  {
    ++*counter_;
  }
  __host__ __device__ TEST_CONSTEXPR_CXX20 virtual ~VirtualCounted()
  {
    --*counter_;
  }
  __host__ __device__ void operator&() const = delete;
};

struct DerivedCounted : VirtualCounted
{
  __host__ __device__ constexpr DerivedCounted(int* counter)
      : VirtualCounted(counter)
  {}
  __host__ __device__ TEST_CONSTEXPR_CXX20 ~DerivedCounted() override {}
};

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test_arrays()
{
  {
    int counter    = 0;
    Counted arr[3] = {{&counter}, {&counter}, {&counter}};
    assert(counter == 3);

    cuda::std::destroy_at(&arr);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy_at(&arr)), void>);
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    for (int i = 0; i < 3; ++i)
    {
      cuda::std::__construct_at(arr + i, &counter);
    }
  }
  {
    int counter       = 0;
    Counted arr[3][2] = {{{&counter}, {&counter}}, {{&counter}, {&counter}}, {{&counter}, {&counter}}};
    assert(counter == 3 * 2);

    cuda::std::destroy_at(&arr);
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy_at(&arr)), void>);
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    for (int i = 0; i < 3; ++i)
    {
      for (int j = 0; j < 2; ++j)
      {
        cuda::std::__construct_at(arr[i] + j, &counter);
      }
    }
  }
  return true;
}

__host__ __device__ TEST_CONSTEXPR_CXX20 bool test()
{
  {
    int counter = 0;
    Counted first{&counter};
    Counted second{&counter};
    assert(counter == 2);

    cuda::std::destroy_at(cuda::std::addressof(first));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy_at(cuda::std::addressof(first))), void>);
    assert(counter == 1);

    cuda::std::destroy_at(cuda::std::addressof(second));
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    cuda::std::__construct_at(cuda::std::addressof(first), &counter);
    cuda::std::__construct_at(cuda::std::addressof(second), &counter);
  }
  {
    int counter = 0;
    DerivedCounted first{&counter};
    DerivedCounted second{&counter};
    assert(counter == 2);

    cuda::std::destroy_at(cuda::std::addressof(first));
    static_assert(cuda::std::is_same_v<decltype(cuda::std::destroy_at(cuda::std::addressof(first))), void>);
    assert(counter == 1);

    cuda::std::destroy_at(cuda::std::addressof(second));
    assert(counter == 0);

    // We need to reconstruct here, as we are dealing with stack variables and they get destroyed at the end of scope
    cuda::std::__construct_at(cuda::std::addressof(first), &counter);
    cuda::std::__construct_at(cuda::std::addressof(second), &counter);
  }

  return true;
}

int main(int, char**)
{
  test();
  test_arrays();
#if TEST_STD_VER > 2017
#  if !TEST_COMPILER(NVRTC)
#    if TEST_COMPILER(CLANG, >, 10) || (TEST_COMPILER(GCC, >, 9) && TEST_COMPILER(GCC, <, 14)) \
      || TEST_COMPILER(MSVC2022) || TEST_COMPILER(NVHPC)
  static_assert(test());
  // TODO: Until cuda::std::__construct_at has support for arrays, it's impossible to test this
  //       in a constexpr context (see https://reviews.llvm.org/D114903).
  // static_assert(test_arrays());
#    endif
#  endif // TEST_COMPILER(NVRTC)
#endif // TEST_STD_VER > 2017
  return 0;
}
