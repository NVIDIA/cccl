//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// template <class T, class U, class... Args>
//   constexpr optional<T> make_optional(initializer_list<U> il, Args&&... args);

#include <cuda/std/cassert>
// #include <cuda/std/memory>
#include <cuda/std/optional>
// #include <cuda/std/string>

#include "test_macros.h"

struct TestT
{
  int x;
  int size;
  int* ptr;
  __host__ __device__ constexpr TestT(cuda::std::initializer_list<int> il)
      : x(*il.begin())
      , size(static_cast<int>(il.size()))
      , ptr(nullptr)
  {}
  __host__ __device__ constexpr TestT(cuda::std::initializer_list<int> il, int* p)
      : x(*il.begin())
      , size(static_cast<int>(il.size()))
      , ptr(p)
  {}
};

__host__ __device__ constexpr bool test()
{
  {
    auto opt = cuda::std::make_optional<TestT>({42, 2, 3});
    ASSERT_SAME_TYPE(decltype(opt), cuda::std::optional<TestT>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == nullptr);
  }
  {
    int i    = 42;
    auto opt = cuda::std::make_optional<TestT>({42, 2, 3}, &i);
    ASSERT_SAME_TYPE(decltype(opt), cuda::std::optional<TestT>);
    assert(opt->x == 42);
    assert(opt->size == 3);
    assert(opt->ptr == &i);
  }
  return true;
}

int main(int, char**)
{
  test();
#if defined(_CCCL_BUILTIN_ADDRESSOF)
#  if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
  static_assert(test(), "");
#  endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
#endif
  /*
  {
    auto opt = cuda::std::make_optional<cuda::std::string>({'1', '2', '3'});
    assert(*opt == "123");
  }
  {
    auto opt = cuda::std::make_optional<cuda::std::string>({'a', 'b', 'c'}, cuda::std::allocator<char>{});
    assert(*opt == "abc");
  }
  */
  return 0;
}
