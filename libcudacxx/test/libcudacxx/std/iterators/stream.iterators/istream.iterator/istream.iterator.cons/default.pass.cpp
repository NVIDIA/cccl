//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Usage of is_trivially_constructible is broken with these compilers.
// See https://bugs.llvm.org/show_bug.cgi?id=31016
// XFAIL: clang-3.7, apple-clang-7 && c++17

// <cuda/std/iterator>

// class istream_iterator

// constexpr istream_iterator();
// C++17 says: If is_trivially_default_constructible_v<T> is true, then this
//    constructor is a constexpr constructor.

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#if defined(_LIBCUDACXX_HAS_STRING)
#  include <cuda/std/string>

#  include "test_macros.h"

struct S
{
  S();
}; // not constexpr

#  if TEST_STD_VER > 2014
template <typename T, bool isTrivial = cuda::std::is_trivially_default_constructible_v<T>>
struct test_trivial
{
  void operator()() const
  {
    constexpr cuda::std::istream_iterator<T> it;
    (void) it;
  }
};

template <typename T>
struct test_trivial<T, false>
{
  void operator()() const {}
};
#  endif

int main(int, char**)
{
  {
    typedef cuda::std::istream_iterator<int> T;
    T it;
    assert(it == T());
    constexpr T it2;
    (void) it2;
  }

#  if TEST_STD_VER > 2014
  test_trivial<int>()();
  test_trivial<char>()();
  test_trivial<double>()();
  test_trivial<S>()();
  test_trivial<cuda::std::string>()();
#  endif

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
