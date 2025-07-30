//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

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

template <typename T, bool isTrivial = cuda::std::is_trivially_default_constructible_v<T>>
struct test_trivial
{
  void operator()() const
  {
    [[maybe_unused]] constexpr cuda::std::istream_iterator<T> it;
  }
};

template <typename T>
struct test_trivial<T, false>
{
  void operator()() const {}
};

int main(int, char**)
{
  {
    typedef cuda::std::istream_iterator<int> T;
    T it;
    assert(it == T());
    [[maybe_unused]] constexpr T it2;
  }

  test_trivial<int>()();
  test_trivial<char>()();
  test_trivial<double>()();
  test_trivial<S>()();
  test_trivial<cuda::std::string>()();

  return 0;
}
#else
int main(int, char**)
{
  return 0;
}
#endif
