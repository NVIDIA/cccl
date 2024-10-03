//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// When the debug mode is enabled, we don't unwrap iterators in cuda::std::copy
// so we don't get this optimization.
// UNSUPPORTED: libcpp-has-debug-mode

// check that cuda::std::__unwrap_iter() returns the correct type

// #include <cuda/std/algorithm>
#include <cuda/std/cassert>
#ifdef _LIBCUDACXX_HAS_STRING
#  include <cuda/std/string>
#endif
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_iterators.h"
#include "test_macros.h"

template <class Iter>
using UnwrapT = decltype(cuda::std::__unwrap_iter(cuda::std::declval<Iter>()));

template <class Iter>
using rev_iter = cuda::std::reverse_iterator<Iter>;

template <class Iter>
using rev_rev_iter = rev_iter<rev_iter<Iter>>;

static_assert(cuda::std::is_same<UnwrapT<int*>, int*>::value, "");
static_assert(cuda::std::is_same<UnwrapT<cuda::std::__wrap_iter<int*>>, int*>::value, "");
static_assert(cuda::std::is_same<UnwrapT<rev_iter<int*>>, cuda::std::reverse_iterator<int*>>::value, "");
static_assert(cuda::std::is_same<UnwrapT<rev_rev_iter<int*>>, int*>::value, "");
static_assert(cuda::std::is_same<UnwrapT<rev_rev_iter<cuda::std::__wrap_iter<int*>>>, int*>::value, "");
static_assert(cuda::std::is_same<UnwrapT<rev_rev_iter<rev_iter<cuda::std::__wrap_iter<int*>>>>,
                                 rev_iter<cuda::std::__wrap_iter<int*>>>::value,
              "");

static_assert(cuda::std::is_same<UnwrapT<random_access_iterator<int*>>, random_access_iterator<int*>>::value, "");
static_assert(
  cuda::std::is_same<UnwrapT<rev_iter<random_access_iterator<int*>>>, rev_iter<random_access_iterator<int*>>>::value,
  "");
static_assert(
  cuda::std::is_same<UnwrapT<rev_rev_iter<random_access_iterator<int*>>>, random_access_iterator<int*>>::value, "");
static_assert(cuda::std::is_same<UnwrapT<rev_rev_iter<rev_iter<random_access_iterator<int*>>>>,
                                 rev_iter<random_access_iterator<int*>>>::value,
              "");

#ifdef _LIBCUDACXX_HAS_STRING
TEST_CONSTEXPR_CXX20 bool test()
{
  cuda::std::string str = "Banane";
  using Iter            = cuda::std::string::iterator;

  assert(cuda::std::__unwrap_iter(str.begin()) == str.data());
  assert(cuda::std::__unwrap_iter(str.end()) == str.data() + str.size());
  assert(cuda::std::__unwrap_iter(rev_rev_iter<Iter>(rev_iter<Iter>(str.begin()))) == str.data());
  assert(cuda::std::__unwrap_iter(rev_rev_iter<Iter>(rev_iter<Iter>(str.end()))) == str.data() + str.size());

  return true;
}
#endif // _LIBCUDACXX_HAS_STRING

int main(int, char**)
{
#ifdef _LIBCUDACXX_HAS_STRING
  test();
#  if TEST_STD_VER > 2017
  static_assert(test());
#  endif
#endif

  return 0;
}
