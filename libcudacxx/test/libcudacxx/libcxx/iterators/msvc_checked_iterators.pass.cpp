//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// The iterators of the contiguous containers of the MSVC STL are contiguous iterators, and to_address does not
// dereference them, because with _ITERATOR_DEBUG_LEVEL != 0 they verify that they are dereferenceable.

// ADDITIONAL_COMPILE_DEFINITIONS: _ITERATOR_DEBUG_LEVEL=1

#include <cuda/std/cassert>
#include <cuda/std/iterator>
#include <cuda/std/memory>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <deque>
#  include <string>
#  include <string_view>
#  include <vector>
#  ifdef __cpp_lib_span
#    include <span>
#  endif // __cpp_lib_span
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

#if _CCCL_HAS_HOST_STD_LIB()
static_assert(cuda::std::contiguous_iterator<std::vector<int>::iterator>);
static_assert(cuda::std::contiguous_iterator<std::vector<int>::const_iterator>);
static_assert(cuda::std::contiguous_iterator<std::string::iterator>);
static_assert(cuda::std::contiguous_iterator<std::string::const_iterator>);
static_assert(cuda::std::contiguous_iterator<std::string_view::const_iterator>);
static_assert(cuda::std::contiguous_iterator<std::array<int, 3>::iterator>);
static_assert(cuda::std::contiguous_iterator<std::array<int, 3>::const_iterator>);
#  ifdef __cpp_lib_span
static_assert(cuda::std::contiguous_iterator<std::span<int>::iterator>);
#  endif // __cpp_lib_span
static_assert(!cuda::std::contiguous_iterator<std::deque<int>::iterator>);

template <class C>
constexpr void test_container(C c)
{
  assert(cuda::std::to_address(c.begin()) == c.data());
  assert(cuda::std::to_address(c.end()) == c.data() + c.size());
  assert(cuda::std::to_address(cuda::std::as_const(c).begin()) == cuda::std::as_const(c).data());
  assert(cuda::std::to_address(cuda::std::as_const(c).end()) == cuda::std::as_const(c).data() + c.size());
}

constexpr bool test_constexpr()
{
  test_container(std::array<int, 3>{});
  test_container(std::string_view{"abc"});
#  ifdef __cpp_lib_span
  int arr[3]{};
  test_container(std::span<int>{arr});
#  endif // __cpp_lib_span
  return true;
}
static_assert(test_constexpr());

void test()
{
  test_constexpr();
  test_container(std::array<int, 0>{});
  test_container(std::vector<int>(3));
  test_container(std::vector<int>{});
  test_container(std::string{"abc"});
  test_container(std::string{});
}
#endif // _CCCL_HAS_HOST_STD_LIB()

int main(int, char**)
{
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, (test();))
#endif // _CCCL_HAS_HOST_STD_LIB()
  return 0;
}
