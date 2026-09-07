//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <memory>

// template <class T> constexpr T* to_address(T* p) noexcept;
// template <class Ptr> constexpr auto to_address(const Ptr& p) noexcept;

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/memory>
#include <cuda/std/span>
#include <cuda/std/string_view>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <string>
#  include <string_view>
#  include <vector>
#  ifdef __cpp_lib_span
#    include <span>
#  endif // __cpp_lib_span
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "test_macros.h"

_CCCL_DIAG_SUPPRESS_NVCC(3215) // "if consteval" and "if not consteval" are not standard in this mode

template <class C>
TEST_FUNC constexpr void test_container_iterators(C c)
{
  assert(cuda::std::to_address(c.begin()) == c.data());
  assert(cuda::std::to_address(c.end()) == c.data() + c.size());
  assert(cuda::std::to_address(cuda::std::as_const(c).begin()) == cuda::std::as_const(c).data());
  assert(cuda::std::to_address(cuda::std::as_const(c).end())
         == cuda::std::as_const(c).data() + cuda::std::as_const(c).size());
}

TEST_FUNC constexpr bool test()
{
  test_container_iterators(cuda::std::array<int, 3>());
  test_container_iterators(cuda::std::string_view("abc"));
  test_container_iterators(cuda::std::span<const char>("abc"));

#if _CCCL_HAS_HOST_STD_LIB()
#  if defined(__cpp_lib_span)
  NV_IF_TARGET(NV_IS_HOST, ({ test_container_iterators(std::span<const char>("abc")); }))
#  endif //__cpp_lib_span
  NV_IF_TARGET(NV_IS_HOST, ({
                 test_container_iterators(std::array<int, 3>());
                 test_container_iterators(std::string_view("abc"));
               }))
  if (!cuda::std::__cccl_default_is_constant_evaluated())
  {
    NV_IF_TARGET(NV_IS_HOST, ({
                   test_container_iterators(std::vector<int>(3));
                   test_container_iterators(std::string("abc"));
                 }))
  }
#endif // _CCCL_HAS_HOST_STD_LIB()
  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
