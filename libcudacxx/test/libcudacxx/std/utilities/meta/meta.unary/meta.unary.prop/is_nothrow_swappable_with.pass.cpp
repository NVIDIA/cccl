//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// type_traits

// is_nothrow_swappable_with

#include <cuda/std/type_traits>
// NOTE: This header is not currently supported by libcu++.
// #include <cuda/std/vector>
#include "test_macros.h"

namespace MyNS
{

struct A
{
  A(A const&)            = delete;
  A& operator=(A const&) = delete;
};

struct B
{
  B(B const&)            = delete;
  B& operator=(B const&) = delete;
};

struct C
{};
struct D
{};

__host__ __device__ void swap(A&, A&) {}

__host__ __device__ void swap(A&, B&) noexcept {}
__host__ __device__ void swap(B&, A&) noexcept {}

__host__ __device__ void swap(A&, C&) noexcept {}
__host__ __device__ void swap(C&, A&) {}

struct M
{};

__host__ __device__ void swap(M&&, M&&) noexcept {}

} // namespace MyNS

int main(int, char**)
{
  using namespace MyNS;
  {
    // Test that is_swappable_with doesn't apply an lvalue reference
    // to the type. Instead it is up to the user.
    static_assert(!cuda::std::is_nothrow_swappable_with<int, int>::value, "");
    static_assert(cuda::std::is_nothrow_swappable_with<int&, int&>::value, "");
    static_assert(cuda::std::is_nothrow_swappable_with<M, M>::value, "");
#ifndef TEST_COMPILER_ICC
    static_assert(cuda::std::is_swappable_with<A&, A&>::value && !cuda::std::is_nothrow_swappable_with<A&, A&>::value,
                  "");
#endif // TEST_COMPILER_ICC
  }
  {
    // test that heterogeneous swap is allowed only if both 'swap(A, B)' and
    // 'swap(B, A)' are valid.
    static_assert(cuda::std::is_nothrow_swappable_with<A&, B&>::value, "");
#ifndef TEST_COMPILER_ICC
    static_assert(!cuda::std::is_nothrow_swappable_with<A&, C&>::value && cuda::std::is_swappable_with<A&, C&>::value,
                  "");
#endif // TEST_COMPILER_ICC
    static_assert(!cuda::std::is_nothrow_swappable_with<D&, C&>::value, "");
  }
  {
    // test we guard against cv void inputs as required.
    static_assert(!cuda::std::is_nothrow_swappable_with_v<void, int>, "");
    static_assert(!cuda::std::is_nothrow_swappable_with_v<int, void>, "");
    static_assert(!cuda::std::is_nothrow_swappable_with_v<const void, const volatile void>, "");
  }
  {
    // test for presence of is_nothrow_swappable_with_v
    static_assert(cuda::std::is_nothrow_swappable_with_v<int&, int&>, "");
    static_assert(!cuda::std::is_nothrow_swappable_with_v<int&&, int&&>, "");
  }

  return 0;
}
