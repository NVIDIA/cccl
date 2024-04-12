//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++98, c++03, c++11

// type_traits

// is_swappable_with

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

__host__ __device__ void swap(A&, B&) {}
__host__ __device__ void swap(B&, A&) {}

__host__ __device__ void swap(A&, C&) {} // missing swap(C, A)
__host__ __device__ void swap(D&, C&) {}

struct M
{};

__host__ __device__ void swap(M&&, M&&) {}

} // namespace MyNS

int main(int, char**)
{
  using namespace MyNS;
  {
    // Test that is_swappable_with doesn't apply an lvalue reference
    // to the type. Instead it is up to the user.
    static_assert(!cuda::std::is_swappable_with<int, int>::value, "");
    static_assert(cuda::std::is_swappable_with<int&, int&>::value, "");
    static_assert(cuda::std::is_swappable_with<M, M>::value, "");
    static_assert(cuda::std::is_swappable_with<A&, A&>::value, "");
  }
  {
    // test that heterogeneous swap is allowed only if both 'swap(A, B)' and
    // 'swap(B, A)' are valid.
    static_assert(cuda::std::is_swappable_with<A&, B&>::value, "");
    static_assert(!cuda::std::is_swappable_with<A&, C&>::value, "");
    static_assert(!cuda::std::is_swappable_with<D&, C&>::value, "");
  }
  {
    // test that cv void is guarded against as required.
    static_assert(!cuda::std::is_swappable_with_v<void, int>, "");
    static_assert(!cuda::std::is_swappable_with_v<int, void>, "");
    static_assert(!cuda::std::is_swappable_with_v<const void, const volatile void>, "");
  }
  {
    // test for presence of is_swappable_with_v
    static_assert(cuda::std::is_swappable_with_v<int&, int&>, "");
    static_assert(!cuda::std::is_swappable_with_v<D&, C&>, "");
  }

  return 0;
}
