//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// type_traits

// is_swappable

#include <cuda/std/type_traits>
// NOTE: These headers are not currently supported by libcu++.
// #include <cuda/std/utility>
// #include <cuda/std/vector>
#include "test_macros.h"

namespace MyNS
{
// Make the test types non-copyable so that generic cuda::std::swap is not valid.
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
{
  M(M const&)            = delete;
  M& operator=(M const&) = delete;
};

__host__ __device__ void swap(M&&, M&&) {}

struct DeletedSwap
{
  __host__ __device__ friend void swap(DeletedSwap&, DeletedSwap&) = delete;
};
} // namespace MyNS

#if 0 // We cannot test this because we need to avoid defining ambiguous overloads
namespace MyNS2
{

struct AmbiguousSwap
{};

template <class T>
__host__ __device__ void swap(T&, T&)
{}

} // end namespace MyNS2
#endif // Not implementable

int main(int, char**)
{
  using namespace MyNS;
  {
    // Test that is_swappable applies an lvalue reference to the type.
    static_assert(cuda::std::is_swappable<A>::value, "");
    static_assert(cuda::std::is_swappable<A&>::value, "");
    static_assert(!cuda::std::is_swappable<M>::value, "");
    static_assert(!cuda::std::is_swappable<M&&>::value, "");
  }
  static_assert(!cuda::std::is_swappable<B>::value, "");
  static_assert(cuda::std::is_swappable<C>::value, "");
  {
    // test non-referenceable types
    static_assert(!cuda::std::is_swappable<void>::value, "");
    static_assert(!cuda::std::is_swappable<int() const>::value, "");
    static_assert(!cuda::std::is_swappable<int() &>::value, "");
  }
  {
    // test that a deleted swap is correctly handled.
    static_assert(!cuda::std::is_swappable<DeletedSwap>::value, "");
  }
#if 0 // We cannot test this because we need to avoid defining ambiguous overloads
  {
    // test that a swap with ambiguous overloads is handled correctly.
    static_assert(!cuda::std::is_swappable<MyNS2::AmbiguousSwap>::value, "");
  }
#endif // Not implementable
  {
    // test for presence of is_swappable_v
    static_assert(cuda::std::is_swappable_v<int>, "");
    static_assert(!cuda::std::is_swappable_v<M>, "");
  }

  return 0;
}
