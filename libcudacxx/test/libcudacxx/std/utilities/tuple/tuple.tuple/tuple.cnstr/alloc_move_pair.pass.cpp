//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class U1, class U2>
//   tuple(allocator_arg_t, const Alloc& a, pair<U1, U2>&&);
// In tile mode virtual functions are unsupported

#include <cuda/std/__memory_>
#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#include "../alloc_constexpr_types.h"
#include "../alloc_first.h"
#include "allocators.h"
#include "MoveOnly.h"
#include "test_macros.h"

#if !_CCCL_TILE_COMPILATION()
struct B
{
  int id_;

  TEST_HOST_DEVICE_FUNC explicit B(int i)
      : id_(i)
  {}

  TEST_HOST_DEVICE_FUNC virtual ~B() {}
};

struct D : B
{
  TEST_HOST_DEVICE_FUNC explicit D(int i)
      : B(i)
  {}
};
#endif // !_CCCL_TILE_COMPILATION()

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    cuda::std::pair<int, int> p{8, 9};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, cuda::std::move(p));
    assert(cuda::std::get<0>(t).value == 8);
    assert(cuda::std::get<1>(t).value == 9);
  }
  {
    const cuda::std::pair<int, int> p{10, 11};
    cuda::std::tuple<constexpr_alloc_last, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, cuda::std::move(p));
    assert(cuda::std::get<0>(t).value == 10);
    assert(cuda::std::get<1>(t).value == 11);
  }
  {
    cuda::std::pair<int, MoveOnly> p{2, MoveOnly(3)};
    cuda::std::tuple<constexpr_alloc_arg, MoveOnly> t(cuda::std::allocator_arg, alloc, cuda::std::move(p));
    assert(cuda::std::get<0>(t).value == 2);
    assert(cuda::std::get<1>(t) == 3);
  }
  return true;
}

TEST_FUNC void test_runtime()
{
#if !_CCCL_TILE_COMPILATION()
  {
    using T0 = cuda::std::pair<int, cuda::std::unique_ptr<D>>;
    using T1 = cuda::std::tuple<alloc_first, cuda::std::unique_ptr<B>>;
    T0 t0(2, cuda::std::unique_ptr<D>(new D(3)));
    alloc_first::allocator_constructed() = false;
    T1 t1(cuda::std::allocator_arg, A1<int>(5), cuda::std::move(t0));
    assert(alloc_first::allocator_constructed());
    assert(cuda::std::get<0>(t1) == 2);
    assert(cuda::std::get<1>(t1)->id_ == 3);
  }
#endif // !_CCCL_TILE_COMPILATION()
}

int main(int, char**)
{
  test();
  static_assert(test());
  test_runtime();
  return 0;
}
