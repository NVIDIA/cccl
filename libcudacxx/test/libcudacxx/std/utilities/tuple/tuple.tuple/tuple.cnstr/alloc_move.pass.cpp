//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc>
//   tuple(allocator_arg_t, const Alloc& a, tuple&&);

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "MoveOnly.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    cuda::std::tuple<> empty(cuda::std::allocator_arg, alloc);
    [[maybe_unused]] cuda::std::tuple<> empty_move(cuda::std::allocator_arg, alloc, cuda::std::move(empty));
  }
  {
    cuda::std::tuple<constexpr_alloc_arg> from_int(cuda::std::allocator_arg, alloc, 7);
    cuda::std::tuple<constexpr_alloc_arg> moved(cuda::std::allocator_arg, alloc, cuda::std::move(from_int));
    assert(cuda::std::get<0>(moved).value == 7);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_move_src(cuda::std::allocator_arg, alloc, 4);
    cuda::std::tuple<constexpr_alloc_last> last_moved(cuda::std::allocator_arg, alloc, cuda::std::move(last_move_src));
    assert(cuda::std::get<0>(last_moved).value == 4);
  }
  {
    cuda::std::tuple<MoveOnly> src(MoveOnly(0));
    cuda::std::tuple<MoveOnly> moved(cuda::std::allocator_arg, alloc, cuda::std::move(src));
    assert(cuda::std::get<0>(moved) == 0);
  }
#ifdef _CUDA_STD_VERSION
  {
    cuda::std::tuple<MoveOnly, constexpr_alloc_arg> src(cuda::std::allocator_arg, alloc, MoveOnly(0), 1);
    cuda::std::tuple<MoveOnly, constexpr_alloc_arg> moved(cuda::std::allocator_arg, alloc, cuda::std::move(src));
    assert(cuda::std::get<0>(moved) == 0);
    assert(cuda::std::get<1>(moved).value == 1);
  }
  {
    cuda::std::tuple<MoveOnly, constexpr_alloc_arg, constexpr_alloc_last> src(
      cuda::std::allocator_arg, alloc, MoveOnly(1), 2, 3);
    cuda::std::tuple<MoveOnly, constexpr_alloc_arg, constexpr_alloc_last> moved(
      cuda::std::allocator_arg, alloc, cuda::std::move(src));
    assert(cuda::std::get<0>(moved) == 1);
    assert(cuda::std::get<1>(moved).value == 2);
    assert(cuda::std::get<2>(moved).value == 3);
  }
#endif // _CUDA_STD_VERSION
  return true;
}

using Nothrow = cuda::std::tuple<nothrow_alloc_arg>;
static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>, Nothrow>);

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using ThrowArg        = cuda::std::tuple<throw_on_alloc_arg>;
  using ThrowLast       = cuda::std::tuple<throw_on_alloc_last>;
  using FromComplexLast = cuda::std::tuple<throw_on_alloc_last, throw_on_alloc_last>;
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>, ThrowArg>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>, ThrowLast>);

  try
  {
    ThrowArg src{};
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{}, cuda::std::move(src));
    assert(false);
  }
  catch (int)
  {}

  try
  {
    FromComplexLast src{};
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, cuda::std::move(src));
    assert(false);
  }
  catch (int)
  {}
}
#endif // TEST_HAS_EXCEPTIONS()

int main(int, char**)
{
  test();
  static_assert(test());
#if TEST_HAS_EXCEPTIONS()
  NV_IF_TARGET(NV_IS_HOST, (test_exceptions();))
#endif // TEST_HAS_EXCEPTIONS()
  return 0;
}
