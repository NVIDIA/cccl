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
//   tuple(allocator_arg_t, const Alloc& a, const tuple&);

#include <cuda/std/cassert>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    cuda::std::tuple<> empty(cuda::std::allocator_arg, alloc);
    [[maybe_unused]] cuda::std::tuple<> empty_copy(cuda::std::allocator_arg, alloc, empty);
  }
  {
    cuda::std::tuple<constexpr_alloc_arg> from_int(cuda::std::allocator_arg, alloc, 7);
    cuda::std::tuple<constexpr_alloc_arg> copied(cuda::std::allocator_arg, alloc, from_int);
    assert(cuda::std::get<0>(copied).value == 7);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_from_int(cuda::std::allocator_arg, alloc, 9);
    cuda::std::tuple<constexpr_alloc_last> last_copied(cuda::std::allocator_arg, alloc, last_from_int);
    assert(cuda::std::get<0>(last_copied).value == 9);
  }
  {
    cuda::std::tuple<int> src(2);
    cuda::std::tuple<int> copied(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(copied) == 2);
  }
#ifdef _CUDA_STD_VERSION
  {
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_last> src(cuda::std::allocator_arg, alloc, 2, 3);
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_last> copied(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(copied).value == 2);
    assert(cuda::std::get<1>(copied).value == 3);
  }
  {
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> src(cuda::std::allocator_arg, alloc, 1, 2, 3);
    cuda::std::tuple<int, constexpr_alloc_arg, constexpr_alloc_last> copied(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(copied) == 1);
    assert(cuda::std::get<1>(copied).value == 2);
    assert(cuda::std::get<2>(copied).value == 3);
  }
#endif // _CUDA_STD_VERSION
  return true;
}

using Nothrow = cuda::std::tuple<nothrow_alloc_arg>;
static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>, const Nothrow&>);

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using ThrowArg        = cuda::std::tuple<throw_on_alloc_arg>;
  using ThrowLast       = cuda::std::tuple<throw_on_alloc_last>;
  using FromComplexLast = cuda::std::tuple<throw_on_alloc_last, throw_on_alloc_last>;
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>, const ThrowArg&>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>, const ThrowLast&>);

  try
  {
    ThrowArg src{};
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{}, src);
    assert(false);
  }
  catch (int)
  {}

  try
  {
    FromComplexLast src{};
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, src);
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
