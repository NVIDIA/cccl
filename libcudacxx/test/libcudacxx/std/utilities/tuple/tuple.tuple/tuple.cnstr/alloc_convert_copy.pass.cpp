//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class Alloc, class... UTypes>
//   tuple(allocator_arg_t, const Alloc& a, const tuple<UTypes...>&);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "test_macros.h"

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};
  {
    const cuda::std::tuple<int> const_ints{9};
    cuda::std::tuple<constexpr_alloc_arg> from_const_tuple(cuda::std::allocator_arg, alloc, const_ints);
    assert(cuda::std::get<0>(from_const_tuple).value == 9);
  }
  {
    const cuda::std::array<int, 2> const_array{1, 2};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_const_array(
      cuda::std::allocator_arg, alloc, const_array);
    assert(cuda::std::get<0>(from_const_array).value == 1);
    assert(cuda::std::get<1>(from_const_array).value == 2);
  }
  {
    const cuda::std::tuple<long> src(2);
    cuda::std::tuple<long long> t(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(t) == 2);
  }
  {
    const cuda::std::tuple<int> src(2);
    cuda::std::tuple<constexpr_alloc_arg> t(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(t).value == 2);
  }
  {
    const cuda::std::tuple<int, int> src(2, 3);
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(t).value == 2);
    assert(cuda::std::get<1>(t).value == 3);
  }
  {
    const cuda::std::tuple<long, int, int> src(1, 2, 3);
    cuda::std::tuple<long long, constexpr_alloc_arg, constexpr_alloc_last> t(cuda::std::allocator_arg, alloc, src);
    assert(cuda::std::get<0>(t) == 1);
    assert(cuda::std::get<1>(t).value == 2);
    assert(cuda::std::get<2>(t).value == 3);
  }
  static_assert(
    cuda::std::
      is_constructible_v<cuda::std::tuple<>, cuda::std::allocator_arg_t, A1<int>, const cuda::std::array<int, 0>&>);
  static_assert(
    cuda::std::
      is_constructible_v<cuda::std::tuple<int>, cuda::std::allocator_arg_t, A1<int>, const cuda::std::array<int, 1>&>);
  static_assert(cuda::std::is_constructible_v<cuda::std::tuple<int, int>,
                                              cuda::std::allocator_arg_t,
                                              A1<int>,
                                              const cuda::std::array<int, 2>&>);
  static_assert(cuda::std::is_constructible_v<cuda::std::tuple<int, int, int>,
                                              cuda::std::allocator_arg_t,
                                              A1<int>,
                                              const cuda::std::array<int, 3>&>);
  return true;
}

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using FromComplex = cuda::std::tuple<throw_on_alloc_arg, throw_on_alloc_arg>;
  static_assert(
    cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, const cuda::std::complex<float>&>);
  static_assert(
    !cuda::std::
      is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, const cuda::std::complex<float>&>);

  try
  {
    const cuda::std::complex<float> src{1.f, 2.f};
    [[maybe_unused]] FromComplex t(cuda::std::allocator_arg, A1<int>{}, src);
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
