//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// Allocator-extended tuple constructors:
// - noexcept follows the uses-allocator constructor, not the ordinary one
// - default, element-pack, copy, move, and tuple-like constructors are constexpr
//
// Coverage follows libc++ tuple.cnstr/alloc*.pass.cpp and the MSVC STL tests
// P1032R1_miscellaneous_constexpr, P2165R4_tuple_like_tuple_members, and tr1 tuple t_alloc
// (allocator-arg-first and allocator-last, including pair and array sources).

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "../alloc_constexpr_types.h"
#include "allocators.h"
#include "test_macros.h"

// P2165R4: allocator constructors are constructible from tuple-like lvalues, const lvalues, rvalues, and const rvalues.
template <class Tuple, class Source>
TEST_FUNC constexpr void test_tuple_like_alloc_constructible()
{
  using AllocArg = cuda::std::allocator_arg_t;
  static_assert(cuda::std::is_constructible_v<Tuple, AllocArg, A1<int>, Source&>);
  static_assert(cuda::std::is_constructible_v<Tuple, AllocArg, A1<int>, const Source&>);
  static_assert(cuda::std::is_constructible_v<Tuple, AllocArg, A1<int>, Source>);
  static_assert(cuda::std::is_constructible_v<Tuple, AllocArg, A1<int>, const Source>);
}

TEST_FUNC constexpr bool test()
{
  A1<int> alloc{5};

  {
    cuda::std::tuple<> empty(cuda::std::allocator_arg, alloc);
    [[maybe_unused]] cuda::std::tuple<> empty_copy(cuda::std::allocator_arg, alloc, empty);
    [[maybe_unused]] cuda::std::tuple<> empty_move(cuda::std::allocator_arg, alloc, cuda::std::move(empty));
  }
  {
    cuda::std::tuple<constexpr_alloc_arg> def(cuda::std::allocator_arg, alloc);
    assert(cuda::std::get<0>(def).value == 1);
  }
  {
    cuda::std::tuple<constexpr_alloc_arg> from_int(cuda::std::allocator_arg, alloc, 7);
    cuda::std::tuple<constexpr_alloc_arg> copied(cuda::std::allocator_arg, alloc, from_int);
    assert(cuda::std::get<0>(copied).value == 7);
  }
  {
    cuda::std::tuple<constexpr_alloc_arg> from_int(cuda::std::allocator_arg, alloc, 7);
    cuda::std::tuple<constexpr_alloc_arg> moved(cuda::std::allocator_arg, alloc, cuda::std::move(from_int));
    assert(cuda::std::get<0>(moved).value == 7);
  }
  {
    cuda::std::tuple<int> ints{8};
    cuda::std::tuple<constexpr_alloc_arg> from_tuple(cuda::std::allocator_arg, alloc, ints);
    assert(cuda::std::get<0>(from_tuple).value == 8);
  }
  {
    const cuda::std::tuple<int> const_ints{9};
    cuda::std::tuple<constexpr_alloc_arg> from_const_tuple(cuda::std::allocator_arg, alloc, const_ints);
    assert(cuda::std::get<0>(from_const_tuple).value == 9);
  }
  {
    cuda::std::pair<int, int> pair_vals{3, 4};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_pair(cuda::std::allocator_arg, alloc, pair_vals);
    assert(cuda::std::get<0>(from_pair).value == 3);
    assert(cuda::std::get<1>(from_pair).value == 4);
  }
  {
    const cuda::std::pair<int, int> const_pair_vals{5, 6};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_const_pair(
      cuda::std::allocator_arg, alloc, const_pair_vals);
    assert(cuda::std::get<0>(from_const_pair).value == 5);
    assert(cuda::std::get<1>(from_const_pair).value == 6);
  }
  {
    cuda::std::array<int, 2> array_vals{1, 2};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_array(cuda::std::allocator_arg, alloc, array_vals);
    assert(cuda::std::get<0>(from_array).value == 1);
    assert(cuda::std::get<1>(from_array).value == 2);
  }
  {
    cuda::std::pair<int, int> rvalue_pair_vals{8, 9};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_rvalue_pair(
      cuda::std::allocator_arg, alloc, cuda::std::move(rvalue_pair_vals));
    assert(cuda::std::get<0>(from_rvalue_pair).value == 8);
    assert(cuda::std::get<1>(from_rvalue_pair).value == 9);
  }
  {
    const cuda::std::pair<int, int> const_rvalue_pair_vals{10, 11};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_const_rvalue_pair(
      cuda::std::allocator_arg, alloc, cuda::std::move(const_rvalue_pair_vals));
    assert(cuda::std::get<0>(from_const_rvalue_pair).value == 10);
    assert(cuda::std::get<1>(from_const_rvalue_pair).value == 11);
  }
  {
    cuda::std::array<int, 2> rvalue_array_vals{12, 13};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_rvalue_array(
      cuda::std::allocator_arg, alloc, cuda::std::move(rvalue_array_vals));
    assert(cuda::std::get<0>(from_rvalue_array).value == 12);
    assert(cuda::std::get<1>(from_rvalue_array).value == 13);
  }
  {
    const cuda::std::array<int, 2> const_rvalue_array_vals{14, 15};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg> from_const_rvalue_array(
      cuda::std::allocator_arg, alloc, cuda::std::move(const_rvalue_array_vals));
    assert(cuda::std::get<0>(from_const_rvalue_array).value == 14);
    assert(cuda::std::get<1>(from_const_rvalue_array).value == 15);
  }
  {
    cuda::std::array<int, 0> empty_array{};
    [[maybe_unused]] cuda::std::tuple<> from_empty_array(cuda::std::allocator_arg, alloc, empty_array);
  }
  {
    cuda::std::array<int, 1> one_array{16};
    cuda::std::tuple<constexpr_alloc_arg> from_one_array(cuda::std::allocator_arg, alloc, one_array);
    assert(cuda::std::get<0>(from_one_array).value == 16);
  }
  {
    cuda::std::array<int, 3> three_array{17, 18, 19};
    cuda::std::tuple<constexpr_alloc_arg, constexpr_alloc_arg, constexpr_alloc_arg> from_three_array(
      cuda::std::allocator_arg, alloc, three_array);
    assert(cuda::std::get<0>(from_three_array).value == 17);
    assert(cuda::std::get<1>(from_three_array).value == 18);
    assert(cuda::std::get<2>(from_three_array).value == 19);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_def(cuda::std::allocator_arg, alloc);
    assert(cuda::std::get<0>(last_def).value == 2);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_from_int(cuda::std::allocator_arg, alloc, 9);
    cuda::std::tuple<constexpr_alloc_last> last_copied(cuda::std::allocator_arg, alloc, last_from_int);
    assert(cuda::std::get<0>(last_copied).value == 9);
  }
  {
    cuda::std::tuple<constexpr_alloc_last> last_move_src(cuda::std::allocator_arg, alloc, 4);
    cuda::std::tuple<constexpr_alloc_last> last_moved(cuda::std::allocator_arg, alloc, cuda::std::move(last_move_src));
    assert(cuda::std::get<0>(last_moved).value == 4);
  }
  {
    cuda::std::pair<int, int> pair_vals{3, 4};
    cuda::std::tuple<constexpr_alloc_last, constexpr_alloc_last> last_from_pair(
      cuda::std::allocator_arg, alloc, pair_vals);
    assert(cuda::std::get<0>(last_from_pair).value == 3);
    assert(cuda::std::get<1>(last_from_pair).value == 4);
  }

  {
    test_tuple_like_alloc_constructible<cuda::std::tuple<>, cuda::std::array<int, 0>>();
    test_tuple_like_alloc_constructible<cuda::std::tuple<int>, cuda::std::array<int, 1>>();
    test_tuple_like_alloc_constructible<cuda::std::tuple<int, int>, cuda::std::array<int, 2>>();
    test_tuple_like_alloc_constructible<cuda::std::tuple<int, int, int>, cuda::std::array<int, 3>>();
    test_tuple_like_alloc_constructible<cuda::std::tuple<int, int>, cuda::std::pair<int, int>>();
  }
  {
    using Nothrow = cuda::std::tuple<nothrow_alloc_arg>;
    static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>>);
    static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>, int>);
    static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>, const Nothrow&>);
    static_assert(cuda::std::is_nothrow_constructible_v<Nothrow, cuda::std::allocator_arg_t, A1<int>, Nothrow>);
  }
  return true;
}

#if TEST_HAS_EXCEPTIONS() && _CCCL_HOST_COMPILATION()
void test_exceptions()
{
  using ThrowArg        = cuda::std::tuple<throw_on_alloc_arg>;
  using ThrowLast       = cuda::std::tuple<throw_on_alloc_last>;
  using FromComplex     = cuda::std::tuple<throw_on_alloc_arg, throw_on_alloc_arg>;
  using FromComplexLast = cuda::std::tuple<throw_on_alloc_last, throw_on_alloc_last>;
  using FromPair        = cuda::std::pair<int, int>;
  using FromArray       = cuda::std::array<int, 2>;
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>, int>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>, const ThrowArg&>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowArg, cuda::std::allocator_arg_t, A1<int>, ThrowArg>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>, int>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>, const ThrowLast&>);
  static_assert(!cuda::std::is_nothrow_constructible_v<ThrowLast, cuda::std::allocator_arg_t, A1<int>, ThrowLast>);
  static_assert(
    cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>&>);
  static_assert(
    cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, const cuda::std::complex<float>&>);
  static_assert(
    cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>&>);
  static_assert(
    !cuda::std::
      is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, const cuda::std::complex<float>&>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromPair&>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, const FromPair&>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromPair>);
  static_assert(!cuda::std::is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromPair>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromArray&>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, const FromArray&>);
  static_assert(cuda::std::is_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromArray>);
  static_assert(!cuda::std::is_nothrow_constructible_v<FromComplex, cuda::std::allocator_arg_t, A1<int>, FromArray>);
  static_assert(
    cuda::std::is_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(
    !cuda::std::
      is_nothrow_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, cuda::std::complex<float>>);
  static_assert(cuda::std::is_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, FromPair>);
  static_assert(!cuda::std::is_nothrow_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, FromPair>);
  static_assert(cuda::std::is_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, FromArray>);
  static_assert(
    !cuda::std::is_nothrow_constructible_v<FromComplexLast, cuda::std::allocator_arg_t, A1<int>, FromArray>);

  try
  {
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{}, 1);
    assert(false);
  }
  catch (int)
  {}

  try
  {
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{});
    assert(false);
  }
  catch (int)
  {}

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
    ThrowArg src{};
    [[maybe_unused]] ThrowArg t(cuda::std::allocator_arg, A1<int>{}, cuda::std::move(src));
    assert(false);
  }
  catch (int)
  {}

  try
  {
    [[maybe_unused]] FromComplex t(cuda::std::allocator_arg, A1<int>{}, cuda::std::complex<float>{1.f, 2.f});
    assert(false);
  }
  catch (int)
  {}

  try
  {
    FromPair p{3, 4};
    [[maybe_unused]] FromComplex t(cuda::std::allocator_arg, A1<int>{}, p);
    assert(false);
  }
  catch (int)
  {}

  try
  {
    FromArray a{1, 2};
    [[maybe_unused]] FromComplex t(cuda::std::allocator_arg, A1<int>{}, a);
    assert(false);
  }
  catch (int)
  {}

  try
  {
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{});
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

  try
  {
    FromComplexLast src{};
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, cuda::std::move(src));
    assert(false);
  }
  catch (int)
  {}

  try
  {
    FromArray a{1, 2};
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, a);
    assert(false);
  }
  catch (int)
  {}

  try
  {
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, cuda::std::complex<float>{1.f, 2.f});
    assert(false);
  }
  catch (int)
  {}

  try
  {
    FromPair p{3, 4};
    [[maybe_unused]] FromComplexLast t(cuda::std::allocator_arg, A1<int>{}, p);
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
