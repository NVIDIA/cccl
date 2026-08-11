//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class TupleLike>
//   tuple& operator=(TupleLike&& u);

#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/tuple>
#include <cuda/std/utility>

#if _CCCL_HAS_HOST_STD_LIB()
#  include <array>
#  include <complex>
#  include <tuple>
#  include <utility>
#endif // _CCCL_HAS_HOST_STD_LIB()

#include "copy_move_types.h"
#include "test_macros.h"

struct NonAssignable
{
  NonAssignable& operator=(NonAssignable const&) = delete;
  NonAssignable& operator=(NonAssignable&&)      = delete;
};
struct CopyAssignable
{
  CopyAssignable& operator=(CopyAssignable const&) = default;
  CopyAssignable& operator=(CopyAssignable&&)      = delete;
};
static_assert(cuda::std::is_copy_assignable<CopyAssignable>::value);
struct MoveAssignable
{
  MoveAssignable& operator=(MoveAssignable const&) = delete;
  MoveAssignable& operator=(MoveAssignable&&)      = default;
};

TEST_FUNC constexpr bool test()
{
  {
    const cuda::std::array<short, 0> t0{};
    cuda::std::tuple<> t1{};
    t1 = t0;
  }

  {
    const cuda::std::array<short, 1> t0{3};
    cuda::std::tuple<int> t1{};
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 3);
  }

  {
    const cuda::std::array<short, 2> t0{3, 42};
    cuda::std::tuple<int, int> t1{};
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 3);
    assert(cuda::std::get<1>(t1) == 42);
  }

  {
    const cuda::std::array<short, 3> t0{3, 42, 1337};
    cuda::std::tuple<int, int, int> t1{};
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 3);
    assert(cuda::std::get<1>(t1) == 42);
    assert(cuda::std::get<2>(t1) == 1337);
  }

  { // Ensure that we SFINAE away types that are not copy assignnable
    static_assert(
      !cuda::std::is_assignable_v<cuda::std::tuple<NonAssignable>&, const cuda::std::array<NonAssignable, 1>&>);
    static_assert(
      cuda::std::is_assignable_v<cuda::std::tuple<CopyAssignable>&, const cuda::std::array<CopyAssignable, 1>&>);
    static_assert(
      !cuda::std::is_assignable_v<cuda::std::tuple<MoveAssignable>&, const cuda::std::array<MoveAssignable, 1>&>);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        const std::array<short, 1> t0{3};
        cuda::std::tuple<int> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3);
      }

      {
        const std::array<short, 2> t0{3, 42};
        cuda::std::tuple<int, int> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
      }

      {
        const std::array<short, 3> t0{3, 42, 1337};
        cuda::std::tuple<int, int, int> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
        assert(cuda::std::get<2>(t1) == 1337);
      }))

  { // Ensure that we SFINAE away types that are not copy assignnable
    static_assert(!cuda::std::is_assignable_v<cuda::std::tuple<NonAssignable>&, const std::array<NonAssignable, 1>&>);
    static_assert(cuda::std::is_assignable_v<cuda::std::tuple<CopyAssignable>&, const std::array<CopyAssignable, 1>&>);
    static_assert(!cuda::std::is_assignable_v<cuda::std::tuple<MoveAssignable>&, const std::array<MoveAssignable, 1>&>);
  }
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    const cuda::std::complex<float> t0(3.0f, 42.0f);
    cuda::std::tuple<float, float> t1{};
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 3.0f);
    assert(cuda::std::get<1>(t1) == 42.0f);
  }

  {
    const cuda::std::complex<float> t0(3.0f, 42.0f);
    cuda::std::tuple<double, double> t1{};
    t1 = t0;
    assert(cuda::std::get<0>(t1) == 3.0f);
    assert(cuda::std::get<1>(t1) == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        const std::complex<float> t0(3.0f, 42.0f);
        cuda::std::tuple<float, float> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3.0f);
        assert(cuda::std::get<1>(t1) == 42.0f);
      }

      {
        const std::complex<float> t0(3.0f, 42.0f);
        cuda::std::tuple<double, double> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3.0f);
        assert(cuda::std::get<1>(t1) == 42.0f);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        const std::tuple<> t0{};
        cuda::std::tuple<> t1{};
        t1 = t0;
      }

      {
        const std::tuple<cuda::std::int8_t, int> t0{cuda::std::int8_t{3}, 42};
        cuda::std::tuple<short, int> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
      }

      {
        const std::pair<cuda::std::int8_t, int> t0{cuda::std::int8_t{3}, 42};
        cuda::std::tuple<short, int> t1{};
        t1 = t0;
        assert(cuda::std::get<0>(t1) == 3);
        assert(cuda::std::get<1>(t1) == 42);
      }))

  { // Ensure that we SFINAE away types that are not copy assignnable
    static_assert(!cuda::std::is_assignable_v<cuda::std::tuple<NonAssignable>&, const std::tuple<NonAssignable>&>);
    static_assert(cuda::std::is_assignable_v<cuda::std::tuple<CopyAssignable>&, const std::tuple<CopyAssignable>&>);
    static_assert(!cuda::std::is_assignable_v<cuda::std::tuple<MoveAssignable>&, const std::tuple<MoveAssignable>&>);

    static_assert(!cuda::std::is_assignable_v<cuda::std::tuple<NonAssignable, NonAssignable>&,
                                              const std::pair<NonAssignable, NonAssignable>&>);
    static_assert(cuda::std::is_assignable_v<cuda::std::tuple<CopyAssignable, CopyAssignable>&,
                                             const std::pair<CopyAssignable, CopyAssignable>&>);
    static_assert(!cuda::std::is_assignable_v<cuda::std::tuple<MoveAssignable, MoveAssignable>&,
                                              const std::pair<MoveAssignable, MoveAssignable>&>);
  }
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  // Ensure that the right constructor was called
  {
    cuda::std::tuple<CopyAssign, TracedAssignment> t1{1};
    cuda::std::tuple<AssignableFrom<CopyAssign>, AssignableFrom<TracedAssignment>> t2{3};
    t2 = t1;
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2).v.copyAssign == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
