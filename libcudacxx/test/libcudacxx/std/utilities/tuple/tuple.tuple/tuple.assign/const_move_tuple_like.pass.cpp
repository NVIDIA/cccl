//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <tuple>

// template <class... Types>
// template <class Tuple>
// constexpr explicit(see below) tuple<Types...>::tuple(const Tuple&& u);

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

TEST_FUNC constexpr bool test()
{
  {
    cuda::std::pair<ConstMoveAssign, ConstMoveAssign> t1{1, 2};
    const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign> t2{3, 4};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2).val == 2);
  }
#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::pair<ConstMoveAssign, ConstMoveAssign> t1{1, 2};
                 const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign> t2{3, 4};
                 t2 = cuda::std::move(t1);
                 assert(cuda::std::get<0>(t2).v.val == 1);
                 assert(cuda::std::get<1>(t2).val == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::array<ConstMoveAssign, 0> t1{};
    const cuda::std::tuple<> t2{};
    t2 = cuda::std::move(t1);
  }

  {
    cuda::std::array<ConstMoveAssign, 1> t1{1};
    const cuda::std::tuple<AssignableFrom<ConstMoveAssign>> t2{3};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
  }

  {
    cuda::std::array<ConstMoveAssign, 3> t1{1, 42, 1337};
    const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign, ConstMoveAssign> t2{3, 4, 5};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2).val == 42);
    assert(cuda::std::get<2>(t2).val == 1337);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::array<ConstMoveAssign, 0> t1{};
        const cuda::std::tuple<> t2{};
        t2 = cuda::std::move(t1);
      }

      {
        std::array<ConstMoveAssign, 1> t1{1};
        const cuda::std::tuple<AssignableFrom<ConstMoveAssign>> t2{3};
        t2 = cuda::std::move(t1);
        assert(cuda::std::get<0>(t2).v.val == 1);
      }

      {
        std::array<ConstMoveAssign, 3> t1{1, 42, 1337};
        const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign, ConstMoveAssign> t2{3, 4, 5};
        t2 = cuda::std::move(t1);
        assert(cuda::std::get<0>(t2).v.val == 1);
        assert(cuda::std::get<1>(t2).val == 42);
        assert(cuda::std::get<2>(t2).val == 1337);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<float, float>&, cuda::std::complex<float>&&>);
    static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<float, double>&, cuda::std::complex<double>&&>);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(
    NV_IS_HOST, ({
      static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<float, float>&, std::complex<float>&&>);
      static_assert(!cuda::std::is_assignable_v<const cuda::std::tuple<float, double>&, std::complex<double>&&>);
    }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::tuple<> t1{};
        const cuda::std::tuple<> t2{};
        t2 = cuda::std::move(t1);
      }

      {
        std::tuple<ConstMoveAssign> t1{1};
        const cuda::std::tuple<AssignableFrom<ConstMoveAssign>> t2{3};
        t2 = cuda::std::move(t1);
        assert(cuda::std::get<0>(t2).v.val == 1);
      }

      {
        std::tuple<ConstMoveAssign, ConstMoveAssign> t1{1, 42};
        const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign> t2{3, 4};
        t2 = cuda::std::move(t1);
        assert(cuda::std::get<0>(t2).v.val == 1);
        assert(cuda::std::get<1>(t2).val == 42);
      }

      {
        std::pair<ConstMoveAssign, ConstMoveAssign> t1{1, 42};
        const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, ConstMoveAssign> t2{3, 4};
        t2 = cuda::std::move(t1);
        assert(cuda::std::get<0>(t2).v.val == 1);
        assert(cuda::std::get<1>(t2).val == 42);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  // Ensure that the right constructor was called
  {
    cuda::std::tuple<ConstMoveAssign, TracedAssignment> t1{1};
    const cuda::std::tuple<AssignableFrom<ConstMoveAssign>, AssignableFrom<TracedAssignment>> t2{3};
    t2 = cuda::std::move(t1);
    assert(cuda::std::get<0>(t2).v.val == 1);
    assert(cuda::std::get<1>(t2).v.constMoveAssign == 1);
  }

  return true;
}

int main(int, char**)
{
  test();
// gcc cannot have mutable member in constant expression
#if !TEST_COMPILER(GCC)
  static_assert(test());
#endif // !TEST_COMPILER(GCC)

  return 0;
}
