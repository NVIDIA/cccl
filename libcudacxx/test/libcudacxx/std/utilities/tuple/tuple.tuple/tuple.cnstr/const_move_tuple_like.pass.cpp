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
  // test implicit conversions.
  {
    const cuda::std::pair<ConstMove, int> p{1, 2};
    cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<int>> t = cuda::std::move(p);
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::pair<ConstMove, int> p{1, 2};
                 cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<int>> t = cuda::std::move(p);
                 assert(cuda::std::get<0>(t).v.val == 1);
                 assert(cuda::std::get<1>(t).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  {
    const cuda::std::array<ConstMove, 1> p{1};
    cuda::std::tuple<ConvertibleFrom<ConstMove>> t = cuda::std::move(p);
    assert(cuda::std::get<0>(t).v.val == 1);
  }

  {
    const cuda::std::array<ConstMove, 2> p{1, 42};
    cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t = cuda::std::move(p);
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 42);
  }

  {
    const cuda::std::array<ConstMove, 3> p{1, 42, 1337};
    cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t =
      cuda::std::move(p);
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 42);
    assert(cuda::std::get<2>(t).v.val == 1337);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        const std::array<ConstMove, 1> p{1};
        cuda::std::tuple<ConvertibleFrom<ConstMove>> t = cuda::std::move(p);
        assert(cuda::std::get<0>(t).v.val == 1);
      }

      {
        const std::array<ConstMove, 2> p{1, 42};
        cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t = cuda::std::move(p);
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 42);
      }

      {
        const std::array<ConstMove, 3> p{1, 42, 1337};
        cuda::std::tuple<ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t =
          cuda::std::move(p);
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 42);
        assert(cuda::std::get<2>(t).v.val == 1337);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    const cuda::std::complex<float> p{1.0f, 42.0f};
    cuda::std::tuple<ConvertibleFrom<float>, ConvertibleFrom<float>> t = cuda::std::move(p);
    assert(cuda::std::get<0>(t).v == 1.0f);
    assert(cuda::std::get<1>(t).v == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::complex<float> p{1.0f, 42.0f};
                 cuda::std::tuple<ConvertibleFrom<float>, ConvertibleFrom<float>> t = cuda::std::move(p);
                 assert(cuda::std::get<0>(t).v == 1.0f);
                 assert(cuda::std::get<1>(t).v == 42.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // test explicit conversions.
  {
    const cuda::std::pair<ConstMove, int> p{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ExplicitConstructibleFrom<int>> t{cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::pair<ConstMove, int> p{1, 2};
                 cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ExplicitConstructibleFrom<int>> t{
                   cuda::std::move(p)};
                 assert(cuda::std::get<0>(t).v.val == 1);
                 assert(cuda::std::get<1>(t).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    const cuda::std::array<ConstMove, 1> p{1};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>> t{cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v.val == 1);
  }

  {
    const cuda::std::array<ConstMove, 2> p{1, 42};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t{cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 42);
  }

  {
    const cuda::std::array<ConstMove, 3> p{1, 42, 1337};
    cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t{
      cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 42);
    assert(cuda::std::get<2>(t).v.val == 1337);
  }

#if _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        const std::array<ConstMove, 1> p{1};
        cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>> t{cuda::std::move(p)};
        assert(cuda::std::get<0>(t).v.val == 1);
      }

      {
        const std::array<ConstMove, 2> p{1, 42};
        cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t{cuda::std::move(p)};
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 42);
      }

      {
        const std::array<ConstMove, 3> p{1, 42, 1337};
        cuda::std::tuple<ExplicitConstructibleFrom<ConstMove>, ConvertibleFrom<ConstMove>, ConvertibleFrom<ConstMove>> t{
          cuda::std::move(p)};
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 42);
        assert(cuda::std::get<2>(t).v.val == 1337);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB() && !TEST_COMPILER(GCC, <, 9)

  {
    const cuda::std::complex<float> p{1.0f, 42.0f};
    cuda::std::tuple<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> t{cuda::std::move(p)};
    assert(cuda::std::get<0>(t).v == 1.0f);
    assert(cuda::std::get<1>(t).v == 42.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 const std::complex<float> p{1.0f, 42.0f};
                 cuda::std::tuple<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> t{cuda::std::move(p)};
                 assert(cuda::std::get<0>(t).v == 1.0f);
                 assert(cuda::std::get<1>(t).v == 42.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // const overload should be called
  {
    const cuda::std::pair<TracedCopyMove, TracedCopyMove> p{};
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = cuda::std::move(p);
    assert(constMoveCtrCalled(cuda::std::get<0>(t).v));
    assert(constMoveCtrCalled(cuda::std::get<1>(t)));
  }

  {
    const cuda::std::array<TracedCopyMove, 1> p{};
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>> t = cuda::std::move(p);
    assert(constMoveCtrCalled(cuda::std::get<0>(t).v));
  }

  {
    const cuda::std::array<TracedCopyMove, 2> p{};
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = cuda::std::move(p);
    assert(constMoveCtrCalled(cuda::std::get<0>(t).v));
    assert(constMoveCtrCalled(cuda::std::get<1>(t)));
  }

  {
    const cuda::std::array<TracedCopyMove, 3> p{};
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove, TracedCopyMove> t = cuda::std::move(p);
    assert(constMoveCtrCalled(cuda::std::get<0>(t).v));
    assert(constMoveCtrCalled(cuda::std::get<1>(t)));
    assert(constMoveCtrCalled(cuda::std::get<2>(t)));
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
