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
// constexpr explicit(see below) tuple<Types...>::tuple(Tuple& u);

#include <cuda/std/__memory_>
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
    cuda::std::pair<MutableCopy, int> p{1, 2};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<int>> t = p;
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
  }

  {
    cuda::std::array<MutableCopy, 1> p{1};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>> t = p;
    assert(cuda::std::get<0>(t).v.val == 1);
  }

  {
    cuda::std::array<MutableCopy, 2> p{1, 2};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> t = p;
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 2);
  }

  {
    cuda::std::array<MutableCopy, 3> p{1, 42, 1337};
    cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> t = p;
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 42);
    assert(cuda::std::get<2>(t).v.val == 1337);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::array<MutableCopy, 1> p{1};
        cuda::std::tuple<ConvertibleFrom<MutableCopy>> t = p;
        assert(cuda::std::get<0>(t).v.val == 1);
      }

      {
        std::array<MutableCopy, 2> p{1, 2};
        cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> t = p;
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 2);
      }

      {
        std::array<MutableCopy, 3> p{1, 42, 1337};
        cuda::std::tuple<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> t =
          p;
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 42);
        assert(cuda::std::get<2>(t).v.val == 1337);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::complex<float> p{1.0f, 2.0f};
    cuda::std::tuple<ConvertibleFrom<float>, ConvertibleFrom<float>> t = p;
    assert(cuda::std::get<0>(t).v == 1.0f);
    assert(cuda::std::get<1>(t).v == 2.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::complex<float> p{1.0f, 2.0f};
                 cuda::std::tuple<ConvertibleFrom<float>, ConvertibleFrom<float>> t = p;
                 assert(cuda::std::get<0>(t).v.val == 1.0f);
                 assert(cuda::std::get<1>(t).v == 2.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // test explicit conversions.
  {
    cuda::std::pair<MutableCopy, int> p{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, ExplicitConstructibleFrom<int>> t{p};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v == 2);
  }

  {
    cuda::std::array<MutableCopy, 1> p{1};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>> t{p};
    assert(cuda::std::get<0>(t).v.val == 1);
  }

  {
    cuda::std::array<MutableCopy, 2> p{1, 2};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> t{p};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 2);
  }

  {
    cuda::std::array<MutableCopy, 3> p{1, 42, 1337};
    cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>
      t{p};
    assert(cuda::std::get<0>(t).v.val == 1);
    assert(cuda::std::get<1>(t).v.val == 42);
    assert(cuda::std::get<2>(t).v.val == 1337);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(
    NV_IS_HOST,
    (
      {
        std::array<MutableCopy, 1> p{1};
        cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>> t{p};
        assert(cuda::std::get<0>(t).v.val == 1);
      }

      {
        std::array<MutableCopy, 2> p{1, 2};
        cuda::std::tuple<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> t{p};
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 2);
      }

      {
        std::array<MutableCopy, 3> p{1, 42, 1337};
        cuda::std::
          tuple<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>>
            t{p};
        assert(cuda::std::get<0>(t).v.val == 1);
        assert(cuda::std::get<1>(t).v.val == 42);
        assert(cuda::std::get<2>(t).v.val == 1337);
      }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::complex<float> p{1.0f, 2.0f};
    cuda::std::tuple<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> t{p};
    assert(cuda::std::get<0>(t).v == 1.0f);
    assert(cuda::std::get<1>(t).v == 2.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::complex<float> p{1.0f, 2.0f};
                 cuda::std::tuple<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> t{p};
                 assert(cuda::std::get<0>(t).v.val == 1.0f);
                 assert(cuda::std::get<1>(t).v == 2.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && __cpp_lib_tuple_like >= 202311L

  // non const overload should be called
  {
    cuda::std::pair<TracedCopyMove, TracedCopyMove> p;
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = p;
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(t).v));
    assert(nonConstCopyCtrCalled(cuda::std::get<1>(t)));
  }

  {
    cuda::std::array<TracedCopyMove, 2> p;
    cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = p;
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(t).v));
    assert(nonConstCopyCtrCalled(cuda::std::get<1>(t)));
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::array<TracedCopyMove, 2> p;
                 cuda::std::tuple<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> t = p;
                 assert(nonConstCopyCtrCalled(cuda::std::get<0>(t).v));
                 assert(nonConstCopyCtrCalled(cuda::std::get<1>(t)));
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
