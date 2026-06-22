//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/utility>

// template <class T1, class T2> struct pair

// template <pair-like Pair> EXPLICIT constexpr pair(Pair&& p);

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
    cuda::std::tuple<MutableCopy, int> p{1, 2};
    cuda::std::pair<ConvertibleFrom<MutableCopy>, ConvertibleFrom<int>> p2 = p;
    assert(cuda::std::get<0>(p2).v.val == 1);
    assert(cuda::std::get<1>(p2).v == 2);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::tuple<MutableCopy, int> p{1, 2};
                 cuda::std::pair<ConvertibleFrom<MutableCopy>, ConvertibleFrom<int>> p2 = p;
                 assert(cuda::std::get<0>(p2).v.val == 1);
                 assert(cuda::std::get<1>(p2).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::array<MutableCopy, 2> p{1, 2};
    cuda::std::pair<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> p2 = p;
    assert(cuda::std::get<0>(p2).v.val == 1);
    assert(cuda::std::get<1>(p2).v.val == 2);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::array<MutableCopy, 2> p{1, 2};
                 cuda::std::pair<ConvertibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> p2 = p;
                 assert(cuda::std::get<0>(p2).v.val == 1);
                 assert(cuda::std::get<1>(p2).v.val == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::complex<float> p{1.0f, 2.0f};
    cuda::std::pair<ConvertibleFrom<float>, ConvertibleFrom<float>> p2 = p;
    assert(cuda::std::get<0>(p2).v == 1.0f);
    assert(cuda::std::get<1>(p2).v == 2.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::complex<float> p{1.0f, 2.0f};
                 cuda::std::pair<ConvertibleFrom<float>, ConvertibleFrom<float>> p2 = p;
                 assert(cuda::std::get<0>(p2).v == 1.0f);
                 assert(cuda::std::get<1>(p2).v == 2.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)

  // test explicit conversions.
  {
    cuda::std::tuple<MutableCopy, int> p{1, 2};
    cuda::std::pair<ExplicitConstructibleFrom<MutableCopy>, ExplicitConstructibleFrom<int>> p2{p};
    assert(cuda::std::get<0>(p2).v.val == 1);
    assert(cuda::std::get<1>(p2).v == 2);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::tuple<MutableCopy, int> p{1, 2};
                 cuda::std::pair<ExplicitConstructibleFrom<MutableCopy>, ExplicitConstructibleFrom<int>> p2{p};
                 assert(cuda::std::get<0>(p2).v.val == 1);
                 assert(cuda::std::get<1>(p2).v == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::array<MutableCopy, 2> p{1, 2};
    cuda::std::pair<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> p2{p};
    assert(cuda::std::get<0>(p2).v.val == 1);
    assert(cuda::std::get<1>(p2).v.val == 2);
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::array<MutableCopy, 2> p{1, 2};
                 cuda::std::pair<ExplicitConstructibleFrom<MutableCopy>, ConvertibleFrom<MutableCopy>> p2{p};
                 assert(cuda::std::get<0>(p2).v.val == 1);
                 assert(cuda::std::get<1>(p2).v.val == 2);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB()

  {
    cuda::std::complex<float> p{1.0f, 2.0f};
    cuda::std::pair<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> p2{p};
    assert(cuda::std::get<0>(p2).v == 1.0f);
    assert(cuda::std::get<1>(p2).v == 2.0f);
  }

#if _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::complex<float> p{1.0f, 2.0f};
                 cuda::std::pair<ExplicitConstructibleFrom<float>, ConvertibleFrom<float>> p2{p};
                 assert(cuda::std::get<0>(p2).v.val == 1.0f);
                 assert(cuda::std::get<1>(p2).v.val == 2.0f);
               }))
#endif // _CCCL_HAS_HOST_STD_LIB() && defined(__cpp_lib_tuple_like)

  // non const overload should be called
  {
    cuda::std::tuple<TracedCopyMove, TracedCopyMove> p;
    cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = p;
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(p2).v));
    assert(nonConstCopyCtrCalled(cuda::std::get<1>(p2)));
  }

  {
    cuda::std::array<TracedCopyMove, 2> p;
    cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = p;
    assert(nonConstCopyCtrCalled(cuda::std::get<0>(p2).v));
    assert(nonConstCopyCtrCalled(cuda::std::get<1>(p2)));
  }

#if _CCCL_HAS_HOST_STD_LIB()
  NV_IF_TARGET(NV_IS_HOST, ({
                 std::array<TracedCopyMove, 2> p;
                 cuda::std::pair<ConvertibleFrom<TracedCopyMove>, TracedCopyMove> p2 = p;
                 assert(nonConstCopyCtrCalled(cuda::std::get<0>(p2).v));
                 assert(nonConstCopyCtrCalled(cuda::std::get<1>(p2)));
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
