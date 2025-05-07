//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// Some basic examples of how transform_view might be used in the wild. This is a general
// collection of sample algorithms and functions that try to mock general usage of
// this view.

#include <cuda/std/cassert>
#include <cuda/std/functional>
#include <cuda/std/ranges>
#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
#  include <cuda/std/string_view>
#endif // _LIBCUDACXX_HAS_STRING_VIEW
#if defined(_LIBCUDACXX_HAS_VECTOR)
#  include <cuda/std/vector>
#endif // _LIBCUDACXX_HAS_VECTOR

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class T, class F>
_CCCL_CONCEPT ValidTransformView =
  _CCCL_REQUIRES_EXPR((T, F))(typename(typename cuda::std::ranges::transform_view<T, F>));

struct BadFunction
{};
static_assert(ValidTransformView<MoveOnlyView, PlusOne>);
static_assert(!ValidTransformView<Range, PlusOne>);
static_assert(!ValidTransformView<MoveOnlyView, BadFunction>);

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
template <class R, cuda::std::enable_if_t<cuda::std::ranges::range<R>, int> = 0>
__host__ __device__ auto toUpper(R range)
{
  return cuda::std::ranges::transform_view(range, [](char c) {
    return cuda::std::toupper(c);
  });
}
#endif // _LIBCUDACXX_HAS_STRING_VIEW

template <class E1, class E2, size_t N, class Join = cuda::std::plus<E1>>
__host__ __device__ auto joinArrays(E1 (&a)[N], E2 (&b)[N], Join join = Join())
{
  return cuda::std::ranges::transform_view(a, [&a, &b, join](E1& x) {
    auto idx = (&x) - a;
    return join(x, b[idx]);
  });
}

struct NonConstView : cuda::std::ranges::view_base
{
  __host__ __device__ explicit NonConstView(int* b, int* e)
      : b_(b)
      , e_(e)
  {}
  __host__ __device__ const int* begin()
  {
    return b_;
  } // deliberately non-const
  __host__ __device__ const int* end()
  {
    return e_;
  } // deliberately non-const
  const int* b_;
  const int* e_;
};

template <class Range, class Expected>
__host__ __device__ constexpr bool equal(Range&& range, Expected&& expected)
{
  for (size_t i = 0; i < cuda::std::size(expected); ++i)
  {
    if (range[i] != expected[i])
    {
      return false;
    }
  }
  return true;
}

int main(int, char**)
{
#if defined(_LIBCUDACXX_HAS_VECTOR)
  {
    cuda::std::vector<int> vec = {1, 2, 3, 4};
    auto transformed           = cuda::std::ranges::transform_view(vec, [](int x) {
      return x + 42;
    });
    int expected[]             = {43, 44, 45, 46};
    assert(equal(transformed, expected));
    const auto& ct = transformed;
    assert(equal(ct, expected));
  }
#endif // _LIBCUDACXX_HAS_VECTOR

  {
    // Test a view type that is not const-iterable.
    int a[]          = {1, 2, 3, 4};
    auto transformed = NonConstView(a, a + 4) | cuda::std::views::transform([](int x) {
                         return x + 42;
                       });
    int expected[4]  = {43, 44, 45, 46};
    assert(equal(transformed, expected));
  }

  {
    int a[4]     = {1, 2, 3, 4};
    int b[4]     = {4, 3, 2, 1};
    auto out     = joinArrays(a, b);
    int check[4] = {5, 5, 5, 5};
    assert(equal(out, check));
  }

#if defined(_LIBCUDACXX_HAS_STRING_VIEW)
  {
    cuda::std::string_view str   = "Hello, World.";
    auto upp                     = toUpper(str);
    cuda::std::string_view check = "HELLO, WORLD.";
    assert(equal(upp, check));
  }
#endif // _LIBCUDACXX_HAS_STRING_VIEW

  return 0;
}
