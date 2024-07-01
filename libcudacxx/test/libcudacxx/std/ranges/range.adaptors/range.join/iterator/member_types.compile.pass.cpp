//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// Iterator traits and member typedefs in join_view::<iterator>.

#include <cuda/std/ranges>

#include "../types.h"
#include "test_iterators.h"
#include "test_macros.h"

template <class T>
struct ForwardView : cuda::std::ranges::view_base
{
  __host__ __device__ forward_iterator<T*> begin() const
  {
    return forward_iterator<T*>{nullptr};
  }
  __host__ __device__ sentinel_wrapper<forward_iterator<T*>> end() const
  {
    return sentinel_wrapper<forward_iterator<T*>>{};
  }
};

template <class T>
struct InputView : cuda::std::ranges::view_base
{
  __host__ __device__ cpp17_input_iterator<T*> begin() const
  {
    return cpp17_input_iterator<T*>{nullptr};
  }
  __host__ __device__ sentinel_wrapper<cpp17_input_iterator<T*>> end() const
  {
    return sentinel_wrapper<cpp17_input_iterator<T*>>{};
  }
};

template <class T, class V>
struct diff_type_iter
{
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = V;
  using difference_type   = T;

  __host__ __device__ V& operator*() const;
  __host__ __device__ diff_type_iter& operator++();
  __host__ __device__ void operator++(int);
#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(diff_type_iter, diff_type_iter) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const diff_type_iter&, const diff_type_iter&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(const diff_type_iter&, const diff_type_iter&)
  {
    return false;
  }
#endif // TEST_STD_VER <= 2017
};

template <class T, class V = int>
struct DiffTypeRange : cuda::std::ranges::view_base
{
  __host__ __device__ diff_type_iter<T, V> begin() const
  {
    return diff_type_iter<T, V>{};
  }
  __host__ __device__ diff_type_iter<T, V> end() const
  {
    return diff_type_iter<T, V>{};
  }
};

#if TEST_STD_VER >= 2020
template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasIterCategory = false;

template <class T>
inline constexpr bool HasIterCategory<T, cuda::std::void_t<typename T::iterator_category>> = true;
#endif // TEST_STD_VER <= 2017

__host__ __device__ void test()
{
  {
    int buffer[4][4];
    cuda::std::ranges::join_view jv(buffer);
    using Iter = cuda::std::ranges::iterator_t<decltype(jv)>;

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::join_view<ForwardView<ForwardView<int>>>>;

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::join_view<InputView<InputView<int>>>>;

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIterCategory<Iter>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
  }

  {
    // LWG3535 `join_view::iterator::iterator_category` and `::iterator_concept` lie
    // Bidi non common inner range should not have bidirectional_iterator_tag
    using Base = BidiCommonOuter<BidiNonCommonInner>;
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::join_view<Base>>;
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(HasIterCategory<Iter>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, int>);
  }

  {
    // !ref-is-glvalue
    using Outer = InnerRValue<BidiCommonOuter<BidiCommonInner>>;
    using Iter  = cuda::std::ranges::iterator_t<cuda::std::ranges::join_view<Outer>>;
    static_assert(!HasIterCategory<Iter>);
    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::input_iterator_tag>);
  }

  {
    // value_type == inner's value_type
    using Inner          = IterMoveSwapAwareView;
    using InnerValue     = cuda::std::ranges::range_value_t<Inner>;
    using InnerReference = cuda::std::ranges::range_reference_t<Inner>;
    static_assert(!cuda::std::is_same_v<InnerValue, cuda::std::remove_cvref<InnerReference>>);

    using Outer = BidiCommonOuter<Inner>;
    using Iter  = cuda::std::ranges::iterator_t<cuda::std::ranges::join_view<Outer>>;
    static_assert(cuda::std::is_same_v<InnerValue, cuda::std::pair<int, int>>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::pair<int, int>>);
  }

  {
    // difference_type
    using Inner = DiffTypeRange<cuda::std::intptr_t>;
    using Outer = DiffTypeRange<cuda::std::ptrdiff_t, Inner>;
    using Iter  = cuda::std::ranges::iterator_t<cuda::std::ranges::join_view<Outer>>;
    static_assert(
      cuda::std::is_same_v<Iter::difference_type, cuda::std::common_type_t<cuda::std::intptr_t, cuda::std::ptrdiff_t>>);
  }
}

int main(int, char**)
{
  return 0;
}
