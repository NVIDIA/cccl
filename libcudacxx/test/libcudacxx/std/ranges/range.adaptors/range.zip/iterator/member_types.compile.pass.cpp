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

// Iterator traits and member typedefs in zip_view::<iterator>.

#include <cuda/std/array>
#include <cuda/std/ranges>
#include <cuda/std/tuple>

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

#if TEST_STD_VER >= 2020
template <class T>
concept HasIterCategory = requires { typename T::iterator_category; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasIterCategory = false;

template <class T>
inline constexpr bool HasIterCategory<T, cuda::std::void_t<typename T::iterator_category>> = true;
#endif // TEST_STD_VER <=2017

template <class T>
struct DiffTypeIter
{
  using iterator_category = cuda::std::input_iterator_tag;
  using value_type        = int;
  using difference_type   = T;

  __host__ __device__ int operator*() const;
  __host__ __device__ DiffTypeIter& operator++();
  __host__ __device__ void operator++(int);
#if TEST_STD_VER >= 2020
  __host__ __device__ friend constexpr bool operator==(DiffTypeIter, DiffTypeIter) = default;
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
  __host__ __device__ friend constexpr bool operator==(const DiffTypeIter&, const DiffTypeIter&)
  {
    return true;
  }
  __host__ __device__ friend constexpr bool operator!=(const DiffTypeIter&, const DiffTypeIter&)
  {
    return false;
  }
#endif // TEST_STD_VER <=2017
};

template <class T>
struct DiffTypeRange
{
  __host__ __device__ DiffTypeIter<T> begin() const
  {
    return DiffTypeIter<T>{};
  }
  __host__ __device__ DiffTypeIter<T> end() const
  {
    return DiffTypeIter<T>{};
  }
};

struct Foo
{};
struct Bar
{};

struct ConstVeryDifferentRange
{
  __host__ __device__ int* begin()
  {
    return nullptr;
  }
  __host__ __device__ int* end()
  {
    return nullptr;
  }

  __host__ __device__ forward_iterator<double*> begin() const
  {
    return forward_iterator<double*>{};
  }
  __host__ __device__ forward_iterator<double*> end() const
  {
    return forward_iterator<double*>{};
  }
};

__host__ __device__ void test()
{
  int buffer[] = {1, 2, 3, 4};
  {
    // 2 views should have pair value_type
    // random_access_iterator_tag
    cuda::std::ranges::zip_view v(buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::pair<int, int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // !=2 views should have tuple value_type
    cuda::std::ranges::zip_view v(buffer, buffer, buffer);
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int, int, int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // bidirectional_iterator_tag
    cuda::std::ranges::zip_view v(BidiCommonView{buffer});
    using Iter = decltype(v.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
  }

  {
    // forward_iterator_tag
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::zip_view<ForwardView<int>>>;

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // nested zip_view
    cuda::std::ranges::zip_view v(buffer, buffer);
    cuda::std::ranges::zip_view v2(buffer, v);
    using Iter = decltype(v2.begin());

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::pair<int, cuda::std::pair<int, int>>>);
    static_assert(HasIterCategory<Iter>);
  }

  {
    // input_iterator_tag
    using Iter = cuda::std::ranges::iterator_t<cuda::std::ranges::zip_view<InputView<int>>>;

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIterCategory<Iter>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);
  }

  {
    // difference_type of single view
    cuda::std::ranges::zip_view v{DiffTypeRange<intptr_t>{}};
    decltype(auto) it = v.begin();
    using Iter        = decltype(it);
    static_assert(cuda::std::is_same_v<Iter::difference_type, intptr_t>);
    unused(it);
  }

  {
    // difference_type of multiple views should be the common type
    cuda::std::ranges::zip_view v{DiffTypeRange<intptr_t>{}, DiffTypeRange<cuda::std::ptrdiff_t>{}};
    decltype(auto) it = v.begin();
    using Iter        = decltype(it);
    static_assert(
      cuda::std::is_same_v<Iter::difference_type, cuda::std::common_type_t<intptr_t, cuda::std::ptrdiff_t>>);
    unused(it);
  }

  const cuda::std::array<Foo, 1> foos{Foo{}};
  cuda::std::array<Bar, 2> bars{Bar{}, Bar{}};
  {
    // value_type of single view
    cuda::std::ranges::zip_view v{foos};
    using Iter = decltype(v.begin());
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<Foo>>);
    unused(v);
  }

  {
    // value_type of multiple views with different value_type
    cuda::std::ranges::zip_view v{foos, bars};
    using Iter = decltype(v.begin());
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::pair<Foo, Bar>>);
    unused(v);
  }

  {
    // const-iterator different from iterator
    cuda::std::ranges::zip_view v{ConstVeryDifferentRange{}};
    decltype(auto) it  = v.begin();
    decltype(auto) cit = cuda::std::as_const(v).begin();
    using Iter         = decltype(it);
    using ConstIter    = decltype(cit);

    static_assert(cuda::std::is_same_v<Iter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<Iter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<Iter::value_type, cuda::std::tuple<int>>);

    static_assert(cuda::std::is_same_v<ConstIter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::is_same_v<ConstIter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::is_same_v<ConstIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::is_same_v<ConstIter::value_type, cuda::std::tuple<double>>);
    unused(it);
    unused(cit);
  }
}

int main(int, char**)
{
  return 0;
}
