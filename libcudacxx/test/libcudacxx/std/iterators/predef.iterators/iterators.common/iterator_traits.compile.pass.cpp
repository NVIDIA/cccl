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

// template<input_iterator I, class S>
//   struct iterator_traits<common_iterator<I, S>>;

// clang complains about _LIBCUDACXX_NO_CFI on addressoff for builtin pointer types
#if defined(__clang__)
#  pragma clang diagnostic ignored "-Wignored-attributes"
#endif

#include <cuda/std/cstddef>
#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

#if TEST_STD_VER >= 2020
template <class T>
concept HasIteratorConcept = requires { typename T::iterator_concept; };
#else // ^^^ C++20 ^^^ / vvv C++17 vvv
template <class T, class = void>
inline constexpr bool HasIteratorConcept = false;

template <class T>
inline constexpr bool HasIteratorConcept<T, cuda::std::void_t<typename T::iterator_concept>> = true;
#endif // TEST_STD_VER <= 2017

struct NonVoidOutputIterator
{
  using value_type      = int;
  using difference_type = cuda::std::ptrdiff_t;
  __host__ __device__ const NonVoidOutputIterator& operator*() const;
  __host__ __device__ NonVoidOutputIterator& operator++();
  __host__ __device__ NonVoidOutputIterator& operator++(int);
  __host__ __device__ void operator=(int) const;
};

__host__ __device__ void test()
{
  {
    using Iter       = simple_iterator<int*>;
    using CommonIter = cuda::std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, int*>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter       = value_iterator<int*>;
    using CommonIter = cuda::std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    // Note: IterTraits::pointer == __proxy.
    static_assert(!cuda::std::same_as<IterTraits::pointer, int*>);
    static_assert(cuda::std::same_as<IterTraits::reference, int>);
  }
  // Test with an output_iterator that has a void value_type
  {
    using Iter       = cpp17_output_iterator<int*>;
    using CommonIter = cuda::std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(!HasIteratorConcept<IterTraits>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, void>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, void>);
  }
  // Test with an output_iterator that has a non-void value_type
  {
    using CommonIter = cuda::std::common_iterator<NonVoidOutputIterator, sentinel_wrapper<NonVoidOutputIterator>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(!HasIteratorConcept<IterTraits>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::output_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, void>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, void>);
  }
  {
    using Iter       = cpp17_input_iterator<int*>;
    using CommonIter = cuda::std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, int*>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter       = forward_iterator<int*>;
    using CommonIter = cuda::std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, int*>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter       = random_access_iterator<int*>;
    using CommonIter = cuda::std::common_iterator<Iter, sentinel_type<int*>>;
    using IterTraits = cuda::std::iterator_traits<CommonIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, int*>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }

  // Testing iterator conformance.
  {
    static_assert(
      cuda::std::input_iterator<cuda::std::common_iterator<cpp17_input_iterator<int*>, sentinel_type<int*>>>);
    static_assert(cuda::std::forward_iterator<cuda::std::common_iterator<forward_iterator<int*>, sentinel_type<int*>>>);
    static_assert(
      cuda::std::forward_iterator<cuda::std::common_iterator<random_access_iterator<int*>, sentinel_type<int*>>>);
    static_assert(
      cuda::std::forward_iterator<cuda::std::common_iterator<contiguous_iterator<int*>, sentinel_type<int*>>>);
    // Even these are only forward.
    static_assert(
      !cuda::std::bidirectional_iterator<cuda::std::common_iterator<random_access_iterator<int*>, sentinel_type<int*>>>);
    static_assert(
      !cuda::std::bidirectional_iterator<cuda::std::common_iterator<contiguous_iterator<int*>, sentinel_type<int*>>>);

    using Iter = cuda::std::common_iterator<forward_iterator<int*>, sentinel_type<int*>>;
    static_assert(cuda::std::indirectly_writable<Iter, int>);
    static_assert(cuda::std::indirectly_swappable<Iter, Iter>);
  }
}

int main(int, char**)
{
  test();
  return 0;
}
