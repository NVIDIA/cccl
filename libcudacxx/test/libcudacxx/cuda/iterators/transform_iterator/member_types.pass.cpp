//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/iterator>
#include <cuda/std/cassert>
#include <cuda/std/concepts>

#include "test_iterators.h"
#include "test_macros.h"
#include "types.h"

template <class Iter, class Fn>
_CCCL_CONCEPT HasIterCategory =
  _CCCL_REQUIRES_EXPR((Iter, Fn))(typename(typename cuda::transform_iterator<Iter, Fn>::iterator_category));

__host__ __device__ constexpr bool test()
{
  {
    // Member typedefs for contiguous iterator.
    static_assert(
      cuda::std::same_as<cuda::std::iterator_traits<int*>::iterator_concept, cuda::std::contiguous_iterator_tag>);
    static_assert(
      cuda::std::same_as<cuda::std::iterator_traits<int*>::iterator_category, cuda::std::random_access_iterator_tag>);

    using TIter = cuda::transform_iterator<int*, Increment>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<TIter>);
  }
  {
    // Member typedefs for random access iterator.
    using TIter = cuda::transform_iterator<random_access_iterator<int*>, Increment>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<TIter>);
  }
  {
    // Member typedefs for random access iterator, LWG3798 rvalue reference.
    using TIter = cuda::transform_iterator<random_access_iterator<int*>, IncrementRvalueRef>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<TIter>);
  }
  {
    // Member typedefs for random access iterator/not-lvalue-ref.
    using TIter = cuda::transform_iterator<random_access_iterator<int*>, PlusOneMutable>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<TIter>);
  }
  {
    // Member typedefs for bidirectional iterator.
    using TIter = cuda::transform_iterator<bidirectional_iterator<int*>, Increment>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::bidirectional_iterator<TIter>);
  }
  {
    // Member typedefs for forward iterator.
    using TIter = cuda::transform_iterator<forward_iterator<int*>, Increment>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::forward_iterator<TIter>);
  }
  {
    // Member typedefs for input iterator.
    using TIter = cuda::transform_iterator<cpp17_input_iterator<int*>, Increment>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIterCategory<cpp17_input_iterator<int*>, Increment>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::input_iterator<TIter>);
  }

  {
    // Ensure we can work with other cuda iterators
    using TIter = cuda::transform_iterator<cuda::counting_iterator<int>, TimesTwo>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<TIter>);
  }

  {
    // Ensure we can work with other cuda iterators
    using TIter = cuda::std::reverse_iterator<cuda::transform_iterator<cuda::counting_iterator<int>, TimesTwo>>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::random_access_iterator<TIter>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test());

  return 0;
}
