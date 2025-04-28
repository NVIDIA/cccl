//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// transform_view::<iterator>::difference_type
// transform_view::<iterator>::value_type
// transform_view::<iterator>::iterator_category
// transform_view::<iterator>::iterator_concept

#include "../types.h"

#include <cuda/std/ranges>

#include "test_macros.h"

template <class V, class F>
_CCCL_CONCEPT HasIterCategory =
  _CCCL_REQUIRES_EXPR((V, F))(typename(typename cuda::std::ranges::transform_view<V, F>::iterator_category));

__host__ __device__ constexpr bool test()
{
  {
    // Member typedefs for contiguous iterator.
    static_assert(
      cuda::std::same_as<cuda::std::iterator_traits<int*>::iterator_concept, cuda::std::contiguous_iterator_tag>);
    static_assert(
      cuda::std::same_as<cuda::std::iterator_traits<int*>::iterator_category, cuda::std::random_access_iterator_tag>);

    using TView = cuda::std::ranges::transform_view<MoveOnlyView, Increment>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }
  {
    // Member typedefs for random access iterator.
    using TView = cuda::std::ranges::transform_view<RandomAccessView, Increment>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }
  {
    // Member typedefs for random access iterator, LWG3798 rvalue reference.
    using TView = cuda::std::ranges::transform_view<RandomAccessView, IncrementRvalueRef>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }
  {
    // Member typedefs for random access iterator/not-lvalue-ref.
    using TView = cuda::std::ranges::transform_view<RandomAccessView, PlusOneMutable>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::random_access_iterator_tag>);
    static_assert(
      cuda::std::same_as<typename TIter::iterator_category, cuda::std::input_iterator_tag>); // Note: this is now
                                                                                             // input_iterator_tag.
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }
  {
    // Member typedefs for bidirectional iterator.
    using TView = cuda::std::ranges::transform_view<BidirectionalView, Increment>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::bidirectional_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }
  {
    // Member typedefs for forward iterator.
    using TView = cuda::std::ranges::transform_view<ForwardView, Increment>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }
  {
    // Member typedefs for input iterator.
    using TView = cuda::std::ranges::transform_view<InputView, Increment>;
    using TIter = cuda::std::ranges::iterator_t<TView>;
    static_assert(cuda::std::same_as<typename TIter::iterator_concept, cuda::std::input_iterator_tag>);
    static_assert(!HasIterCategory<InputView, Increment>);
    static_assert(cuda::std::same_as<typename TIter::value_type, int>);
    static_assert(cuda::std::same_as<typename TIter::difference_type, cuda::std::ptrdiff_t>);
  }

  return true;
}

int main(int, char**)
{
  test();
  static_assert(test(), "");

  return 0;
}
