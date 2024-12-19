//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: msvc-19.16

// template<input_iterator I>
//   requires same_as<ITER_TRAITS(I), iterator_traits<I>>   // see [iterator.concepts.general]
// struct iterator_traits<counted_iterator<I>> : iterator_traits<I> {
//   using pointer = conditional_t<contiguous_iterator<I>,
//                                 add_pointer_t<iter_reference_t<I>>, void>;
// };

#include <cuda/std/iterator>

#include "test_iterators.h"
#include "test_macros.h"

__host__ __device__ void test()
{
  {
    using Iter        = cpp17_input_iterator<int*>;
    using CountedIter = cuda::std::counted_iterator<Iter>;
    using IterTraits  = cuda::std::iterator_traits<CountedIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::input_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter        = forward_iterator<int*>;
    using CountedIter = cuda::std::counted_iterator<Iter>;
    using IterTraits  = cuda::std::iterator_traits<CountedIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::forward_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter        = random_access_iterator<int*>;
    using CountedIter = cuda::std::counted_iterator<Iter>;
    using IterTraits  = cuda::std::iterator_traits<CountedIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::random_access_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, void>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
  {
    using Iter        = contiguous_iterator<int*>;
    using CountedIter = cuda::std::counted_iterator<Iter>;
    using IterTraits  = cuda::std::iterator_traits<CountedIter>;

    static_assert(cuda::std::same_as<IterTraits::iterator_category, cuda::std::contiguous_iterator_tag>);
    static_assert(cuda::std::same_as<IterTraits::value_type, int>);
    static_assert(cuda::std::same_as<IterTraits::difference_type, cuda::std::ptrdiff_t>);
    static_assert(cuda::std::same_as<IterTraits::pointer, int*>);
    static_assert(cuda::std::same_as<IterTraits::reference, int&>);
  }
}

int main(int, char**)
{
  return 0;
}
