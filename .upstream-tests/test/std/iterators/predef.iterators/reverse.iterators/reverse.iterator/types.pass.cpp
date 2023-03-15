//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// reverse_iterator

// Test nested types and data member:

// template <BidirectionalIterator Iter>
// class reverse_iterator {
// protected:
//   Iter current;
// public:
//   iterator<typename iterator_traits<Iterator>::iterator_category,
//   typename iterator_traits<Iterator>::value_type,
//   typename iterator_traits<Iterator>::difference_type,
//   typename iterator_traits<Iterator>::pointer,
//   typename iterator_traits<Iterator>::reference> {
// };

#include <cuda/std/iterator>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
struct find_current
    : private cuda::std::reverse_iterator<It>
{
__host__ __device__
    void test() {++(this->current);}
};

template <class It>
__host__ __device__
void
test()
{
    typedef cuda::std::reverse_iterator<It> R;
    typedef cuda::std::iterator_traits<It> T;
    find_current<It> q;
    q.test();
    static_assert((cuda::std::is_same<typename R::iterator_type, It>::value), "");
    static_assert((cuda::std::is_same<typename R::value_type, typename T::value_type>::value), "");
    static_assert((cuda::std::is_same<typename R::difference_type, typename T::difference_type>::value), "");
    static_assert((cuda::std::is_same<typename R::reference, typename T::reference>::value), "");
    static_assert((cuda::std::is_same<typename R::pointer, typename cuda::std::iterator_traits<It>::pointer>::value), "");
    static_assert((cuda::std::is_same<typename R::iterator_category, typename T::iterator_category>::value), "");
}

int main(int, char**)
{
    test<bidirectional_iterator<char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();

  return 0;
}
