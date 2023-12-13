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

// requires RandomAccessIterator<Iter>
//   constexpr reverse_iterator& operator+=(difference_type n);
//
// constexpr in C++17

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
__host__ __device__
void
test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
    cuda::std::reverse_iterator<It> r(i);
    cuda::std::reverse_iterator<It>& rr = r += n;
    assert(r.base() == x);
    assert(&rr == &r);
}

int main(int, char**)
{
    const char* s = "1234567890";
    test(random_access_iterator<const char*>(s+5), 5, random_access_iterator<const char*>(s));
    test(s+5, 5, s);

#if TEST_STD_VER > 14
#if !defined(TEST_COMPILER_ICC)
    {
        constexpr const char *p = "123456789";
        constexpr auto it1 = cuda::std::make_reverse_iterator(p);
        constexpr auto it2 = cuda::std::make_reverse_iterator(p+5) += 5;
        static_assert(it1 == it2, "");
    }
#endif // !TEST_COMPILER_ICC
#endif // TEST_STD_VER > 14

  return 0;
}
