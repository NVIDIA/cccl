//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/iterator>

// template <InputIterator Iter>
//   Iter prev(Iter x, Iter::difference_type n = 1);

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
__host__ __device__
void
test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
    assert(cuda::std::prev(i, n) == x);

    It (*prev)(It, typename cuda::std::iterator_traits<It>::difference_type) = cuda::std::prev;
    assert(prev(i, n) == x);
}

template <class It>
__host__ __device__
void
test(It i, It x)
{
    assert(cuda::std::prev(i) == x);
}

#if TEST_STD_VER > 14
template <class It>
__host__ __device__
constexpr bool
constexpr_test(It i, typename cuda::std::iterator_traits<It>::difference_type n, It x)
{
    return cuda::std::prev(i, n) == x;
}

template <class It>
__host__ __device__
constexpr bool
constexpr_test(It i, It x)
{
    return cuda::std::prev(i) == x;
}
#endif

int main(int, char**)
{
    {
    const char* s = "1234567890";
    test(forward_iterator      <const char*>(s),    -10, forward_iterator      <const char*>(s+10));
    test(bidirectional_iterator<const char*>(s+10),  10, bidirectional_iterator<const char*>(s));
    test(bidirectional_iterator<const char*>(s),    -10, bidirectional_iterator<const char*>(s+10));
    test(random_access_iterator<const char*>(s+10),  10, random_access_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s),    -10, random_access_iterator<const char*>(s+10));
    test(s+10, 10, s);

    test(bidirectional_iterator<const char*>(s+1), bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s+1), random_access_iterator<const char*>(s));
    test(s+1, s);
    }
#if TEST_STD_VER > 14
    {
    constexpr const char* s = "1234567890";
    static_assert( constexpr_test(forward_iterator      <const char*>(s),    -10, forward_iterator      <const char*>(s+10)), "" );
    static_assert( constexpr_test(bidirectional_iterator<const char*>(s+10),  10, bidirectional_iterator<const char*>(s)), "" );
    static_assert( constexpr_test(forward_iterator      <const char*>(s),    -10, forward_iterator      <const char*>(s+10)), "" );
    static_assert( constexpr_test(random_access_iterator<const char*>(s+10),  10, random_access_iterator<const char*>(s)), "" );
    static_assert( constexpr_test(forward_iterator      <const char*>(s),    -10, forward_iterator      <const char*>(s+10)), "" );
    static_assert( constexpr_test(s+10, 10, s), "" );

    static_assert( constexpr_test(bidirectional_iterator<const char*>(s+1), bidirectional_iterator<const char*>(s)), "" );
    static_assert( constexpr_test(random_access_iterator<const char*>(s+1), random_access_iterator<const char*>(s)), "" );
    static_assert( constexpr_test(s+1, s), "" );
    }
#endif


  return 0;
}
