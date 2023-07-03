//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// <algorithm>

// template<ForwardIterator Iter, StrictWeakOrder<auto, Iter::value_type> Compare>
//   requires CopyConstructible<Compare>
//   Iter
//   min_element(Iter first, Iter last, Compare comp);

#include <cuda/std/ranges> // #include <cuda/std/algorithm>
#include <cuda/std/functional>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class Iter>
__host__ __device__ void
test(Iter first, Iter last)
{
    Iter i = cuda::std::min_element(first, last, cuda::std::greater<int>());
    if (first != last)
    {
        for (Iter j = first; j != last; ++j)
            assert(!cuda::std::greater<int>()(*j, *i));
    }
    else
        assert(i == last);
}

template <class Iter>
__host__ __device__ void
test()
{
    constexpr int input[100] = { 62, 21, 27, 85,  4, 50, 40, 66, 60, 81,
                                16, 12, 22, 69, 23, 28, 10, 58, 15, 90,
                                33,  9, 13,  5, 31, 52, 91, 65,  6, 39,
                                71, 86,  7,  0, 34, 57, 67, 38, 87, 98,
                                61,  1, 35, 54,  8, 70, 48, 99, 46, 42,
                                55, 89, 29, 80, 44, 47, 49, 72, 19, 51,
                                3,  73,  2, 79, 94, 96, 82, 75, 77, 95,
                                56, 30, 36, 76, 64, 88, 24, 41, 53, 68,
                                93, 43, 97, 63, 14, 84, 92, 18, 32, 45,
                                83, 37, 59, 25, 20, 17, 74, 11, 78, 26 };

    test(Iter(input), Iter(input+0));
    test(Iter(input), Iter(input+1));
    test(Iter(input), Iter(input+2));
    test(Iter(input), Iter(input+3));
    test(Iter(input), Iter(input+10));
    test(Iter(input), Iter(input+100));
}

template <class Iter, class Pred>
__host__ __device__ void test_eq0(Iter first, Iter last, Pred p)
{
    assert(first == cuda::std::min_element(first, last, p));
}

__host__ __device__ void test_eq()
{
    int a[10] = {};
    for (int i = 0; i < 10; ++i)
        a[i] = 10; // all the same
    test_eq0(a, a+10, cuda::std::less<int>());
    test_eq0(a, a+10, cuda::std::greater<int>());
}

#if TEST_STD_VER >= 14
constexpr int il[] = { 2, 4, 6, 8, 7, 5, 3, 1 };

__host__ __device__ constexpr void constexpr_test()
{
    struct less { __host__ __device__ constexpr bool operator ()( const int &x, const int &y) const { return x < y; }};
    constexpr auto p = cuda::std::min_element(il, il+8, less());
    static_assert(*p == 1, "");
}
#endif

int main(int, char**)
{
    test<forward_iterator<const int*> >();
    test<bidirectional_iterator<const int*> >();
    test<random_access_iterator<const int*> >();
    test<const int*>();
    test_eq();

#if TEST_STD_VER >= 14
    constexpr_test();
#endif

  return 0;
}
