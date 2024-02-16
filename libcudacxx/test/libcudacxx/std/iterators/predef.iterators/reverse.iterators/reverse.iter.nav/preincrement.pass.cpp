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

// reverse_iterator& operator++(); // constexpr in C++17

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 void test(It i, It x) {
    cuda::std::reverse_iterator<It> r(i);
    cuda::std::reverse_iterator<It>& rr = ++r;
    assert(r.base() == x);
    assert(&rr == &r);
}

TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 bool tests() {
    const char* s = "123";
    test(bidirectional_iterator<const char*>(s+1), bidirectional_iterator<const char*>(s));
    test(random_access_iterator<const char*>(s+1), random_access_iterator<const char*>(s));
    test(s+1, s);
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 2011
    static_assert(tests(), "");
#endif
    return 0;
}
