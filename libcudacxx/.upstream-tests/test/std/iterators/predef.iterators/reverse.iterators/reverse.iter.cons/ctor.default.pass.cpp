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

// reverse_iterator(); // constexpr since C++17

#include <cuda/std/iterator>

#include "test_macros.h"
#include "test_iterators.h"

template <class It>
__host__ __device__ TEST_CONSTEXPR_CXX14 void test() {
    cuda::std::reverse_iterator<It> r;
    unused(r);
}

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests() {
    test<bidirectional_iterator<const char*> >();
    test<random_access_iterator<char*> >();
    test<char*>();
    test<const char*>();
    return true;
}

int main(int, char**) {
    tests();
#if TEST_STD_VER > 11
    static_assert(tests(), "");
#endif
    return 0;
}
