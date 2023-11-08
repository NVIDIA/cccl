//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// ADDITIONAL_COMPILE_FLAGS: -D_LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

// <cuda/std/iterator>

// move_iterator

// pointer operator->() const;
//
//  constexpr in C++17

#define _LIBCUDACXX_DISABLE_DEPRECATION_WARNINGS

#include <cuda/std/iterator>
#include <cuda/std/cassert>

#include "test_macros.h"

__host__ __device__ TEST_CONSTEXPR_CXX14 bool test()
{
    char a[] = "123456789";
    cuda::std::move_iterator<char *> it1 = cuda::std::make_move_iterator(a);
    cuda::std::move_iterator<char *> it2 = cuda::std::make_move_iterator(a + 1);
    assert(it1.operator->() == a);
    assert(it2.operator->() == a + 1);

    return true;
}

int main(int, char**)
{
    test();
#if TEST_STD_VER > 11
    static_assert(test(), "");
#endif

    return 0;
}
