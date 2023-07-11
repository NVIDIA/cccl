//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//
//
// UNSUPPORTED: no-threads
// UNSUPPORTED: pre-sm-70

// <mutex>

// struct defer_lock_t { explicit defer_lock_t() = default; };
// struct try_to_lock_t { explicit try_to_lock_t() = default; };
// struct adopt_lock_t { explicit adopt_lock_t() = default; };
//
// constexpr defer_lock_t  defer_lock{};
// constexpr try_to_lock_t try_to_lock{};
// constexpr adopt_lock_t  adopt_lock{};

#include<cuda/std/mutex>

#include "test_macros.h"

int main(int, char**)
{
    typedef cuda::std::defer_lock_t T1;
    typedef cuda::std::try_to_lock_t T2;
    typedef cuda::std::adopt_lock_t T3;

    T1 t1 = cuda::std::defer_lock; unused(t1);
    T2 t2 = cuda::std::try_to_lock; unused(t2);
    T3 t3 = cuda::std::adopt_lock; unused(t3);

    return 0;
}
