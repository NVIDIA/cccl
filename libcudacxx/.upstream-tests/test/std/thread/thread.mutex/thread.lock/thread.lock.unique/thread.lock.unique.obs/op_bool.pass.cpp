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

// template <class Mutex> class unique_lock;

// explicit operator bool() const noexcept;

#include<cuda/std/mutex>
#include<cuda/std/cassert>
#include<cuda/std/type_traits>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR cuda::std::mutex m;

int main(int, char**)
{
    static_assert(cuda::std::is_constructible<bool, cuda::std::unique_lock<cuda::std::mutex> >::value, "");
    static_assert(!cuda::std::is_convertible<cuda::std::unique_lock<cuda::std::mutex>, bool>::value, "");

    cuda::std::unique_lock<cuda::std::mutex> lk0;
    assert(static_cast<bool>(lk0) == false);
    cuda::std::unique_lock<cuda::std::mutex> lk1(m);
    assert(static_cast<bool>(lk1) == true);
    lk1.unlock();
    assert(static_cast<bool>(lk1) == false);
    ASSERT_NOEXCEPT(static_cast<bool>(lk0));

  return 0;
}
