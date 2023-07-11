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

// mutex_type *mutex() const;

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

STATIC_TEST_GLOBAL_VAR cuda::std::mutex m;

int main(int, char**)
{
    cuda::std::unique_lock<cuda::std::mutex> lk0;
    assert(lk0.mutex() == nullptr);
    cuda::std::unique_lock<cuda::std::mutex> lk1(m);
    assert(lk1.mutex() == &m);
    lk1.unlock();
    assert(lk1.mutex() == &m);

  return 0;
}
