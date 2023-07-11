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

// unique_lock();

#include<cuda/std/mutex>
#include<cuda/std/cassert>

#include "test_macros.h"

int main(int, char**)
{
    cuda::std::unique_lock<cuda::std::mutex> ul;
    assert(!ul.owns_lock());
    assert(ul.mutex() == nullptr);

  return 0;
}
