//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++98, c++03, c++11, c++14
// UNSUPPORTED: pre-sm-70

// <mutex>

// lock_guard

// Make sure that the implicitly-generated CTAD works.

#include<cuda/std/mutex>

#include "test_macros.h"

int main(int, char**) {
  cuda::std::mutex mutex;
  {
    cuda::std::lock_guard lock(mutex);
    ASSERT_SAME_TYPE(decltype(lock), cuda::std::lock_guard<cuda::std::mutex>);
  }

  return 0;
}

