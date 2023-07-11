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
// UNSUPPORTED: gcc-7
// UNSUPPORTED: pre-sm-70

// <mutex>

// scoped_lock

// Make sure that the implicitly-generated CTAD works.

#include<cuda/std/mutex>

#include "test_macros.h"

int main(int, char**) {
  cuda::std::mutex m1;
  {
    cuda::std::scoped_lock lock(m1);
    ASSERT_SAME_TYPE(decltype(lock), cuda::std::scoped_lock<cuda::std::mutex>);
  }
#if 0 // No recursive mutex
  cuda::std::recursive_mutex m2;
  cuda::std::recursive_timed_mutex m3;
  {
    cuda::std::scoped_lock lock(m1, m2);
    ASSERT_SAME_TYPE(decltype(lock), cuda::std::scoped_lock<cuda::std::mutex, cuda::std::recursive_mutex>);
  }
  {
    cuda::std::scoped_lock lock(m1, m2, m3);
    ASSERT_SAME_TYPE(decltype(lock), cuda::std::scoped_lock<cuda::std::mutex, cuda::std::recursive_mutex, cuda::std::recursive_timed_mutex>);
  }
#endif

  return 0;
}

