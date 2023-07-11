//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-70

// <mutex>

// class timed_mutex;

// timed_mutex(const timed_mutex&) = delete;

#include<cuda/mutex>

template<int scope>
void test() {
    cuda::timed_mutex<scope> m0;
    cuda::timed_mutex<scope> m1{m0};
}

int main(int, char**)
{
  test<cuda::thread_scope_system>();
  test<cuda::thread_scope_device>();
  test<cuda::thread_scope_block>();
  test<cuda::thread_scope_thread>();

  return 0;
}
