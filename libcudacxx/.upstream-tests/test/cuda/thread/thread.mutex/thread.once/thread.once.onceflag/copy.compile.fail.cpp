//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-70

// <mutex>

// struct once_flag;

// once_flag(const once_flag&) = delete;

#include<cuda/mutex>

template<int scope>
void test() {
    cuda::once_flag<scope> f0;
    cuda::once_flag<scope> f1{f0};
}

int main(int, char**)
{
  test<cuda::thread_scope_system>();
  test<cuda::thread_scope_device>();
  test<cuda::thread_scope_block>();
  test<cuda::thread_scope_thread>();

  return 0;
}
