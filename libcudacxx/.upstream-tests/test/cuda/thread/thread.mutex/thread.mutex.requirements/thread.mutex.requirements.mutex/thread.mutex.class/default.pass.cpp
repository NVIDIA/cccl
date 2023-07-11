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

// class mutex;

// mutex();

#include<cuda/mutex>
#include<cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
  static_assert(cuda::std::is_nothrow_default_constructible<cuda::mutex<cuda::thread_scope_system>>::value, "");
  static_assert(cuda::std::is_nothrow_default_constructible<cuda::mutex<cuda::thread_scope_device>>::value, "");
  static_assert(cuda::std::is_nothrow_default_constructible<cuda::mutex<cuda::thread_scope_block>>::value, "");
  static_assert(cuda::std::is_nothrow_default_constructible<cuda::mutex<cuda::thread_scope_thread>>::value, "");

  return 0;
}
