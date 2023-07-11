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

// timed_mutex& operator=(const timed_mutex&) = delete;

#include<cuda/std/mutex>

int main(int, char**)
{
    cuda::std::timed_mutex m0;
    cuda::std::timed_mutex m1;
    m1 = m0;

  return 0;
}
