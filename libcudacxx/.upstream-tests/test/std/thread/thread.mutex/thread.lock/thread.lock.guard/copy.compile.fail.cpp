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

// template <class Mutex> class lock_guard;

// lock_guard(lock_guard const&) = delete;

#include<cuda/std/mutex>

int main(int, char**)
{
    cuda::std::mutex m;
    cuda::std::lock_guard<cuda::std::mutex> lg0(m);
    cuda::std::lock_guard<cuda::std::mutex> lg(lg0);

  return 0;
}
