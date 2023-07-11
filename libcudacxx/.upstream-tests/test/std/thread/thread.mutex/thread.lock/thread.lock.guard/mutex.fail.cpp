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

// template <class Mutex> class lock_guard;

// explicit lock_guard(mutex_type& m);

#include<cuda/std/mutex>

int main(int, char**)
{
    cuda::std::mutex m;
    cuda::std::lock_guard<cuda::std::mutex> lg = m; // expected-error{{no viable conversion}}

  return 0;
}
