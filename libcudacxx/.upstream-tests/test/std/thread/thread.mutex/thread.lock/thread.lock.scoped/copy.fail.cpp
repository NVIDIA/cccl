//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: no-threads
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: nvrtc
// UNSUPPORTED: pre-sm-70

// <mutex>

// template <class ...Mutex> class scoped_lock;

// scoped_lock(scoped_lock const&) = delete;

#include<cuda/std/mutex>
#include "test_macros.h"

int main(int, char**)
{
    using M = cuda::std::mutex;
    M m0, m1, m2;
    {
        using LG = cuda::std::scoped_lock<>;
        const LG Orig;
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }
    {
        using LG = cuda::std::scoped_lock<M>;
        const LG Orig(m0);
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }
    {
        using LG = cuda::std::scoped_lock<M, M>;
        const LG Orig(m0, m1);
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }
    {
        using LG = cuda::std::scoped_lock<M, M, M>;
        const LG Orig(m0, m1, m2);
        LG Copy(Orig); // expected-error{{call to deleted constructor of 'LG'}}
    }

  return 0;
}
