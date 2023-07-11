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

// template <class Mutex>
// class unique_lock
// {
// public:
//     typedef Mutex mutex_type;
//     ...
// };

#include<cuda/std/mutex>
#include<cuda/std/type_traits>

#include "test_macros.h"

int main(int, char**)
{
    static_assert((cuda::std::is_same<cuda::std::unique_lock<cuda::std::mutex>::mutex_type,
                   cuda::std::mutex>::value), "");

  return 0;
}
