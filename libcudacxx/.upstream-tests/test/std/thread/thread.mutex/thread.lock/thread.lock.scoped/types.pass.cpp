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
// UNSUPPORTED: c++03, c++11, c++14
// UNSUPPORTED: pre-sm-70

// <mutex>

// template <class... MutexTypes>
// class scoped_lock
// {
// public:
//     typedef Mutex mutex_type;  // Only if sizeof...(MutexTypes) == 1
//     ...
// };

#include<cuda/std/mutex>
#include<cuda/std/type_traits>
#include "test_macros.h"

struct NAT {};

template <class LG>
__host__ __device__ auto test_typedef(int) -> typename LG::mutex_type;

template <class LG>
__host__ __device__ auto test_typedef(...) -> NAT;

template <class LG>
__host__ __device__ constexpr bool has_mutex_type() {
    return !cuda::std::is_same<decltype(test_typedef<LG>(0)), NAT>::value;
}

int main(int, char**)
{
    {
        using T = cuda::std::scoped_lock<>;
        static_assert(!has_mutex_type<T>(), "");
    }
    {
        using M1 = cuda::std::mutex;
        using T = cuda::std::scoped_lock<M1>;
        static_assert(cuda::std::is_same<T::mutex_type, M1>::value, "");
    }
#if 0 // No recursive mutex
    {
        using M1 = cuda::std::recursive_mutex;
        using T = cuda::std::scoped_lock<M1>;
        static_assert(cuda::std::is_same<T::mutex_type, M1>::value, "");
    }
    {
        using M1 = cuda::std::mutex;
        using M2 = cuda::std::recursive_mutex;
        using T = cuda::std::scoped_lock<M1, M2>;
        static_assert(!has_mutex_type<T>(), "");
    }
    {
        using M1 = cuda::std::mutex;
        using M2 = cuda::std::recursive_mutex;
        using T = cuda::std::scoped_lock<M1, M1, M2>;
        static_assert(!has_mutex_type<T>(), "");
    }
#endif
    {
        using M1 = cuda::std::mutex;
        using T = cuda::std::scoped_lock<M1, M1>;
        static_assert(!has_mutex_type<T>(), "");
    }
#if 0 // No recursive mutex
    {
        using M1 = cuda::std::recursive_mutex;
        using T = cuda::std::scoped_lock<M1, M1, M1>;
        static_assert(!has_mutex_type<T>(), "");
    }
#endif

  return 0;
}
