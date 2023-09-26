//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef TEST_OVERRUN_GUARD
#define TEST_OVERRUN_GUARD

#include <cuda/std/type_traits>

template<typename T>
struct overrun_guard {
    cuda::std::size_t init_value;
    T val[2];

    template<typename U>
    __host__ __device__
    overrun_guard(U u) : init_value(u), val{static_cast<T>(u), static_cast<T>(u + 1)} {}

    __host__ __device__
    overrun_guard(const overrun_guard & other) : init_value(other.init_value), val{other.val[0], other.val[1]} {
        assert(other.val[1] == static_cast<T>(other.init_value + 1));
    }

    template<typename U>
    __host__ __device__
    overrun_guard & operator=(U u) {
        assert(val[1] == static_cast<T>(init_value + 1));
        val[0] = u;
        return *this;
    }

    __host__ __device__
    overrun_guard & operator=(const overrun_guard & other) {
        assert(val[1] == static_cast<T>(init_value + 1));
        assert(other.val[1] == static_cast<T>(other.init_value + 1));
        val[0] = other.val[0];
        return *this;
    }

    template<typename U, typename TT = T, typename = typename cuda::std::enable_if<cuda::std::is_integral<TT>::value>::type>
    __host__ __device__
    bool operator==(const U & u) const {
        assert(val[1] == static_cast<T>(init_value + 1));
        return val[0] == static_cast<T>(u);
    }

    template<typename U, typename TT = T>
    __host__ __device__
    typename cuda::std::enable_if<!cuda::std::is_integral<TT>::value, bool>::type operator==(const U & u) const {
        assert(val[1] == static_cast<T>(init_value + 1));
        return val[0] == u;
    }

    __host__ __device__
    bool operator==(const overrun_guard & other) const {
        assert(val[1] == static_cast<T>(init_value + 1));
        assert(other.val[1] == static_cast<T>(other.init_value + 1));
        return val[0] == other.val[0];
    }

    // ...without the noinline, NVCC generates invalid calls to the isshared nvvm intrinsic
    // TODO: file a bug about this
    __host__ __device__ __noinline__
    T * get() {
        return &val[0];
    }
};

#endif
