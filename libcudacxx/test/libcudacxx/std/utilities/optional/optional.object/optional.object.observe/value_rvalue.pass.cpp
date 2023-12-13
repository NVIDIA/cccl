//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// UNSUPPORTED: c++03, c++11
// <cuda/std/optional>

// Throwing bad_optional_access is supported starting in macosx10.13
// XFAIL: use_system_cxx_lib && target={{.+}}-apple-macosx10.{{9|10|11|12}} && !no-exceptions

// constexpr T& optional<T>::value() &&;

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

using cuda::std::optional;
#ifndef TEST_HAS_NO_EXCEPTIONS
using cuda::std::bad_optional_access;
#endif

struct X
{
    X() = default;
    X(const X&) = delete;
    X& operator=(const X&) = delete;
    __host__ __device__
    constexpr int test() const & {return 3;}
    __host__ __device__
    int test() & {return 4;}
    __host__ __device__
    constexpr int test() const && {return 5;}
    __host__ __device__
    int test() && {return 6;}
};

struct Y
{
    __host__ __device__
    constexpr int test() && {return 7;}
};

__host__ __device__
constexpr int
test()
{
    optional<Y> opt{Y{}};
    return cuda::std::move(opt).value().test();
}

int main(int, char**)
{
    {
        optional<X> opt; unused(opt);
#ifndef TEST_COMPILER_ICC
        ASSERT_NOT_NOEXCEPT(cuda::std::move(opt).value());
#endif // TEST_COMPILER_ICC
        ASSERT_SAME_TYPE(decltype(cuda::std::move(opt).value()), X&&);
    }
    {
        optional<X> opt;
        opt.emplace();
        assert(cuda::std::move(opt).value().test() == 6);
    }
#ifndef TEST_HAS_NO_EXCEPTIONS
    {
        optional<X> opt;
        try
        {
            (void)cuda::std::move(opt).value();
            assert(false);
        }
        catch (const bad_optional_access&)
        {
        }
    }
#endif
    assert(test() == 7);
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    static_assert(test() == 7, "");
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

  return 0;
}
