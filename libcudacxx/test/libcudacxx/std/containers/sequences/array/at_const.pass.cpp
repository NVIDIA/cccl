//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/array>

// const_reference at (size_type) const; // constexpr in C++14

#include <cuda/std/array>
#include <cuda/std/cassert>

#include "test_macros.h"

#ifndef TEST_HAS_NO_EXCEPTIONS
#include <cuda/std/detail/libcxx/include/stdexcept>
#endif // TEST_HAS_NO_EXCEPTIONS

__host__ __device__ TEST_CONSTEXPR_CXX14 bool tests()
{
    {
        typedef double T;
        typedef cuda::std::array<T, 3> C;
        C const c = {1, 2, 3.5};
        typename C::const_reference r1 = c.at(0);
        assert(r1 == 1);

        typename C::const_reference r2 = c.at(2);
        assert(r2 == 3.5);
    }
    return true;
}

#ifndef TEST_HAS_NO_EXCEPTIONS
__host__ __device__ void test_exceptions()
{
    {
        cuda::std::array<int, 4> const array = {1, 2, 3, 4};

        try {
            TEST_IGNORE_NODISCARD array.at(4);
            assert(false);
        } catch (cuda::std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }

        try {
            TEST_IGNORE_NODISCARD array.at(5);
            assert(false);
        } catch (cuda::std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }

        try {
            TEST_IGNORE_NODISCARD array.at(6);
            assert(false);
        } catch (cuda::std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }

        try {
            using size_type = decltype(array)::size_type;
            TEST_IGNORE_NODISCARD array.at(static_cast<size_type>(-1));
            assert(false);
        } catch (cuda::std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }
    }

    {
        cuda::std::array<int, 0> array = {};

        try {
            TEST_IGNORE_NODISCARD array.at(0);
            assert(false);
        } catch (cuda::std::out_of_range const&) {
            // pass
        } catch (...) {
            assert(false);
        }
    }
}
#endif // TEST_HAS_NO_EXCEPTIONS

int main(int, char**)
{
    tests();
#ifndef TEST_HAS_NO_EXCEPTIONS
    test_exceptions();
#endif // TEST_HAS_NO_EXCEPTIONS

#if TEST_STD_VER >= 2014
    static_assert(tests(), "");
#endif
    return 0;
}
