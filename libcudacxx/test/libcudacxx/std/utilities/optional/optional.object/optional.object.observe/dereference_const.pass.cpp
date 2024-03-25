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

// constexpr const T& optional<T>::operator*() const &;

#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

using cuda::std::optional;

struct X
{
    __host__ __device__
    constexpr int test() const& {return 3;}
    __host__ __device__
    int test() & {return 4;}
    __host__ __device__
    constexpr int test() const&& {return 5;}
    __host__ __device__
    int test() && {return 6;}
};

struct Y
{
    __host__ __device__
    int test() const {return 2;}
};

int main(int, char**)
{
    {
        const optional<X> opt; unused(opt);
        ASSERT_SAME_TYPE(decltype(*opt), X const&);
        LIBCPP_STATIC_ASSERT(noexcept(*opt), "");
        // ASSERT_NOT_NOEXCEPT(*opt);
        // FIXME: This assertion fails with GCC because it can see that
        // (A) operator*() is constexpr, and
        // (B) there is no path through the function that throws.
        // It's arguable if this is the correct behavior for the noexcept
        // operator.
        // Regardless this function should still be noexcept(false) because
        // it has a narrow contract.
    }
#if !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))
    {
        constexpr optional<X> opt(X{});
        static_assert((*opt).test() == 3, "");
    }
    {
        constexpr optional<Y> opt(Y{});
        assert((*opt).test() == 2);
    }
#endif // !(defined(TEST_COMPILER_CUDACC_BELOW_11_3) && defined(TEST_COMPILER_CLANG))

    return 0;
}
