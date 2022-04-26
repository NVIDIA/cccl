//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// constexpr unspecified ignore;

// UNSUPPORTED: c++98, c++03
// Internal compiler error in 14.24
// XFAIL: msvc-19.20, msvc-19.21, msvc-19.22, msvc-19.23, msvc-19.24, msvc-19.25

#include <cuda/std/tuple>
#include <cuda/std/cassert>

#include "test_macros.h"

template <typename T>
__host__ __device__
constexpr bool unused(T &&) {return true;}

__host__ __device__
constexpr bool test_ignore_constexpr()
{
// NVCC does not support constexpr non-integral types
#if TEST_STD_VER > 11 && !defined(__CUDA_ARCH__)
    { // Test that std::ignore provides constexpr converting assignment.
        auto& res = (cuda::std::ignore = 42);
        assert(&res == &cuda::std::ignore);
    }
    { // Test that cuda::std::ignore provides constexpr copy/move constructors
        auto copy = cuda::std::ignore;
        auto moved = cuda::std::move(copy);
        unused(moved);
    }
    { // Test that cuda::std::ignore provides constexpr copy/move assignment
        auto copy = cuda::std::ignore;
        copy = cuda::std::ignore;
        auto moved = cuda::std::ignore;
        moved = cuda::std::move(copy);
        unused(moved);
    }
#endif
    return true;
}

int main(int, char**) {
    {
        constexpr auto& ignore_v = cuda::std::ignore;
        ((void)ignore_v);
    }
    {
        static_assert(test_ignore_constexpr(), "");
    }
    {
        LIBCPP_STATIC_ASSERT(cuda::std::is_trivial<decltype(cuda::std::ignore)>::value, "");
    }

  return 0;
}
