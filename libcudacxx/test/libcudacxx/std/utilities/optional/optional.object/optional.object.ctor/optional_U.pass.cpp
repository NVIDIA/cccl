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

// template <class U>
//   optional(optional<U>&& rhs);

#include <cuda/std/cassert>
#include <cuda/std/optional>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "test_macros.h"

using cuda::std::optional;

template <class T, class U>
__host__ __device__
TEST_CONSTEXPR_CXX14 void
test(optional<U>&& rhs, bool is_going_to_throw = false)
{
    bool rhs_engaged = static_cast<bool>(rhs);
#ifndef TEST_HAS_NO_EXCEPTIONS
    try
    {
        optional<T> lhs = cuda::std::move(rhs);
        assert(is_going_to_throw == false);
        assert(static_cast<bool>(lhs) == rhs_engaged);
    }
    catch (int i)
    {
        assert(i == 6);
    }
#else
    if (is_going_to_throw) return;
    optional<T> lhs = cuda::std::move(rhs);
    assert(static_cast<bool>(lhs) == rhs_engaged);
#endif
}

class X
{
    int i_;
public:
    __host__ __device__
    TEST_CONSTEXPR_CXX20 X(int i) : i_(i) {}
    __host__ __device__
    TEST_CONSTEXPR_CXX20 X(X&& x) : i_(cuda::std::exchange(x.i_, 0)) {}
    __host__ __device__
    TEST_CONSTEXPR_CXX20 ~X() {i_ = 0;}
    __host__ __device__
    friend constexpr bool operator==(const X& x, const X& y) {return x.i_ == y.i_;}
};

struct Z
{
    __host__ __device__
    Z(int) { TEST_THROW(6); }
};

template<class T, class U>
__host__ __device__
TEST_CONSTEXPR_CXX20 bool test_all()
{
    {
        optional<T> rhs;
        test<U>(cuda::std::move(rhs));
    }
    {
        optional<T> rhs(short{3});
        test<U>(cuda::std::move(rhs));
    }
    return true;
}

int main(int, char**)
{
    test_all<short, int>();
    test_all<int, X>();
#if TEST_STD_VER > 17 && defined(_LIBCUDACXX_ADDRESSOF)
    static_assert(test_all<short, int>());
    static_assert(test_all<int, X>());
#endif
    {
        optional<int> rhs;
        test<Z>(cuda::std::move(rhs));
    }
    {
        optional<int> rhs(3);
        test<Z>(cuda::std::move(rhs), true);
    }

    static_assert(!(cuda::std::is_constructible<optional<X>, optional<Z>>::value), "");

  return 0;
}
