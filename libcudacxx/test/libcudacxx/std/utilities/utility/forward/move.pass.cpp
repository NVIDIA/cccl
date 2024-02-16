//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2023 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// test move

#include <cuda/std/utility>
#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"

class move_only
{
    TEST_HOST_DEVICE move_only(const move_only&);
    TEST_HOST_DEVICE move_only& operator=(const move_only&);
public:
    TEST_HOST_DEVICE move_only(move_only&&) {}
    TEST_HOST_DEVICE move_only& operator=(move_only&&) {return *this;}

    TEST_HOST_DEVICE move_only() {}
};

TEST_HOST_DEVICE move_only source() {return move_only();}
TEST_HOST_DEVICE const move_only csource() {return move_only();}

TEST_HOST_DEVICE void test(move_only) {}

STATIC_TEST_GLOBAL_VAR int global_var = 42;
STATIC_TEST_GLOBAL_VAR const int& global_reference = global_var;

template <class QualInt>
TEST_HOST_DEVICE QualInt get() TEST_NOEXCEPT { return static_cast<QualInt>(global_var); }

STATIC_TEST_GLOBAL_VAR int copy_ctor = 0;
STATIC_TEST_GLOBAL_VAR int move_ctor = 0;

struct A {
    TEST_HOST_DEVICE A() {}
    TEST_HOST_DEVICE A(const A&) {++copy_ctor;}
    TEST_HOST_DEVICE A(A&&) {++move_ctor;}
    TEST_HOST_DEVICE A& operator=(const A&) = delete;
};

#if TEST_STD_VER > 2011
TEST_HOST_DEVICE constexpr bool test_constexpr_move() {
    int y = 42;
    const int cy = y;
    return cuda::std::move(y) == 42
        && cuda::std::move(cy) == 42
        && cuda::std::move(static_cast<int&&>(y)) == 42
        && cuda::std::move(static_cast<int const&&>(y)) == 42;
}
#endif
int main(int, char**)
{
    { // Test return type and noexcept.
        static_assert(cuda::std::is_same<decltype(cuda::std::move(global_var)), int&&>::value, "");
        ASSERT_NOEXCEPT(cuda::std::move(global_var));
        static_assert(cuda::std::is_same<decltype(cuda::std::move(global_reference)), const int&&>::value, "");
        ASSERT_NOEXCEPT(cuda::std::move(global_reference));
        static_assert(cuda::std::is_same<decltype(cuda::std::move(42)), int&&>::value, "");
        ASSERT_NOEXCEPT(cuda::std::move(42));
        static_assert(cuda::std::is_same<decltype(cuda::std::move(get<const int&&>())), const int&&>::value, "");
        ASSERT_NOEXCEPT(cuda::std::move(get<int const&&>()));
        (void)global_reference;
    }
    { // test copy and move semantics
        A a;
        const A ca = A();

        assert(copy_ctor == 0);
        assert(move_ctor == 0);

        A a2 = a; (void)a2;
        assert(copy_ctor == 1);
        assert(move_ctor == 0);

        A a3 = cuda::std::move(a); (void)a3;
        assert(copy_ctor == 1);
        assert(move_ctor == 1);

        A a4 = ca; (void)a4;
        assert(copy_ctor == 2);
        assert(move_ctor == 1);

        A a5 = cuda::std::move(ca); (void)a5;
        assert(copy_ctor == 3);
        assert(move_ctor == 1);
    }
    { // test on a move only type
        move_only mo;
        test(cuda::std::move(mo));
        test(source());
    }
#if TEST_STD_VER > 2011
    {
        constexpr int y = 42;
        static_assert(cuda::std::move(y) == 42, "");
        static_assert(test_constexpr_move(), "");
    }
#endif
#if TEST_STD_VER == 2011 && defined(_LIBCUDACXX_VERSION)
    // Test that cuda::std::forward is constexpr in C++11. This is an extension
    // provided by both libc++ and libstdc++.
    {
        constexpr int y = 42;
        static_assert(cuda::std::move(y) == 42, "");
    }
#endif

  return 0;
}
