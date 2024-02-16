//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

// <cuda/std/tuple>

// template <class... Types> class tuple;

// template <class... UTypes> tuple(const tuple<UTypes...>& u);

// XFAIL: gcc-4.8, gcc-4.9

// UNSUPPORTED: c++98, c++03


#include <cuda/std/tuple>
#include <cuda/std/cassert>

#include "test_macros.h"

struct Explicit {
  int value;
  TEST_HOST_DEVICE explicit Explicit(int x) : value(x) {}
};

struct Implicit {
  int value;
  TEST_HOST_DEVICE Implicit(int x) : value(x) {}
};

struct ExplicitTwo {
    TEST_HOST_DEVICE ExplicitTwo() {}
    TEST_HOST_DEVICE ExplicitTwo(ExplicitTwo const&) {}
    TEST_HOST_DEVICE ExplicitTwo(ExplicitTwo &&) {}

    template <class T, class = typename cuda::std::enable_if<!cuda::std::is_same<T, ExplicitTwo>::value>::type>
    TEST_HOST_DEVICE explicit ExplicitTwo(T) {}
};

struct B
{
    int id_;

    TEST_HOST_DEVICE explicit B(int i) : id_(i) {}
};

struct D
    : B
{
    TEST_HOST_DEVICE explicit D(int i) : B(i) {}
};

#if TEST_STD_VER > 2011

struct A
{
    int id_;

    TEST_HOST_DEVICE constexpr A(int i) : id_(i) {}
    TEST_HOST_DEVICE friend constexpr bool operator==(const A& x, const A& y) {return x.id_ == y.id_;}
};

struct C
{
    int id_;

    TEST_HOST_DEVICE constexpr explicit C(int i) : id_(i) {}
    TEST_HOST_DEVICE friend constexpr bool operator==(const C& x, const C& y) {return x.id_ == y.id_;}
};

#endif

int main(int, char**)
{
    {
        typedef cuda::std::tuple<long> T0;
        typedef cuda::std::tuple<long long> T1;
        T0 t0(2);
        T1 t1 = t0;
        assert(cuda::std::get<0>(t1) == 2);
    }
#if TEST_STD_VER > 2011
    {
        typedef cuda::std::tuple<int> T0;
        typedef cuda::std::tuple<A> T1;
        constexpr T0 t0(2);
        constexpr T1 t1 = t0;
        static_assert(cuda::std::get<0>(t1) == 2, "");
    }
    {
        typedef cuda::std::tuple<int> T0;
        typedef cuda::std::tuple<C> T1;
        constexpr T0 t0(2);
        constexpr T1 t1{t0};
        static_assert(cuda::std::get<0>(t1) == C(2), "");
    }
#endif
    {
        typedef cuda::std::tuple<long, char> T0;
        typedef cuda::std::tuple<long long, int> T1;
        T0 t0(2, 'a');
        T1 t1 = t0;
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1) == int('a'));
    }
    {
        typedef cuda::std::tuple<long, char, D> T0;
        typedef cuda::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', D(3));
        T1 t1 = t0;
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1) == int('a'));
        assert(cuda::std::get<2>(t1).id_ == 3);
    }
    {
        D d(3);
        typedef cuda::std::tuple<long, char, D&> T0;
        typedef cuda::std::tuple<long long, int, B&> T1;
        T0 t0(2, 'a', d);
        T1 t1 = t0;
        d.id_ = 2;
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1) == int('a'));
        assert(cuda::std::get<2>(t1).id_ == 2);
    }
    {
        typedef cuda::std::tuple<long, char, int> T0;
        typedef cuda::std::tuple<long long, int, B> T1;
        T0 t0(2, 'a', 3);
        T1 t1(t0);
        assert(cuda::std::get<0>(t1) == 2);
        assert(cuda::std::get<1>(t1) == int('a'));
        assert(cuda::std::get<2>(t1).id_ == 3);
    }
    {
        const cuda::std::tuple<int> t1(42);
        cuda::std::tuple<Explicit> t2(t1);
        assert(cuda::std::get<0>(t2).value == 42);
    }
    {
        const cuda::std::tuple<int> t1(42);
        cuda::std::tuple<Implicit> t2 = t1;
        assert(cuda::std::get<0>(t2).value == 42);
    }
    {
        static_assert(cuda::std::is_convertible<ExplicitTwo&&, ExplicitTwo>::value, "");
        static_assert(cuda::std::is_convertible<cuda::std::tuple<ExplicitTwo&&>&&, const cuda::std::tuple<ExplicitTwo>&>::value, "");

        ExplicitTwo e;
        cuda::std::tuple<ExplicitTwo> t = cuda::std::tuple<ExplicitTwo&&>(cuda::std::move(e));
        ((void)t);
    }
  return 0;
}
