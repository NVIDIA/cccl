//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//



// <cuda/std/tuple>

// template <class... Types> class tuple;

// explicit tuple(const T&...);

// UNSUPPORTED: c++98, c++03


#include <cuda/std/tuple>
#include <cuda/std/cassert>

#include "test_macros.h"

template <class ...>
struct never {
    enum { value = 0 };
};

struct NoValueCtor
{
    STATIC_MEMBER_VAR(count, int);

    TEST_HOST_DEVICE NoValueCtor() : id(++count()) {}
    TEST_HOST_DEVICE NoValueCtor(NoValueCtor const & other) : id(other.id) { ++count(); }

    // The constexpr is required to make is_constructible instantiate this template.
    // The explicit is needed to test-around a similar bug with is_convertible.
    template <class T>
    TEST_HOST_DEVICE constexpr explicit NoValueCtor(T)
    { static_assert(never<T>::value, "This should not be instantiated"); }

    int id;
};


struct NoValueCtorEmpty
{
    TEST_HOST_DEVICE NoValueCtorEmpty() {}
    TEST_HOST_DEVICE NoValueCtorEmpty(NoValueCtorEmpty const &) {}

    template <class T>
    TEST_HOST_DEVICE constexpr explicit NoValueCtorEmpty(T)
    { static_assert(never<T>::value, "This should not be instantiated"); }
};


struct ImplicitCopy {
  TEST_HOST_DEVICE explicit ImplicitCopy(int) {}
  TEST_HOST_DEVICE ImplicitCopy(ImplicitCopy const&) {}
};

// Test that tuple(cuda::std::allocator_arg, Alloc, Types const&...) allows implicit
// copy conversions in return value expressions.
TEST_HOST_DEVICE cuda::std::tuple<ImplicitCopy> testImplicitCopy1() {
    ImplicitCopy i(42);
    return {i};
}

TEST_HOST_DEVICE cuda::std::tuple<ImplicitCopy> testImplicitCopy2() {
    const ImplicitCopy i(42);
    return {i};
}

TEST_HOST_DEVICE cuda::std::tuple<ImplicitCopy> testImplicitCopy3() {
    const ImplicitCopy i(42);
    return i;
}

int main(int, char**)
{
    NoValueCtor::count() = 0;
    {
        // check that the literal '0' can implicitly initialize a stored pointer.
        cuda::std::tuple<int*> t = 0;
        assert(cuda::std::get<0>(t) == nullptr);
    }
    {
        cuda::std::tuple<int> t(2);
        assert(cuda::std::get<0>(t) == 2);
    }
#if TEST_STD_VER > 2011
    {
        constexpr cuda::std::tuple<int> t(2);
        static_assert(cuda::std::get<0>(t) == 2, "");
    }
    {
        constexpr cuda::std::tuple<int> t;
        static_assert(cuda::std::get<0>(t) == 0, "");
    }
#endif
    {
        cuda::std::tuple<int, char*> t(2, 0);
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == nullptr);
    }
#if TEST_STD_VER > 2011
    {
        constexpr cuda::std::tuple<int, char*> t(2, nullptr);
        static_assert(cuda::std::get<0>(t) == 2, "");
        static_assert(cuda::std::get<1>(t) == nullptr, "");
    }
#endif
    {
        cuda::std::tuple<int, char*> t(2, nullptr);
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == nullptr);
    }
    // cuda::std::string not supported
    /*
    {
        cuda::std::tuple<int, char*, cuda::std::string> t(2, nullptr, "text");
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == nullptr);
        assert(cuda::std::get<2>(t) == "text");
    }
    */
    // __tuple_leaf<T> uses is_constructible<T, U> to disable its explicit converting
    // constructor overload __tuple_leaf(U &&). Evaluating is_constructible can cause a compile error.
    // This overload is evaluated when __tuple_leafs copy or move ctor is called.
    // This checks that is_constructible is not evaluated when U == __tuple_leaf.
    {
        cuda::std::tuple<int, NoValueCtor, int, int> t(1, NoValueCtor(), 2, 3);
        assert(cuda::std::get<0>(t) == 1);
        assert(cuda::std::get<1>(t).id == 1);
        assert(cuda::std::get<2>(t) == 2);
        assert(cuda::std::get<3>(t) == 3);
    }
    {
        cuda::std::tuple<int, NoValueCtorEmpty, int, int> t(1, NoValueCtorEmpty(), 2, 3);
        assert(cuda::std::get<0>(t) == 1);
        assert(cuda::std::get<2>(t) == 2);
        assert(cuda::std::get<3>(t) == 3);
    }
    // extensions
    // cuda::std::string not supported
    /*
#ifdef _LIBCUDACXX_VERSION
    {
        cuda::std::tuple<int, char*, cuda::std::string> t(2);
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == nullptr);
        assert(cuda::std::get<2>(t) == "");
    }
    {
        cuda::std::tuple<int, char*, cuda::std::string> t(2, nullptr);
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == nullptr);
        assert(cuda::std::get<2>(t) == "");
    }
    {
        cuda::std::tuple<int, char*, cuda::std::string, double> t(2, nullptr, "text");
        assert(cuda::std::get<0>(t) == 2);
        assert(cuda::std::get<1>(t) == nullptr);
        assert(cuda::std::get<2>(t) == "text");
        assert(cuda::std::get<3>(t) == 0.0);
    }
#endif
    */

  return 0;
}
