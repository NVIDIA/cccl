#ifndef _MY_INT_HPP
#define _MY_INT_HPP

#include "test_macros.h"

struct my_int_non_convertible;

struct my_int
{
    int _val;

    TEST_HOST_DEVICE my_int( my_int_non_convertible ) noexcept;
    TEST_HOST_DEVICE constexpr my_int( int val ) : _val( val ){};
    TEST_HOST_DEVICE constexpr operator int() const noexcept { return _val; }
};

template <> struct cuda::std::is_integral<my_int> : cuda::std::true_type {};

// Wrapper type that's not implicitly convertible

struct my_int_non_convertible
{
    my_int _val;

    TEST_HOST_DEVICE my_int_non_convertible();
    TEST_HOST_DEVICE my_int_non_convertible( my_int val ) : _val( val ){};
    TEST_HOST_DEVICE operator my_int() const noexcept { return _val; }
};

TEST_HOST_DEVICE my_int::my_int( my_int_non_convertible ) noexcept {}

template <> struct cuda::std::is_integral<my_int_non_convertible> : cuda::std::true_type {};

// Wrapper type that's not nothrow-constructible

struct my_int_non_nothrow_constructible
{
    int _val;

    TEST_HOST_DEVICE my_int_non_nothrow_constructible();
    TEST_HOST_DEVICE my_int_non_nothrow_constructible( int val ) : _val( val ){};
    TEST_HOST_DEVICE operator int() const { return _val; }
};

template <> struct cuda::std::is_integral<my_int_non_nothrow_constructible> : cuda::std::true_type {};

#endif
