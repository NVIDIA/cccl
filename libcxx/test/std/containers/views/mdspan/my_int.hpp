#ifndef _MY_INT_HPP
#define _MY_INT_HPP

struct my_int_non_convertible;

struct my_int
{
    int _val;

    my_int( my_int_non_convertible ) noexcept;
    constexpr my_int( int val ) : _val( val ){};
    constexpr operator int() const noexcept { return _val; }
};

template <> struct std::is_integral<my_int> : std::true_type {};

// Wrapper type that's not implicitly convertible

struct my_int_non_convertible
{
    my_int _val;

    my_int_non_convertible();
    my_int_non_convertible( my_int val ) : _val( val ){};
    operator my_int() const noexcept { return _val; }
};

my_int::my_int( my_int_non_convertible ) noexcept {}

template <> struct std::is_integral<my_int_non_convertible> : std::true_type {};

// Wrapper type that's not nothrow-constructible

struct my_int_non_nothrow_constructible
{
    int _val;

    my_int_non_nothrow_constructible();
    my_int_non_nothrow_constructible( int val ) : _val( val ){};
    operator int() const { return _val; }
};

template <> struct std::is_integral<my_int_non_nothrow_constructible> : std::true_type {};

#endif
