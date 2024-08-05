#ifndef _MY_INT_HPP
#define _MY_INT_HPP

#include "test_macros.h"

struct my_int_non_convertible;

struct my_int
{
  int _val;

  __host__ __device__ my_int(my_int_non_convertible) noexcept;
  __host__ __device__ constexpr my_int(int val)
      : _val(val){};
  __host__ __device__ constexpr operator int() const noexcept
  {
    return _val;
  }
};

template <>
struct cuda::std::is_integral<my_int> : cuda::std::true_type
{};

// Wrapper type that's not implicitly convertible

struct my_int_non_convertible
{
  my_int _val;

  __host__ __device__ my_int_non_convertible();
  __host__ __device__ my_int_non_convertible(my_int val)
      : _val(val) {};
  __host__ __device__ operator my_int() const noexcept
  {
    return _val;
  }
};

__host__ __device__ my_int::my_int(my_int_non_convertible) noexcept {}

template <>
struct cuda::std::is_integral<my_int_non_convertible> : cuda::std::true_type
{};

// Wrapper type that's not nothrow-constructible

struct my_int_non_nothrow_constructible
{
  int _val;

  __host__ __device__ my_int_non_nothrow_constructible();
  __host__ __device__ my_int_non_nothrow_constructible(int val)
      : _val(val) {};
  __host__ __device__ operator int() const
  {
    return _val;
  }
};

template <>
struct cuda::std::is_integral<my_int_non_nothrow_constructible> : cuda::std::true_type
{};

#endif
