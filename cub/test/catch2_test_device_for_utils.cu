// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_for.cuh>

#include <c2h/catch2_test_helper.h>

template <class T>
struct value_t
{
  __device__ void operator()(T) {}
};

template <class T>
struct const_ref_t
{
  __device__ void operator()(const T&) {}
};

template <class T>
struct rref_t
{
  __device__ void operator()(T&&) {}
};

template <class T>
struct value_ret_t
{
  __device__ T operator()(T v)
  {
    return v;
  }
};

template <class T>
struct ref_t
{
  __device__ void operator()(T&) {}
};

struct tpl_value_t
{
  template <class T>
  __device__ void operator()(T)
  {}
};

template <class T>
struct overload_value_t
{
  __device__ void operator()(T) {}
  __device__ void operator()(T) const {}
};

template <class T>
struct value_const_t
{
  __device__ void operator()(T) const {}
};

template <class T>
void test()
{
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<T, value_t<T>>::value);
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<T, value_const_t<T>>::value);
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<T, value_ret_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, rref_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, ref_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, const_ref_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, overload_value_t<T>>::value);
  STATIC_REQUIRE(!cub::detail::for_each::has_unique_value_overload<T, tpl_value_t>::value);
}

C2H_TEST("Device for utils correctly detect value overloads", "[for][device]")
{
  ::test<int>();
  ::test<double>();

  // conversions do not work ;(
  STATIC_REQUIRE(cub::detail::for_each::has_unique_value_overload<char, value_t<int>>::value);
}
