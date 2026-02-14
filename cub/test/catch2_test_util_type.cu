// SPDX-FileCopyrightText: Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <cub/util_type.cuh>

#include <cuda/iterator>
#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

C2H_TEST("Tests non_void_value_t", "[util][type]")
{
  using fallback_t        = float;
  using void_fancy_it     = cuda::discard_iterator;
  using non_void_fancy_it = cuda::counting_iterator<int>;

  // falls back for const void*
  STATIC_REQUIRE(cuda::std::is_same_v<fallback_t, //
                                      cub::detail::non_void_value_t<const void*, fallback_t>>);
  // falls back for const volatile void*
  STATIC_REQUIRE(cuda::std::is_same_v<fallback_t, //
                                      cub::detail::non_void_value_t<const volatile void*, fallback_t>>);
  // falls back for volatile void*
  STATIC_REQUIRE(cuda::std::is_same_v<fallback_t, //
                                      cub::detail::non_void_value_t<volatile void*, fallback_t>>);
  // falls back for void*
  STATIC_REQUIRE(cuda::std::is_same_v<fallback_t, //
                                      cub::detail::non_void_value_t<void*, fallback_t>>);
  // works for int*
  STATIC_REQUIRE(cuda::std::is_same_v<int, //
                                      cub::detail::non_void_value_t<int*, void>>);
  // falls back for fancy iterator with a void value type
  STATIC_REQUIRE(cuda::std::is_same_v<fallback_t, //
                                      cub::detail::non_void_value_t<void_fancy_it, fallback_t>>);
  // works for a fancy iterator that has int as value type
  STATIC_REQUIRE(cuda::std::is_same_v<int, //
                                      cub::detail::non_void_value_t<non_void_fancy_it, fallback_t>>);
}

CUB_DEFINE_DETECT_NESTED_TYPE(cat_detect, cat);

struct HasCat
{
  using cat = int;
};
struct HasDog
{
  using dog = int;
};

C2H_TEST("Test CUB_DEFINE_DETECT_NESTED_TYPE", "[util][type]")
{
  STATIC_REQUIRE(cat_detect<HasCat>::value);
  STATIC_REQUIRE(!cat_detect<HasDog>::value);
}

// Lots of libraries (like pytorch or tensorflow) bring their own half types and customize cub::Traits for them.
struct CustomHalf
{
  int16_t payload;
};

C2H_TEST("Test CustomHalf", "[util][type]")
{
  // type not registered with cub::Traits
  STATIC_REQUIRE(!cub::detail::is_primitive<CustomHalf>::value);
  STATIC_REQUIRE(!cuda::std::is_floating_point<CustomHalf>::value);
  STATIC_REQUIRE(!cuda::std::is_floating_point_v<CustomHalf>);
  STATIC_REQUIRE(!cuda::is_floating_point<CustomHalf>::value);
  STATIC_REQUIRE(!cuda::is_floating_point_v<CustomHalf>);
  STATIC_REQUIRE(!cuda::std::numeric_limits<CustomHalf>::is_specialized);

  // type registered with cub::Traits (specializes NumericTraits, numeric_limits, and is_floating_point)
  STATIC_REQUIRE(cub::detail::is_primitive<half_t>::value);
  STATIC_REQUIRE(!cuda::std::is_floating_point<half_t>::value); // the std traits are not affected
  STATIC_REQUIRE(!cuda::std::is_floating_point_v<half_t>);
  STATIC_REQUIRE(cuda::is_floating_point<half_t>::value);
  STATIC_REQUIRE(cuda::is_floating_point_v<half_t>);
  STATIC_REQUIRE(cuda::std::numeric_limits<half_t>::is_specialized);
  CHECK(cuda::std::numeric_limits<half_t>::max() == half_t::max());
  CHECK(cuda::std::numeric_limits<half_t>::lowest() == half_t::lowest());
}

C2H_TEST("Test FutureValue", "[util][type]")
{
  // read
  int value;
  cub::FutureValue<int> fv{&value};
  value = 42;
  CHECK(fv == 42);
  value = 43;
  CHECK(fv == 43);

  // CTAD
  cub::FutureValue fv2{&value};
  STATIC_REQUIRE(cuda::std::is_same_v<decltype(fv2), cub::FutureValue<int, int*>>);

  c2h::device_vector<int> v(0);
  cub::FutureValue fv3{v.begin()};
  STATIC_REQUIRE(
    cuda::std::is_same_v<decltype(fv3), cub::FutureValue<int, typename c2h::device_vector<int>::iterator>>);
}
