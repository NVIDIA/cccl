/******************************************************************************
 * Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#include <cub/util_type.cuh>

#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/discard_iterator.h>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

C2H_TEST("Tests non_void_value_t", "[util][type]")
{
  using fallback_t        = float;
  using void_fancy_it     = thrust::discard_iterator<std::size_t>;
  using non_void_fancy_it = thrust::counting_iterator<int>;

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
