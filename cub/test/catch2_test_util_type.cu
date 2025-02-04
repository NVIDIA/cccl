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

#include <cub/iterator/counting_input_iterator.cuh>
#include <cub/iterator/discard_output_iterator.cuh>
#include <cub/util_type.cuh>

#include <cuda/std/type_traits>

#include <c2h/catch2_test_helper.h>
#include <c2h/extended_types.h>

C2H_TEST("Tests non_void_value_t", "[util][type]")
{
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  using fallback_t        = float;
  using void_fancy_it     = cub::DiscardOutputIterator<std::size_t>;
  using non_void_fancy_it = cub::CountingInputIterator<int>;

  // falls back for const void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<const void*, fallback_t>>::value);
  // falls back for const volatile void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<const volatile void*, fallback_t>>::value);
  // falls back for volatile void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<volatile void*, fallback_t>>::value);
  // falls back for void*
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<void*, fallback_t>>::value);
  // works for int*
  STATIC_REQUIRE(::cuda::std::is_same<int, //
                                      cub::detail::non_void_value_t<int*, void>>::value);
  // falls back for fancy iterator with a void value type
  STATIC_REQUIRE(::cuda::std::is_same<fallback_t, //
                                      cub::detail::non_void_value_t<void_fancy_it, fallback_t>>::value);
  // works for a fancy iterator that has int as value type
  STATIC_REQUIRE(::cuda::std::is_same<int, //
                                      cub::detail::non_void_value_t<non_void_fancy_it, fallback_t>>::value);
  _CCCL_SUPPRESS_DEPRECATED_POP
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

using types = c2h::type_list<
  char,
  signed char,
  unsigned char,
  short,
  unsigned short,
  int,
  unsigned int,
  long,
  unsigned long,
  long long,
  unsigned long long,
#ifdef TEST_HALF_T
  __half,
  half_t,
#endif // TEST_HALF_T
#ifdef TEST_BF_T
  __nv_bfloat16,
  bfloat16_t,
#endif // TEST_BF_T
  float,
  double
#ifndef _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  ,
  long double
#endif // _LIBCUDACXX_HAS_NO_LONG_DOUBLE
  >;

C2H_TEST("Test FpLimits agrees with numeric_limits", "[util][type]", types)
{
  using T = c2h::get<0, TestType>;
  CAPTURE(c2h::type_name<T>());
  _CCCL_SUPPRESS_DEPRECATED_PUSH
  CHECK(cub::FpLimits<T>::Max() == cuda::std::numeric_limits<T>::max());
  CHECK(cub::FpLimits<T>::Lowest() == cuda::std::numeric_limits<T>::lowest());

  CHECK(cub::FpLimits<const T>::Max() == cuda::std::numeric_limits<const T>::max());
  CHECK(cub::FpLimits<const T>::Lowest() == cuda::std::numeric_limits<const T>::lowest());
  _CCCL_SUPPRESS_DEPRECATED_POP
}
