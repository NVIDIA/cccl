/******************************************************************************
* Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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

#pragma once

#include <metal.hpp>
#include <type_traits>
#include <cstdint>
#include <tuple>

#include <thrust/host_vector.h>
#include <thrust/device_vector.h>

#include "cub/util_compiler.cuh"
#include "test_util_vec.h"

// cudafe considers some catch2 variables as unused. We have to suppress
// these warnings until NVBug 3423950 is addressed.
#if (CUB_HOST_COMPILER == CUB_HOST_COMPILER_MSVC) || \
    (CUB_HOST_COMPILER == CUB_HOST_COMPILER_CLANG) || \
    defined(__ICC) || defined(_NVHPC_CUDA)
#  if defined(__NVCC_DIAG_PRAGMA_SUPPORT__)
#    pragma nv_diag_suppress 177
#  else
#    pragma diag_suppress 177
#  endif
#endif

#include "catch2_main.cuh"

#ifndef VAR_IDX
#define VAR_IDX 0
#endif

namespace c2h
{

template <typename... Ts>
using type_list = metal::list<Ts...>;

template <typename TypeList>
using size = metal::size<TypeList>;

template <std::size_t Index, typename TypeList>
using get = metal::at<TypeList, metal::number<Index>>;

template <class... TypeLists>
using cartesian_product = metal::cartesian<TypeLists...>;

template <typename T, T... Ts>
using enum_type_list = c2h::type_list<std::integral_constant<T, Ts>...>;

template <typename T0, typename T1>
using pair = metal::pair<T0, T1>;

template <typename P>
using first = metal::first<P>;

template <typename P>
using second = metal::second<P>;

template <std::size_t start, std::size_t size, std::size_t stride = 1>
using iota = metal::iota<metal::number<start>, metal::number<size>, metal::number<stride>>;

} // namespace c2h 

namespace detail
{
  template <class T>
  std::vector<T> to_vec(thrust::device_vector<T> const& vec)
  {
    thrust::host_vector<T> tmp = vec;
    return std::vector<T>{tmp.begin(), tmp.end()};
  }

  template <class T>
  std::vector<T> to_vec(thrust::host_vector<T> const& vec)
  {
    return std::vector<T>{vec.begin(), vec.end()};
  }

  template <class T>
  std::vector<T> to_vec(std::vector<T> const& vec)
  {
    return vec;
  }
}

#define REQUIRE_APPROX_EQ(ref,out) { \
  auto vec_ref = detail::to_vec(ref);  \
  auto vec_out = detail::to_vec(out);  \
  REQUIRE_THAT(vec_ref, Catch::Approx(vec_out)); \
}

#include <c2h/generators.cuh>
#include <c2h/custom_type.cuh>


#define CUB_TEST_NAME_IMPL(NAME, PARAM) \
  CUB_TEST_STR(NAME) "(" CUB_TEST_STR(PARAM) ")"

#define CUB_TEST_NAME(NAME) \
  CUB_TEST_NAME_IMPL(NAME, VAR_IDX)

#define CUB_TEST_CONCAT(A, B) CUB_TEST_CONCAT_INNER(A, B)
#define CUB_TEST_CONCAT_INNER(A, B) A ## B

#define CUB_TEST_IMPL(ID, NAME, TAG, ...)                                       \
  using CUB_TEST_CONCAT(types_, ID) =                                           \
    c2h::cartesian_product<__VA_ARGS__>;                                        \
  TEMPLATE_LIST_TEST_CASE(CUB_TEST_NAME(NAME), TAG, CUB_TEST_CONCAT(types_, ID))

#define CUB_TEST(NAME, TAG, ...) \
  CUB_TEST_IMPL(__LINE__, NAME, TAG, __VA_ARGS__)

#define CUB_TEST_LIST_IMPL(ID, NAME, TAG, ...)                                  \
  using CUB_TEST_CONCAT(types_, ID) =                                           \
    c2h::type_list<__VA_ARGS__>;                                                \
  TEMPLATE_LIST_TEST_CASE(CUB_TEST_NAME(NAME), TAG, CUB_TEST_CONCAT(types_, ID))

#define CUB_TEST_LIST(NAME, TAG, ...) \
  CUB_TEST_LIST_IMPL(__LINE__, NAME, TAG, __VA_ARGS__)

#define CUB_TEST_STR(a) #a

#define CUB_SEED(N)                                                            \
  c2h::seed_t{                                                                 \
    GENERATE_COPY(                                                             \
      take(N,                                                                  \
           random(std::numeric_limits<unsigned long long int>::min(),          \
                  std::numeric_limits<unsigned long long int>::max())))        \
  }

