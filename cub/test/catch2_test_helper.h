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

#include <cub/util_compiler.cuh>

#include <cstdint>
#include <cstdlib>
#include <tuple>
#include <type_traits>

#include "cuda/std/__cccl/diagnostic.h"
#include "test_util_vec.h"

#if __CUDACC_VER_MAJOR__ == 11
_CCCL_NV_DIAG_SUPPRESS(177) // catch2 may contain unused variableds
#endif // nvcc-11

#include <cuda/std/bit>
#include <cuda/std/cmath>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include "catch2_main.cuh"
#include "test_warning_suppression.cuh"
#include <c2h/device_policy.cuh>
#include <c2h/utility.cuh>
#include <c2h/vector.cuh>

#ifndef VAR_IDX
#  define VAR_IDX 0
#endif

namespace c2h
{

template <typename... Ts>
using type_list = ::cuda::std::__type_list<Ts...>;

template <typename TypeList>
using size = ::cuda::std::__type_list_size<TypeList>;

template <std::size_t Index, typename TypeList>
using get = ::cuda::std::__type_at_c<Index, TypeList>;

template <class... TypeLists>
using cartesian_product = ::cuda::std::__type_cartesian_product<TypeLists...>;

template <typename T, T... Ts>
using enum_type_list = ::cuda::std::__type_value_list<T, Ts...>;

template <typename T0, typename T1>
using pair = ::cuda::std::__type_pair<T0, T1>;

template <typename P>
using first = ::cuda::std::__type_pair_first<P>;

template <typename P>
using second = ::cuda::std::__type_pair_second<P>;

template <std::size_t Start, std::size_t Size, std::size_t Stride = 1>
using iota = ::cuda::std::__type_iota<std::size_t, Start, Size, Stride>;

template <typename TypeList, typename T>
using remove = ::cuda::std::__type_remove<TypeList, T>;

} // namespace c2h

namespace detail
{
template <class T>
std::vector<T> to_vec(c2h::device_vector<T> const& vec)
{
  c2h::host_vector<T> temp = vec;
  return std::vector<T>{temp.begin(), temp.end()};
}

template <class T>
std::vector<T> to_vec(c2h::host_vector<T> const& vec)
{
  return std::vector<T>{vec.begin(), vec.end()};
}

template <class T>
std::vector<T> to_vec(std::vector<T> const& vec)
{
  return vec;
}
} // namespace detail

#define REQUIRE_APPROX_EQ(ref, out)                \
  {                                                \
    auto vec_ref = detail::to_vec(ref);            \
    auto vec_out = detail::to_vec(out);            \
    REQUIRE_THAT(vec_ref, Catch::Approx(vec_out)); \
  }

namespace detail
{
// Returns true if values are equal, or both NaN:
struct equal_or_nans
{
  template <typename T>
  bool operator()(const T& a, const T& b) const
  {
    return (cuda::std::isnan(a) && cuda::std::isnan(b)) || a == b;
  }
};

struct bitwise_equal
{
  template <typename T>
  bool operator()(const T& a, const T& b) const
  {
    using bits_t  = typename cub::Traits<T>::UnsignedBits;
    bits_t a_bits = ::cuda::std::bit_cast<bits_t>(a);
    bits_t b_bits = ::cuda::std::bit_cast<bits_t>(b);
    return a_bits == b_bits;
  }
};

// Catch2 Matcher that calls `std::equal` with a default-constructable custom predicate
template <typename Range, typename Pred>
struct CustomEqualsRangeMatcher : Catch::MatcherBase<Range>
{
  CustomEqualsRangeMatcher(Range const& range)
      : range{range}
  {}

  bool match(Range const& other) const override
  {
    using std::begin;
    using std::end;

    return std::equal(begin(range), end(range), begin(other), Pred{});
  }

  std::string describe() const override
  {
    return "Equals: " + Catch::rangeToString(range);
  }

private:
  Range const& range;
};

template <typename Range>
auto NaNEqualsRange(const Range& range) -> CustomEqualsRangeMatcher<Range, equal_or_nans>
{
  return CustomEqualsRangeMatcher<Range, equal_or_nans>(range);
}

template <typename Range>
auto BitwiseEqualsRange(const Range& range) -> CustomEqualsRangeMatcher<Range, bitwise_equal>
{
  return CustomEqualsRangeMatcher<Range, bitwise_equal>(range);
}
} // namespace detail

#define REQUIRE_EQ_WITH_NAN_MATCHING(ref, out)              \
  {                                                         \
    auto vec_ref = detail::to_vec(ref);                     \
    auto vec_out = detail::to_vec(out);                     \
    REQUIRE_THAT(vec_ref, detail::NaNEqualsRange(vec_out)); \
  }

#define REQUIRE_BITWISE_EQ(ref, out)                        \
  {                                                         \
    auto vec_ref = detail::to_vec(ref);                     \
    auto vec_out = detail::to_vec(out);                     \
    REQUIRE_THAT(vec_ref, detail::NaNEqualsRange(vec_out)); \
  }

#include <cuda/std/tuple>
_LIBCUDACXX_BEGIN_NAMESPACE_STD
template <size_t N, typename... T>
__enable_if_t<(N == sizeof...(T))> print_elem(::std::ostream&, const tuple<T...>&)
{}

template <size_t N, typename... T>
__enable_if_t<(N < sizeof...(T))> print_elem(::std::ostream& os, const tuple<T...>& tup)
{
  _CCCL_IF_CONSTEXPR (N != 0)
  {
    os << ", ";
  }
  os << _CUDA_VSTD::get<N>(tup);
  _CUDA_VSTD::print_elem<N + 1>(os, tup);
}

template <typename... T>
::std::ostream& operator<<(::std::ostream& os, const tuple<T...>& tup)
{
  os << "[";
  _CUDA_VSTD::print_elem<0>(os, tup);
  return os << "]";
}
_LIBCUDACXX_END_NAMESPACE_STD

template <>
struct Catch::StringMaker<cudaError>
{
  static auto convert(cudaError e) -> std::string
  {
    return std::to_string(cuda::std::__to_underlying(e)) + " (" + cudaGetErrorString(e) + ")";
  }
};

#include <c2h/custom_type.cuh>
#include <c2h/generators.cuh>

#define CUB_TEST_NAME_IMPL(NAME, PARAM) CUB_TEST_STR(NAME) "(" CUB_TEST_STR(PARAM) ")"

#define CUB_TEST_NAME(NAME) CUB_TEST_NAME_IMPL(NAME, VAR_IDX)

#define CUB_TEST_CONCAT(A, B)       CUB_TEST_CONCAT_INNER(A, B)
#define CUB_TEST_CONCAT_INNER(A, B) A##B

#define CUB_TEST_IMPL(ID, NAME, TAG, ...)                                  \
  using CUB_TEST_CONCAT(types_, ID) = c2h::cartesian_product<__VA_ARGS__>; \
  TEMPLATE_LIST_TEST_CASE(CUB_TEST_NAME(NAME), TAG, CUB_TEST_CONCAT(types_, ID))

#define CUB_TEST(NAME, TAG, ...) CUB_TEST_IMPL(__LINE__, NAME, TAG, __VA_ARGS__)

#define CUB_TEST_LIST_IMPL(ID, NAME, TAG, ...)                     \
  using CUB_TEST_CONCAT(types_, ID) = c2h::type_list<__VA_ARGS__>; \
  TEMPLATE_LIST_TEST_CASE(CUB_TEST_NAME(NAME), TAG, CUB_TEST_CONCAT(types_, ID))

#define CUB_TEST_LIST(NAME, TAG, ...) CUB_TEST_LIST_IMPL(__LINE__, NAME, TAG, __VA_ARGS__)

#define CUB_TEST_STR(a) #a

namespace detail
{
inline std::size_t adjust_seed_count(std::size_t requested)
{
  // Setting this environment variable forces a fixed number of seeds to be generated, regardless of the requested
  // count. Set to 1 to reduce redundant, expensive testing when using sanitizers, etc.
  static const char* override_str = std::getenv("CCCL_SEED_COUNT_OVERRIDE");
  static int override             = override_str ? std::atoi(override_str) : 0;
  return override_str ? override : requested;
}
} // namespace detail

#define CUB_SEED(N)                                                                                                    \
  c2h::seed_t                                                                                                          \
  {                                                                                                                    \
    GENERATE_COPY(take(                                                                                                \
      detail::adjust_seed_count(N),                                                                                    \
      random(std::numeric_limits<unsigned long long int>::min(), std::numeric_limits<unsigned long long int>::max()))) \
  }
