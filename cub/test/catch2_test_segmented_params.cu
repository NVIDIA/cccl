// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include <cub/detail/segmented_params.cuh>

#include <thrust/detail/raw_pointer_cast.h>

#include <cuda/argument>
#include <cuda/std/__utility/cmp.h> // cmp_equal
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/type_traits>

#include "cub_test_macros.h"
#include <c2h/vector.h>

// The type an argument annotation is rewritten to before it may appear in a kernel or policy-selector type. Spelled the
// way the dispatch spells it, so these tests break if the dispatch-facing signature changes.
template <class ParamT>
using normalized_t = decltype(cub::detail::params::normalize_param(cuda::std::declval<ParamT>()));

// Whether `constant<Value>` normalizes to `constant<ExpectedValue>` while still holding `Value`. The expectation is
// spelled as a value rather than a type so that the expected element type is visible at the call site; comparing the
// whole type pins both the non-type template argument and its type, which is what kernel identity depends on.
//
// Keep this body free of `typename normalized_param_t::value_type` and of a `static_cast` in a template argument:
// cudafe++ in nvcc 12.4 to 12.6 fails to re-emit either construct when it is spelled on a dependent `decltype` alias,
// which is the same host-pass regeneration defect these tests cover.
template <auto Value, auto ExpectedValue>
[[nodiscard]] constexpr bool constant_normalizes_to() noexcept
{
  using normalized_param_t = normalized_t<cuda::args::constant<Value>>;

  return cuda::std::is_same_v<normalized_param_t, cuda::args::constant<ExpectedValue>>
      && cuda::std::cmp_equal(normalized_param_t::__get_value(), Value);
}

template <class T>
inline constexpr T lowest_v = cuda::std::numeric_limits<T>::lowest();

template <class T>
inline constexpr T highest_v = cuda::std::numeric_limits<T>::max();

using i32 = cuda::std::int32_t;
using i64 = cuda::std::int64_t;
using u64 = cuda::std::uint64_t;

CUB_TEST("cub::detail::params::normalize_param derives a constant's element type from its value",
         "[params][utils]",
         CUB_SMALL)
{
  // The same value spelled with different integer types must select the same kernel. This is the property the
  // normalization exists for: nvcc 12.4 to 12.6 re-emit `constant<i64{384}>` as `constant<384>` in the host pass.
  STATIC_REQUIRE(!cuda::std::is_same_v<cuda::args::constant<384>, cuda::args::constant<i64{384}>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::constant<384>>, //
                                      normalized_t<cuda::args::constant<i64{384}>>>);

  // The element type is the narrowest of int32_t, int64_t, uint64_t that holds the value...
  STATIC_REQUIRE(constant_normalizes_to<384, i32{384}>());
  STATIC_REQUIRE(constant_normalizes_to<i64{384}, i32{384}>());
  STATIC_REQUIRE(constant_normalizes_to<u64{384}, i32{384}>());
  STATIC_REQUIRE(constant_normalizes_to<0, i32{0}>());
  STATIC_REQUIRE(constant_normalizes_to<-1, i32{-1}>());
  STATIC_REQUIRE(constant_normalizes_to<highest_v<i32>, highest_v<i32>>());
  STATIC_REQUIRE(constant_normalizes_to<lowest_v<i32>, lowest_v<i32>>());

  STATIC_REQUIRE(constant_normalizes_to<i64{highest_v<i32>} + 1, i64{highest_v<i32>} + 1>());
  STATIC_REQUIRE(constant_normalizes_to<i64{lowest_v<i32>} - 1, i64{lowest_v<i32>} - 1>());
  STATIC_REQUIRE(constant_normalizes_to<highest_v<i64>, highest_v<i64>>());
  STATIC_REQUIRE(constant_normalizes_to<lowest_v<i64>, lowest_v<i64>>());

  STATIC_REQUIRE(constant_normalizes_to<u64{highest_v<i64>} + 1, u64{highest_v<i64>} + 1>());
  STATIC_REQUIRE(constant_normalizes_to<highest_v<u64>, highest_v<u64>>());

  // ... and never the type the argument was spelled with, including the explicit second template argument.
  STATIC_REQUIRE(constant_normalizes_to<static_cast<signed char>(3), i32{3}>());
  STATIC_REQUIRE(constant_normalizes_to<static_cast<unsigned char>(3), i32{3}>());
  STATIC_REQUIRE(constant_normalizes_to<static_cast<short>(3), i32{3}>());
  STATIC_REQUIRE(constant_normalizes_to<3u, i32{3}>());
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::constant<384, short>>, cuda::args::constant<i32{384}>>);

  // Normalizing an already normalized argument is a no-op, so a dispatch may normalize defensively.
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<normalized_t<cuda::args::constant<i64{384}>>>,
                                      normalized_t<cuda::args::constant<i64{384}>>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<normalized_t<cuda::args::constant<highest_v<u64>>>>,
                                      normalized_t<cuda::args::constant<highest_v<u64>>>>);
}

CUB_TEST("cub::detail::params::normalize_param leaves non-integer constants unchanged", "[params][utils]", CUB_SMALL)
{
  // Enumerators and pointers spell their type in any constant expression, so they survive the host re-emission and are
  // kept as is. `char` and `bool` are not `__cccl_is_integer` types and take the same path.
  enum class direction
  {
    forward = 1
  };

  using enum_constant_t    = cuda::args::constant<direction::forward>;
  using pointer_constant_t = cuda::args::constant<(static_cast<const int*>(nullptr))>;
  using char_constant_t    = cuda::args::constant<'a'>;
  using bool_constant_t    = cuda::args::constant<true>;

  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<enum_constant_t>, enum_constant_t>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<pointer_constant_t>, pointer_constant_t>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<char_constant_t>, char_constant_t>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<bool_constant_t>, bool_constant_t>);

  // Constant sequences carry no `auto` non-type template parameter of their own to rewrite.
  using constant_sequence_t = cuda::args::__constant_sequence<(static_cast<const int*>(nullptr))>;
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<constant_sequence_t>, constant_sequence_t>);
}

CUB_TEST("cub::detail::params::normalize_param re-types static bounds with the wrapped element type",
         "[params][utils]",
         CUB_SMALL)
{
  // Same bounds, two spellings: `static_bounds` deduces its endpoint type from the non-type template parameters, so
  // the wide spelling and the plain one are distinct types until they are re-typed with the element type of the
  // wrapped argument (`int` for both wrappers below).
  using wide_bounds_t  = cuda::args::static_bounds<i64{0}, i64{100}>;
  using plain_bounds_t = cuda::args::static_bounds<0, 100>;
  STATIC_REQUIRE(!cuda::std::is_same_v<wide_bounds_t, plain_bounds_t>);

  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::immediate<int, wide_bounds_t>>,
                                      cuda::args::immediate<int, plain_bounds_t>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::__immediate_sequence<const int*, wide_bounds_t>>,
                                      cuda::args::__immediate_sequence<const int*, plain_bounds_t>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::deferred<const int*, wide_bounds_t>>,
                                      cuda::args::deferred<const int*, plain_bounds_t>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::deferred_sequence<const int*, wide_bounds_t>>,
                                      cuda::args::deferred_sequence<const int*, plain_bounds_t>>);

  // The element type drives the re-typing, not the wrapped handle type.
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::deferred_sequence<const i64*, wide_bounds_t>>,
                                      cuda::args::deferred_sequence<const i64*, wide_bounds_t>>);

  // `no_bounds` has no endpoints to re-type and passes through.
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::immediate<int>>, cuda::args::immediate<int>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::deferred<const int*>>, //
                                      cuda::args::deferred<const int*>>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<cuda::args::deferred_sequence<const int*>>,
                                      cuda::args::deferred_sequence<const int*>>);

  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<normalized_t<cuda::args::immediate<int, wide_bounds_t>>>,
                                      normalized_t<cuda::args::immediate<int, wide_bounds_t>>>);
}

CUB_TEST("cub::detail::params::normalize_param passes plain values through", "[params][utils]", CUB_SMALL)
{
  // Plain values carry no `auto` non-type template parameter, so their type is kept verbatim -- in particular a
  // runtime `int64_t` count is never narrowed the way `constant<int64_t{...}>` is.
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<int>, int>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<i64>, i64>);
  STATIC_REQUIRE(cuda::std::is_same_v<normalized_t<const i64&>, i64>);

  CHECK(cub::detail::params::normalize_param(i64{384}) == 384);
}

CUB_TEST("cub::detail::params::normalize_param preserves runtime arguments and bounds", "[params][utils]", CUB_SMALL)
{
  const auto immediate_arg        = cuda::args::immediate{7, cuda::args::static_bounds<i64{0}, i64{100}>{}};
  const auto normalized_immediate = cub::detail::params::normalize_param(immediate_arg);

  STATIC_REQUIRE(cuda::std::is_same_v<cuda::std::remove_const_t<decltype(normalized_immediate)>,
                                      cuda::args::immediate<int, cuda::args::static_bounds<0, 100>>>);
  CHECK(cuda::args::__access::__arg(normalized_immediate) == 7);

  const int values[]      = {1, 2, 3};
  const auto sequence_arg = cuda::args::deferred_sequence{
    +values, cuda::args::static_bounds<i64{1}, i64{3}>{}, cuda::args::runtime_bounds{1, 3}};
  const auto normalized_sequence = cub::detail::params::normalize_param(sequence_arg);

  STATIC_REQUIRE(cuda::std::is_same_v<cuda::std::remove_const_t<decltype(normalized_sequence)>,
                                      cuda::args::deferred_sequence<const int*, cuda::args::static_bounds<1, 3>>>);
  CHECK(cuda::args::__access::__arg(normalized_sequence) == +values);
  CHECK(cuda::args::__access::__runtime_bounds(normalized_sequence).lower() == 1);
  CHECK(cuda::args::__access::__runtime_bounds(normalized_sequence).upper() == 3);
}

// The wrapper reaches the kernel both as a template argument and as a kernel parameter, the way DeviceBatchedTopK
// passes it. Both are needed to expose the defect below; a kernel that only takes the wrapper as a template argument
// is re-emitted correctly.
template <class ParamT>
__global__ void write_param_value(ParamT param, i64* d_result)
{
  *d_result = static_cast<i64>(decltype(param)::__get_value());
}

template <class ParamT>
[[nodiscard]] cudaError_t launch_with_param(ParamT param, i64* d_result)
{
  write_param_value<<<1, 1>>>(param, d_result);
  return cudaGetLastError();
}

// Two functions must name the same specialization from a local `constexpr` int64_t, and they must stay separate
// functions: cudafe++ re-emits the argument correctly for the function that *first* names the specialization and as
// the untyped initializer for every other one, so a single call site never fails. Keep both.
[[nodiscard]] cudaError_t launch_from_first_naming(i64* d_result)
{
  constexpr i64 segment_size = 384;
  return launch_with_param(cub::detail::params::normalize_param(cuda::args::constant<segment_size>{}), d_result);
}

[[nodiscard]] cudaError_t launch_from_second_naming(i64* d_result)
{
  constexpr i64 segment_size = 384;
  return launch_with_param(cub::detail::params::normalize_param(cuda::args::constant<segment_size>{}), d_result);
}

CUB_TEST("cub::detail::params::normalize_param names the same kernel in both compilation passes",
         "[params][utils]",
         CUB_SMALL)
{
  // Regression test for nvcc 12.4 to 12.6: cudafe++ re-emits the host translation unit with the argument of an `auto`
  // non-type template parameter printed as the untyped initializer of the variable that first named the
  // specialization, so `constant<segment_size>` reaches the host compiler as `constant<384>` and deduces `int` while
  // the device pass deduced `long`. The host then instantiates a kernel stub that has no device code and the launch
  // fails with cudaErrorInvalidDeviceFunction, undiagnosed at compile or link time. Normalizing rebuilds the wrapper
  // from the value alone, so both passes name the same kernel. Verified to fail on nvcc 12.6 without the
  // normalization (second launch only) and to pass with it.
  constexpr i64 segment_size = 384;

  c2h::device_vector<i64> result(1, 0);
  i64* d_result = thrust::raw_pointer_cast(result.data());

  REQUIRE_CUDART(launch_from_first_naming(d_result));
  REQUIRE_CUDART(cudaDeviceSynchronize());
  CHECK(result[0] == segment_size);

  result[0] = 0;
  REQUIRE_CUDART(launch_from_second_naming(d_result));
  REQUIRE_CUDART(cudaDeviceSynchronize());
  CHECK(result[0] == segment_size);
}
