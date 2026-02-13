// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/cstdint>
#include <cuda/std/iterator>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::params
{
// -----------------------------------------------------------------------------
// Parameter Mixins and Helpers
// -----------------------------------------------------------------------------

// Allows providing constrains on parameter values at compile time
template <typename T, T Min = ::cuda::std::numeric_limits<T>::lowest(), T Max = ::cuda::std::numeric_limits<T>::max()>
struct static_bounds_mixin
{
  static_assert(Min <= Max, "Min must be <= Max");

  // Compile-time bounds
  static constexpr T static_min_value = Min;
  static constexpr T static_max_value = Max;

  // Indicates that there's only one possible value
  static constexpr bool is_exact = (Min == Max);
};

// Allows specifying a list of supported options for a parameter. E.g., the orders (ascending, descending) that are
// supported by a sorting algorithm.
template <typename T, T... Options>
struct supported_options
{
  static constexpr size_t count = sizeof...(Options);
};

// -----------------------------------------------------------------------------
// Fundamental Parameter Types
// -----------------------------------------------------------------------------

// A compile-time constant
template <typename T, T Value>
struct static_constant_param : public static_bounds_mixin<T, Value, Value>
{
  using value_type = T;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE constexpr auto get_param([[maybe_unused]] SegmentIndexT segment_id) const
  {
    static_assert(static_bounds_mixin<T, Value, Value>::is_exact, "Static parameter must have exact value");
    return static_bounds_mixin<T, Value, Value>::static_min_value;
  }
};
// -----------------------------------------------------------------------------
// 1. Uniform Param
// -----------------------------------------------------------------------------
// Added default template args so CTAD can deduce T and default Min/Max
template <typename T, T Min = ::cuda::std::numeric_limits<T>::lowest(), T Max = ::cuda::std::numeric_limits<T>::max()>
struct uniform_param : public static_bounds_mixin<T, Min, Max>
{
  using value_type = T;

  T value;

  _CCCL_HOST_DEVICE constexpr uniform_param(T v)
      : value(v)
  {}

  uniform_param() = default;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE constexpr auto get_param([[maybe_unused]] SegmentIndexT segment_id) const
  {
    return value;
  }
};

template <typename T>
uniform_param(T) -> uniform_param<T>;

// -----------------------------------------------------------------------------
// 2. Per-Segment Param
// -----------------------------------------------------------------------------
// Added defaults for T, Min, and Max based on the Iterator's value_type
template <typename IteratorT,
          typename T = typename ::cuda::std::iterator_traits<IteratorT>::value_type,
          T Min      = ::cuda::std::numeric_limits<T>::lowest(),
          T Max      = ::cuda::std::numeric_limits<T>::max()>
struct per_segment_param : public static_bounds_mixin<T, Min, Max>
{
  using iterator_type = IteratorT;
  using value_type    = T;

  IteratorT iterator;
  T min_value = Min;
  T max_value = Max;

  _CCCL_HOST_DEVICE constexpr per_segment_param(IteratorT iter, T min_v = Min, T max_v = Max)
      : iterator(iter)
      , min_value(min_v)
      , max_value(max_v)
  {}

  per_segment_param() = default;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE constexpr auto get_param(SegmentIndexT segment_id) const
  {
    return iterator[segment_id];
  }
};

// Deduction Guide:
// Allows: per_segment_param{iter} -> per_segment_param<IteratorT, ValueT, Min,
// Max>
template <typename IteratorT>
per_segment_param(IteratorT) -> per_segment_param<IteratorT>;

// -----------------------------------------------------------------------------
// 3. Uniform Discrete Param
// -----------------------------------------------------------------------------
template <typename T, T... Options>
struct uniform_discrete_param
{
  using value_type          = T;
  using supported_options_t = supported_options<T, Options...>;

  T value;

  _CCCL_HOST_DEVICE constexpr uniform_discrete_param(T v)
      : value(v)
  {}

  uniform_discrete_param() = default;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE constexpr auto get_param([[maybe_unused]] SegmentIndexT segment_id) const
  {
    return value;
  }
};

// -----------------------------------------------------------------------------
// 4. Per-Segment Discrete Param
// -----------------------------------------------------------------------------
template <typename IteratorT, typename T, T... Options>
struct per_segment_discrete_param
{
  using iterator_type       = IteratorT;
  using value_type          = T;
  using supported_options_t = supported_options<T, Options...>;

  IteratorT iterator;

  _CCCL_HOST_DEVICE constexpr per_segment_discrete_param(IteratorT iter)
      : iterator(iter)
  {}

  per_segment_discrete_param() = default;

  template <typename SegmentIndexT>
  _CCCL_HOST_DEVICE constexpr auto get_param(SegmentIndexT segment_id) const
  {
    return iterator[segment_id];
  }
};

// -----------------------------------------------------------------------------
// Parameter Type Helpers
// -----------------------------------------------------------------------------
template <typename T>
inline constexpr bool is_static_param_v = false;

template <typename T, T Value>
inline constexpr bool is_static_param_v<static_constant_param<T, Value>> = true;

template <typename T>
inline constexpr bool is_uniform_param_v = false;

template <typename T, T Min, T Max>
inline constexpr bool is_uniform_param_v<uniform_param<T, Min, Max>> = true;

template <typename T, T... Options>
inline constexpr bool is_uniform_param_v<uniform_discrete_param<T, Options...>> = true;

template <typename T>
inline constexpr bool is_per_segment_param_v = false;

template <typename IteratorT, typename T, T Min, T Max>
inline constexpr bool is_per_segment_param_v<per_segment_param<IteratorT, T, Min, Max>> = true;

template <typename IteratorT, typename T, T... Options>
inline constexpr bool is_per_segment_param_v<per_segment_discrete_param<IteratorT, T, Options...>> = true;

// Get max value (works for all types inheriting bounds_mixin)
template <typename T>
inline constexpr auto static_max_value_v = T::static_max_value;

// Get min value (works for all types inheriting bounds_mixin)
template <typename T>
inline constexpr auto static_min_value_v = T::static_min_value;

// Whether a given parameter allows only for a single static value
template <typename T>
inline constexpr bool has_single_static_value_v = (static_max_value_v<T> == static_min_value_v<T>);

// Helper that translates a runtime parameter value into a compile-time constant by matching against a list of supported
// options.
template <typename T, T... Opts, typename Functor>
_CCCL_HOST_DEVICE bool dispatch_impl(T val, supported_options<T, Opts...>, Functor&& f)
{
  // Fold expression over the supported options.
  // This generates code equivalent to:
  // if (val == Opt1) f(integral_constant<Opt1>);
  // else if (val == Opt2) f(integral_constant<Opt2>);
  // ...
  const bool match_found = ((val == Opts ? (f(::cuda::std::integral_constant<T, Opts>{}), true) : false) || ...);

  // Optional: Handling cases where the runtime value was not in the supported
  // list. In a release build, we assume the user respected the contract.
  _CCCL_ASSERT(match_found, "The given runtime parameter value is not in the supported list");
  return match_found;
}

// Dispatcher that matches a runtime parameter value against a list of supported options and invokes a functor with the
// matched option as a compile-time constant.
template <typename ParamT, typename SegmentIndexT, typename Functor>
_CCCL_HOST_DEVICE bool dispatch_discrete(ParamT param, SegmentIndexT segment_id, Functor&& f)
{
  using supported_list = typename ParamT::supported_options_t;
  auto param_value     = param.get_param(segment_id);
  return dispatch_impl(param_value, supported_list{}, ::cuda::std::forward<Functor>(f));
}
} // namespace detail::params

CUB_NAMESPACE_END
