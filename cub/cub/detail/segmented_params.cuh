// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cuda/argument>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__iterator/concepts.h> // indirectly_readable, random_access_iterator
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integer.h> // __cccl_is_integer_v
#include <cuda/std/__type_traits/is_integral.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_signed.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h> // cmp_greater_equal, cmp_less_equal
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN

namespace detail::params
{
// =====================================================================
// Deferred handle requirements
// =====================================================================

// Rejecting range/container arguments (span, array, ...) is a deliberate narrowing of CUB's API surface, not a
// limitation of the cuda::args annotation framework (which accepts ranges by design): no CUB device API takes a range
// today, and annotating one would be unclear to users (does a bound constrain the element values or the range's
// size?). A revisitable decision -- the traits below accept only dereferenceable handles / random-access iterators.

// A `deferred` is read by dereferencing its handle (`*handle`, see the get_param overload below), so the handle must
// be indirectly readable (a pointer or other dereferenceable handle); use `deferred_sequence` for per-segment values.
template <class _Handle>
inline constexpr bool __is_valid_deferred_handle_v = ::cuda::std::indirectly_readable<_Handle>;

// A `deferred_sequence` is indexed per segment (`handle[index]`), so its handle must be a random-access iterator.
template <class _Handle>
inline constexpr bool __is_valid_deferred_sequence_handle_v = ::cuda::std::random_access_iterator<_Handle>;

// =====================================================================
// Argument-form validation (compile-time)
// =====================================================================

// Compile-time validation of an integer argument that may be supplied uniformly or per segment (e.g.
// segment_sizes, k), matching how get_param reads it: a deferred handle or a sequence is read element-wise, so its
// `element_type` is the value; a plain value / `constant` / `immediate` is used directly, so its `value_type` is the
// value (which also rejects a handle wrongly wrapped as a single value, e.g. a pointer in `immediate`). Instantiating
// the struct runs the layered static_asserts below (each gated on the previous so a single misuse yields one targeted
// diagnostic); `all_ok` lets the caller gate the downstream dispatch to avoid follow-on cascades. Any argument-specific
// range/bound check is left to the caller.
template <class _Param>
struct __validate_uniform_or_per_segment_integral_param
{
  using args_traits = ::cuda::args::__traits<_Param>;

  static constexpr bool is_valid_type = ::cuda::args::__is_wrapper_v<_Param> || ::cuda::std::is_integral_v<_Param>;

  using __value_t = ::cuda::std::conditional_t<args_traits::is_deferred || !args_traits::is_single_value,
                                               typename args_traits::element_type,
                                               typename args_traits::value_type>;
  static constexpr bool is_integral =
    ::cuda::std::is_integral_v<__value_t> && !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<__value_t>, bool>;

  static constexpr bool is_deferred_single = args_traits::is_deferred && args_traits::is_single_value;
  static constexpr bool is_deferred_seq    = args_traits::is_deferred && !args_traits::is_single_value;
  static constexpr bool handle_ok =
    (!is_deferred_single || __is_valid_deferred_handle_v<typename args_traits::value_type>)
    && (!is_deferred_seq || __is_valid_deferred_sequence_handle_v<typename args_traits::value_type>);

  static constexpr bool all_ok = is_valid_type && is_integral && handle_ok;

  static_assert(is_valid_type,
                "cub: a uniform-or-per-segment integer argument (e.g. cub::DeviceBatchedTopK segment_sizes or k) must "
                "be a cuda::args annotation or a plain integral value (taken as a uniform immediate). A raw pointer or "
                "iterator is not interpreted as a sequence. Wrap per-segment values in cuda::args::deferred_sequence, "
                "or a single device-side value in cuda::args::deferred.");
  static_assert(!is_valid_type || is_integral,
                "cub: a uniform-or-per-segment integer argument (e.g. cub::DeviceBatchedTopK segment_sizes or k) must "
                "have an integral (non-bool) element type (it is a count of items).");
  static_assert(
    !is_valid_type || !is_integral || !is_deferred_single
      || __is_valid_deferred_handle_v<typename args_traits::value_type>,
    "cub: a uniform-or-per-segment integer argument passed via cuda::args::deferred must wrap a pointer or "
    "other dereferenceable handle to a single device-side integral value (it is read via *handle); a range "
    "or container such as span / array is not accepted -- use cuda::args::deferred_sequence for per-segment "
    "values.");
  static_assert(!is_valid_type || !is_integral || !is_deferred_seq
                  || __is_valid_deferred_sequence_handle_v<typename args_traits::value_type>,
                "cub: a uniform-or-per-segment integer argument passed via cuda::args::deferred_sequence must wrap a "
                "random-access iterator (a pointer qualifies) over the per-segment integral values (it is indexed per "
                "segment).");
};

// Compile-time validation of an integer argument that must be a single value, uniform across all segments (e.g.
// num_segments): a per-segment sequence (`deferred_sequence`) is rejected, but a single `deferred` (device-resident)
// value is allowed here. Read like the single-value forms above: a single deferred via *handle (`element_type`),
// everything else directly (`value_type`). A caller that additionally needs the value on the host (e.g. num_segments,
// which sizes the launch) checks `!args_traits::is_deferred` separately. Instantiating the struct runs the layered
// static_asserts (one diagnostic per misuse).
template <class _Param>
struct __validate_uniform_integral_param
{
  using args_traits = ::cuda::args::__traits<_Param>;

  static constexpr bool is_valid_type = ::cuda::args::__is_wrapper_v<_Param> || ::cuda::std::is_integral_v<_Param>;

  static constexpr bool is_single_value    = args_traits::is_single_value;
  static constexpr bool is_deferred_single = args_traits::is_deferred && args_traits::is_single_value;

  using __value_t = ::cuda::std::conditional_t<args_traits::is_deferred || !args_traits::is_single_value,
                                               typename args_traits::element_type,
                                               typename args_traits::value_type>;
  static constexpr bool is_integral =
    ::cuda::std::is_integral_v<__value_t> && !::cuda::std::is_same_v<::cuda::std::remove_cvref_t<__value_t>, bool>;

  static constexpr bool handle_ok =
    !is_deferred_single || __is_valid_deferred_handle_v<typename args_traits::value_type>;

  static constexpr bool all_ok = is_valid_type && is_single_value && is_integral && handle_ok;

  static_assert(is_valid_type,
                "cub: a uniform integer argument (e.g. cub::DeviceBatchedTopK num_segments) must be a cuda::args "
                "annotation or a plain integral value. A raw pointer or iterator is not accepted.");
  static_assert(!is_valid_type || is_single_value,
                "cub: a uniform integer argument (e.g. cub::DeviceBatchedTopK num_segments) must be a single value "
                "(the same for every segment); a per-segment sequence (cuda::args::deferred_sequence) is not "
                "accepted.");
  static_assert(!is_valid_type || !is_single_value || is_integral,
                "cub: a uniform integer argument (e.g. cub::DeviceBatchedTopK num_segments) must have an integral "
                "(non-bool) type.");
  static_assert(!is_valid_type || !is_single_value || !is_integral || !is_deferred_single
                  || __is_valid_deferred_handle_v<typename args_traits::value_type>,
                "cub: a uniform integer argument passed via cuda::args::deferred must wrap a pointer or other "
                "dereferenceable handle to a single device-side integral value (it is read via *handle).");
};

// =====================================================================
// Argument contract checks (debug-only)
// =====================================================================

// Shared debug-only contract check for the argument reads the host cannot validate at call time: a value read on the
// device in stream order from a `deferred`/`deferred_sequence` handle must lie within that argument's *effective*
// bounds -- the intersection of its static `cuda::args::bounds` and any runtime bounds, as computed by
// ::cuda::args::__lowest_/__highest_. A value outside them breaks the caller's promise and is otherwise undefined
// behavior. Compiled out when assertions are disabled (the bounds are not even computed). Gated on the argument's
// integer element type (not the read value type, which may be a proxy reference): `bool` and character types are
// excluded, matching the `cmp_*` comparators.
template <class _Arg, class _Value>
_CCCL_HOST_DEVICE constexpr void
__assert_param_in_bounds([[maybe_unused]] const _Arg& __arg, [[maybe_unused]] const _Value& __value) noexcept
{
  using __element_t = typename ::cuda::args::__traits<_Arg>::element_type;
  if constexpr (::cuda::std::__cccl_is_integer_v<__element_t>)
  {
    const __element_t __checked = static_cast<__element_t>(__value);
    _CCCL_ASSERT(::cuda::std::cmp_greater_equal(__checked, ::cuda::args::__lowest_(__arg))
                   && ::cuda::std::cmp_less_equal(__checked, ::cuda::args::__highest_(__arg)),
                 "cub argument value is outside its declared bounds");
  }
}

// =====================================================================
// get_param — unified segment parameter access
// =====================================================================

//! @brief Returns the value of an argument for a given segment index.
//!
//! @param[in] __arg Argument or argument wrapper to read.
//! @param[in] __index Segment index to read for sequence arguments.
//! @return The single argument value, or the sequence element at the given index.
_CCCL_TEMPLATE(class _Tp, class _SegmentIndexT)
_CCCL_REQUIRES((!::cuda::args::__is_wrapper_v<::cuda::std::remove_cvref_t<_Tp>>) )
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto get_param(_Tp&& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  if constexpr (::cuda::args::__traits<::cuda::std::remove_cvref_t<_Tp>>::is_single_value)
  {
    return __arg;
  }
  else
  {
    return __arg[__index];
  }
}

template <auto _Value, class _Tp, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::args::constant<_Value, _Tp>& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  return ::cuda::args::__unwrap(__arg);
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::args::immediate<_Arg, _StaticBounds>& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  return ::cuda::args::__unwrap(__arg);
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::args::deferred<_Arg, _StaticBounds>& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  // A single `deferred` wraps a handle to a device-side value (a pointer or input iterator), not the value itself;
  // the value is read on the device by dereferencing the handle.
  auto __value = *::cuda::args::__unwrap(__arg);
  __assert_param_in_bounds(__arg, __value);
  return __value;
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::args::deferred_sequence<_Arg, _StaticBounds>& __arg, _SegmentIndexT __index) noexcept
{
  auto __value = ::cuda::args::__unwrap(__arg)[__index];
  __assert_param_in_bounds(__arg, __value);
  return __value;
}

// Reads parameter `__index` and, when the argument's static lower bound is negative (e.g. an un-annotated `int16_t` or
// an explicit `bounds` with a negative lower end), clamps a negative runtime value up to 0 in the argument's own
// element type -- before any widening/narrowing cast, so a caller that later widens the result cannot reinterpret a
// negative value as a huge unsigned one. A negative count thus becomes "no work". Deferred forms are range-checked
// against their declared bounds by `get_param` in debug builds.
template <class _Arg, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
__get_and_clamp_param_to_nonnegative(const _Arg& __arg, _SegmentIndexT __index) noexcept
{
  // Materialize into the scalar element type: get_param yields a proxy reference for fancy iterators, and the clamp
  // must act on a real value (a `static_cast<proxy>(0)` would form a null proxy that `max` then dereferences).
  using __element_t         = typename ::cuda::args::__traits<_Arg>::element_type;
  const __element_t __value = get_param(__arg, __index);
  constexpr auto __lowest   = ::cuda::args::__traits<_Arg>::lowest;
  // Use a plain `<` against a same-typed zero, not the integer-only `cmp_*` comparators (which reject character element
  // types, see `__assert_param_in_bounds`); the `is_signed_v` guard skips the test for unsigned types, whose lower
  // bound is never negative.
  if constexpr (::cuda::std::is_signed_v<__element_t> && __lowest < __element_t{0})
  {
    return (::cuda::std::max) (__value, __element_t{0});
  }
  else
  {
    return __value;
  }
}

// =====================================================================
// Discrete parameter support
// =====================================================================

//! @brief Specifies a list of supported options for a parameter.
template <typename T, T... Options>
struct supported_options
{
  static constexpr ::cuda::std::size_t count = sizeof...(Options);
};

//! @brief Static discrete parameter — a single compile-time value that is also its only supported option.
//!
//! Holds no runtime value, so it cannot be put into a state that disagrees with its supported option, and
//! @c dispatch_impl therefore always matches it. This is the safe representation for a compile-time-fixed discrete
//! parameter (e.g. a statically known top-k selection direction): modeling such a parameter with a runtime value
//! instead would risk that value silently disagreeing with the supported option (a no-op dispatch unless
//! @c CCCL_ENABLE_ASSERTIONS is set).
template <typename T, T Value>
struct static_discrete_param
{
  using value_type          = T;
  using supported_options_t = supported_options<T, Value>;

  template <typename SegmentIndexT>
  [[nodiscard]] _CCCL_HOST_DEVICE constexpr T get_param(SegmentIndexT) const noexcept
  {
    return Value;
  }
};

// =====================================================================
// Discrete dispatch
// =====================================================================

//! @brief Translates a runtime parameter value into a compile-time constant by matching
//!        against a list of supported options.
//!
//! @param[in] val Runtime value to match.
//! @param[in] __supported_options Supported values for the parameter.
//! @param[in] f Functor invoked with the matched compile-time constant.
//! @return `true` if the value matches one of the supported options.
template <typename T, T... Opts, typename Functor>
[[nodiscard]] _CCCL_HOST_DEVICE bool
dispatch_impl(T val, [[maybe_unused]] supported_options<T, Opts...> __supported_options, Functor&& f)
{
  const bool match_found = ((val == Opts ? (f(::cuda::std::integral_constant<T, Opts>{}), true) : false) || ...);
  _CCCL_ASSERT(match_found, "The given runtime parameter value is not in the supported list");
  return match_found;
}

//! @brief Dispatcher that resolves a discrete parameter to a compile-time constant
//!        and invokes a functor with the matched option.
//!
//! @param[in] param Discrete parameter to resolve.
//! @param[in] segment_id Segment index to read from `param`.
//! @param[in] f Functor invoked with the matched compile-time constant.
//! @return `true` if the parameter value matches one of its supported options.
template <typename ParamT, typename SegmentIndexT, typename Functor>
[[nodiscard]] _CCCL_HOST_DEVICE bool dispatch_discrete(ParamT param, SegmentIndexT segment_id, Functor&& f)
{
  using supported_list = typename ParamT::supported_options_t;
  auto param_value     = param.get_param(segment_id);
  return CUB_NS_QUALIFIER::detail::params::dispatch_impl(
    param_value, supported_list{}, ::cuda::std::forward<Functor>(f));
}
} // namespace detail::params

CUB_NAMESPACE_END
