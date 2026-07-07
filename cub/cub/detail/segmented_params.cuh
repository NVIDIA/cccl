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
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_integer.h> // __cccl_is_integer_v
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/cmp.h> // cmp_less, cmp_greater_equal, cmp_less_equal
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN

namespace detail::params
{
// =====================================================================
// Deferred handle requirements
// =====================================================================

// A `deferred` is read by dereferencing its handle (`*handle`, see the get_param overload below), so the handle must
// be indirectly readable (a pointer or other dereferenceable handle). Ranges/containers (span, array, ...) are
// rejected -- their bounds are ambiguous (values vs. size); use `deferred_sequence` for per-segment values.
template <class _Handle>
inline constexpr bool __is_valid_deferred_handle_v = ::cuda::std::indirectly_readable<_Handle>;

// A `deferred_sequence` is indexed per segment (`handle[index]`), so its handle must be a random-access iterator.
template <class _Handle>
inline constexpr bool __is_valid_deferred_sequence_handle_v = ::cuda::std::random_access_iterator<_Handle>;

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

// Reads the size of segment `__index` as a non-negative count. A signed argument type with a negative static lower
// bound (e.g. an un-annotated `int16_t`, or an explicit `bounds` with a negative lower end) can legitimately produce a
// negative value; it is clamped up to 0 so a negative runtime size becomes an empty segment. A statically non-negative
// lower bound is trusted and left unclamped. For the deferred forms (whose values the host cannot see), `get_param`
// already checks the value against the argument's effective bounds in debug builds, so a value below a non-negative
// lower bound is caught there.
template <class _Arg, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto get_segment_size(const _Arg& __arg, _SegmentIndexT __index) noexcept
{
  auto __size = get_param(__arg, __index);
  if constexpr (::cuda::std::cmp_less(::cuda::args::__traits<_Arg>::lowest, 0))
  {
    return (::cuda::std::max) (__size, static_cast<decltype(__size)>(0));
  }
  else
  {
    return __size;
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
