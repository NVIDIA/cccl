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

#include <cuda/__argument_>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/cstddef>

CUB_NAMESPACE_BEGIN

namespace detail::params
{
// =====================================================================
// get_param — unified segment parameter access
// =====================================================================

//! @brief Returns the value of an argument for a given segment index.
//!
//! @param[in] __arg Argument or argument wrapper to read.
//! @param[in] __index Segment index to read for sequence arguments.
//! @return The single argument value, or the sequence element at the given index.
_CCCL_TEMPLATE(class _Tp, class _SegmentIndexT)
_CCCL_REQUIRES((!::cuda::__argument::__is_wrapper_v<::cuda::std::remove_cvref_t<_Tp>>) )
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto get_param(_Tp&& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  if constexpr (::cuda::__argument::__traits<::cuda::std::remove_cvref_t<_Tp>>::is_single_value)
  {
    return __arg;
  }
  else
  {
    return __arg[__index];
  }
}

template <auto _Value, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::__argument::__constant<_Value>& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  return ::cuda::__argument::__unwrap(__arg);
}

template <auto _Value, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::__argument::__constant_sequence<_Value>& __arg, _SegmentIndexT __index) noexcept
{
  return ::cuda::__argument::__unwrap(__arg)[__index];
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto get_param(
  const ::cuda::__argument::__immediate<_Arg, _StaticBounds>& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  return ::cuda::__argument::__unwrap(__arg);
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::__argument::__immediate_sequence<_Arg, _StaticBounds>& __arg, _SegmentIndexT __index) noexcept
{
  return ::cuda::__argument::__unwrap(__arg)[__index];
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto get_param(
  const ::cuda::__argument::__deferred<_Arg, _StaticBounds>& __arg, [[maybe_unused]] _SegmentIndexT __index) noexcept
{
  return ::cuda::__argument::__unwrap(__arg);
}

template <class _Arg, class _StaticBounds, class _SegmentIndexT>
[[nodiscard]] _CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::__argument::__deferred_sequence<_Arg, _StaticBounds>& __arg, _SegmentIndexT __index) noexcept
{
  return ::cuda::__argument::__unwrap(__arg)[__index];
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

//! @brief Uniform discrete parameter — a single runtime value with a known set of supported options.
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

//! @brief Per-segment discrete parameter — per-segment values with a known set of supported options.
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

//! @brief Dispatcher that resolves a per-segment discrete parameter to a compile-time constant
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
  return ::cub::detail::params::dispatch_impl(param_value, supported_list{}, ::cuda::std::forward<Functor>(f));
}
} // namespace detail::params

CUB_NAMESPACE_END
