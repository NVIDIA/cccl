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

#include <cuda/argument>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/forward.h>

CUB_NAMESPACE_BEGIN

namespace detail::params
{
// =====================================================================
// get_param — unified segment parameter access
// =====================================================================

//! @brief Returns the value of an argument for a given segment index.
//! For single-value arguments, the index is ignored and the value is returned directly.
//! For per-segment arguments, returns the element at the given index.
_CCCL_TEMPLATE(class _Tp)
_CCCL_REQUIRES((!::cuda::argument::__is_wrapper_v<::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<_Tp>>>) )
_CCCL_HOST_DEVICE constexpr auto get_param(_Tp&& __arg, [[maybe_unused]] size_t __index) noexcept
{
  if constexpr (::cuda::argument::__is_single_value_v<::cuda::std::remove_cv_t<::cuda::std::remove_reference_t<_Tp>>>)
  {
    return __arg;
  }
  else
  {
    return __arg[__index];
  }
}

template <auto _Value>
_CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::argument::__constant<_Value>&, [[maybe_unused]] size_t __index) noexcept
{
  return _Value;
}

template <class _Arg, class _StaticBounds>
_CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::argument::__immediate<_Arg, _StaticBounds>& __arg, [[maybe_unused]] size_t __index) noexcept
{
  if constexpr (::cuda::argument::__is_single_value_v<_Arg>)
  {
    return __arg.arg;
  }
  else
  {
    return __arg.arg[__index];
  }
}

template <class _Arg, class _StaticBounds>
_CCCL_HOST_DEVICE constexpr auto get_param(const ::cuda::argument::__deferred_value<_Arg, _StaticBounds>& __arg,
                                           [[maybe_unused]] size_t __index) noexcept
{
  return __arg.arg[__index];
}

template <class _Arg, class _StaticBounds>
_CCCL_HOST_DEVICE constexpr auto
get_param(const ::cuda::argument::__deferred_sequence<_Arg, _StaticBounds>& __arg, size_t __index) noexcept
{
  return __arg.arg[__index];
}

// =====================================================================
// Discrete parameter support
// =====================================================================

//! @brief Specifies a list of supported options for a parameter.
template <typename T, T... Options>
struct supported_options
{
  static constexpr size_t count = sizeof...(Options);
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
template <typename T, T... Opts, typename Functor>
_CCCL_HOST_DEVICE bool dispatch_impl(T val, supported_options<T, Opts...>, Functor&& f)
{
  const bool match_found = ((val == Opts ? (f(::cuda::std::integral_constant<T, Opts>{}), true) : false) || ...);
  _CCCL_ASSERT(match_found, "The given runtime parameter value is not in the supported list");
  return match_found;
}

//! @brief Dispatcher that resolves a per-segment discrete parameter to a compile-time constant
//!        and invokes a functor with the matched option.
template <typename ParamT, typename SegmentIndexT, typename Functor>
_CCCL_HOST_DEVICE bool dispatch_discrete(ParamT param, SegmentIndexT segment_id, Functor&& f)
{
  using supported_list = typename ParamT::supported_options_t;
  auto param_value     = param.get_param(segment_id);
  return dispatch_impl(param_value, supported_list{}, ::cuda::std::forward<Functor>(f));
}
} // namespace detail::params

CUB_NAMESPACE_END
