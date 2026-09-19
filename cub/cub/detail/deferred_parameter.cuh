// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/detail/choose_offset.cuh>

#include <cuda/__argument/argument.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__utility/declval.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
#if !_CCCL_COMPILER(NVRTC)
// Preserve deferred problem sizes for dispatch and canonicalize immediate values to CUB's offset type.
template <typename NumItemsT>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto make_num_items_dispatch_arg(NumItemsT num_items) noexcept
{
  using args_traits_t = ::cuda::args::__traits<NumItemsT>;

  if constexpr (args_traits_t::is_deferred)
  {
    return num_items;
  }
  else
  {
    using offset_t = choose_offset_t<typename args_traits_t::element_type>;
    return static_cast<offset_t>(::cuda::args::__unwrap(num_items));
  }
}

// Forms a kernel parameter from a single-value argument without reading a deferred source.
// Immediate values are converted to TargetT. Deferred arguments are unwrapped to their source, erasing bounds from
// the kernel type and payload.
template <typename TargetT, typename ParameterT>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE constexpr auto parameter_from_host(ParameterT parameter) noexcept
{
  using args_traits_t = ::cuda::args::__traits<ParameterT>;
  static_assert(args_traits_t::is_single_value, "parameter must contain a single value");

  if constexpr (args_traits_t::is_deferred)
  {
    return ::cuda::args::__unwrap(parameter);
  }
  else
  {
    return static_cast<TargetT>(::cuda::args::__unwrap(parameter));
  }
}

template <typename TargetT, typename ParameterT>
using parameter_from_host_t = decltype(parameter_from_host<TargetT>(::cuda::std::declval<ParameterT>()));
#endif // !_CCCL_COMPILER(NVRTC)

// Forms a value from a kernel parameter, reading element zero when the parameter is a deferred source.
template <typename TargetT, typename ParameterT>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TargetT parameter_from_device(ParameterT parameter) noexcept
{
  if constexpr (::cuda::std::is_same_v<ParameterT, TargetT>)
  {
    return parameter;
  }
  else
  {
    return static_cast<TargetT>(parameter[0]);
  }
}
} // namespace detail

CUB_NAMESPACE_END
