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
#include <cuda/std/__type_traits/is_integer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_unsigned.h>
#include <cuda/std/__utility/cmp.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename NumItemsT>
using num_items_offset_t = choose_offset_t<typename ::cuda::args::__traits<NumItemsT>::element_type>;

template <typename NumItemsT>
_CCCL_HOST_DEVICE_API constexpr void assert_num_items_nonnegative([[maybe_unused]] NumItemsT num_items) noexcept
{
  _CCCL_ASSERT(!(::cuda::std::is_unsigned_v<NumItemsT> ? false : (num_items < 0)), "num_items must be nonnegative");
}

#if !_CCCL_COMPILER(NVRTC)
template <typename NumItemsT>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto num_items_upper_bound(NumItemsT num_items) noexcept
{
  using offset_t         = num_items_offset_t<NumItemsT>;
  const auto upper_bound = ::cuda::args::__highest_(num_items);
  assert_num_items_nonnegative(upper_bound);
  return static_cast<offset_t>(upper_bound);
}

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
    using offset_t   = choose_offset_t<typename args_traits_t::value_type>;
    const auto value = ::cuda::args::__unwrap(num_items);
    assert_num_items_nonnegative(value);
    return static_cast<offset_t>(value);
  }
}

// Forms a kernel parameter from a single-value argument without reading a deferred source.
// Immediate values are converted to TargetT. Deferred arguments retain their bounds for device assertions;
// otherwise, they are unwrapped to their source, erasing bounds from the kernel type and payload.
template <typename TargetT, typename ParameterT>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE constexpr auto parameter_from_host(ParameterT parameter) noexcept
{
  using args_traits_t = ::cuda::args::__traits<ParameterT>;
  static_assert(args_traits_t::is_single_value, "parameter must contain a single value");

  if constexpr (args_traits_t::is_deferred)
  {
#  if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
    return parameter;
#  else
    return ::cuda::args::__unwrap(parameter);
#  endif
  }
  else
  {
    return static_cast<TargetT>(::cuda::args::__unwrap(parameter));
  }
}

#  if defined(CCCL_ENABLE_DEVICE_ASSERTIONS)
// Work around for NVCC emitting invalid lambda-dependent type aliases
template <typename TargetT, typename SourceT, typename StaticBoundsT>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE constexpr auto
parameter_from_host(::cuda::args::deferred<SourceT, StaticBoundsT> parameter) noexcept
  -> ::cuda::args::deferred<SourceT, StaticBoundsT>
{
  return parameter;
}
#  endif

template <typename TargetT, typename ParameterT>
using parameter_from_host_t = decltype(parameter_from_host<TargetT>(::cuda::std::declval<ParameterT>()));

template <typename NumItemsT>
[[nodiscard]] CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE constexpr auto
make_num_items_kernel_arg(NumItemsT num_items) noexcept
{
  using args_traits_t = ::cuda::args::__traits<NumItemsT>;
  using element_t     = typename args_traits_t::element_type;

  if constexpr (args_traits_t::is_deferred)
  {
    static_assert(args_traits_t::is_single_value, "num_items must be a single value wrapped in cuda::args::deferred");
    static_assert(::cuda::std::__cccl_is_integer_v<element_t>, "the num_items element type must be an integer");
    static_assert(
      sizeof(element_t) == sizeof(::cuda::std::int32_t) || sizeof(element_t) == sizeof(::cuda::std::int64_t));
  }
  else
  {
    assert_num_items_nonnegative(::cuda::args::__unwrap(num_items));
  }

  return parameter_from_host<num_items_offset_t<NumItemsT>>(num_items);
}
#endif // !_CCCL_COMPILER(NVRTC)

// Forms a value from a kernel parameter, reading element zero when the parameter is a deferred source.
template <typename TargetT, typename ParameterT>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TargetT parameter_from_device(ParameterT parameter) noexcept
{
  if constexpr (::cuda::std::is_same_v<ParameterT, TargetT>)
  {
    assert_num_items_nonnegative(parameter);
    return parameter;
  }
  else
  {
    using element_t       = typename ::cuda::args::__traits<ParameterT>::element_type;
    const element_t value = parameter[0];
    assert_num_items_nonnegative(value);
    return static_cast<TargetT>(value);
  }
}

template <typename TargetT, typename SourceT, typename StaticBoundsT>
[[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE TargetT
parameter_from_device(::cuda::args::deferred<SourceT, StaticBoundsT> parameter) noexcept
{
  using element_t       = typename ::cuda::args::__traits<decltype(parameter)>::element_type;
  const element_t value = ::cuda::args::__unwrap(parameter)[0];
  assert_num_items_nonnegative(value);
  _CCCL_ASSERT(::cuda::std::cmp_greater_equal(value, ::cuda::args::__lowest_(parameter))
                 && ::cuda::std::cmp_less_equal(value, ::cuda::args::__highest_(parameter)),
               "deferred num_items is outside its declared bounds");
  return static_cast<TargetT>(value);
}
} // namespace detail

CUB_NAMESPACE_END
