// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/__cccl/visibility.h>
#include <cuda/std/tuple>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename ValueT, typename FlagT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr FlagT get_flag(::cuda::std::tuple<ValueT, FlagT> fv) noexcept
{
  return ::cuda::std::get<1>(fv);
}

template <typename ValueT, typename FlagT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr ValueT get_value(::cuda::std::tuple<ValueT, FlagT> fv) noexcept
{
  return ::cuda::std::get<0>(fv);
}

template <typename ValueT, typename FlagT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr ::cuda::std::tuple<ValueT, FlagT> make_value_flag(ValueT v, FlagT f) noexcept
{
  return {v, f};
}

template <typename ValueT, typename FlagT, typename BinaryOpT>
struct schwarz_scan_op
{
  using fv_t = ::cuda::std::tuple<ValueT, FlagT>;
  BinaryOpT& scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE fv_t operator()(fv_t o1, fv_t o2)
  {
    if (get_flag(o2))
    {
      return o2;
    }
    const auto o2_value    = get_value(o2);
    const auto o1_value    = get_value(o1);
    const ValueT res_value = scan_op(o1_value, o2_value);

    return make_value_flag(res_value, get_flag(o1));
  }
};

template <typename ValueT, typename FlagT, typename BinaryOpT>
struct iv_schwarz_scan_op
{
  using fv_t = ::cuda::std::tuple<ValueT, FlagT>;
  BinaryOpT& scan_op;
  ValueT init_v;

  _CCCL_DEVICE _CCCL_FORCEINLINE fv_t operator()(fv_t o1, fv_t o2)
  {
    const auto o2_value = get_value(o2);
    if (get_flag(o2))
    {
      return make_value_flag(scan_op(init_v, o2_value), true);
    }
    const auto o1_value    = get_value(o1);
    const ValueT res_value = scan_op(o1_value, o2_value);

    return make_value_flag(res_value, get_flag(o1));
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
