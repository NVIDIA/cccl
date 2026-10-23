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

#include <cuda/std/type_traits>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_scan
{
template <typename ValueT, typename FlagT = bool>
// Producers supply both fields while preserving triviality.
// NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
struct augmented_value_t
{
  ValueT value;
  FlagT flag;
};

template <typename ValueT, typename FlagT = bool>
_CCCL_DEDUCTION_GUIDE_ATTRIBUTES augmented_value_t(ValueT, FlagT) -> augmented_value_t<ValueT, FlagT>;

template <typename ComputeT, int MaxSegmentsPerWorker>
using agent_segmented_scan_compute_t =
  ::cuda::std::conditional_t<MaxSegmentsPerWorker == 1, ComputeT, augmented_value_t<ComputeT>>;

template <typename ToT>
struct initial_value_converter
{
  template <typename FromT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static constexpr ToT cast(FromT v) noexcept
  {
    return static_cast<ToT>(v);
  }
};

template <typename ToValueT, typename ToFlagT>
struct initial_value_converter<augmented_value_t<ToValueT, ToFlagT>>
{
  template <typename FromValueT, typename FromFlagT>
  _CCCL_DEVICE _CCCL_FORCEINLINE static constexpr augmented_value_t<ToValueT, ToFlagT>
  cast(augmented_value_t<FromValueT, FromFlagT> fv) noexcept
  {
    return {static_cast<ToValueT>(fv.value), static_cast<ToFlagT>(fv.flag)};
  }
};

template <typename ToT, typename FromT>
_CCCL_DEVICE _CCCL_FORCEINLINE constexpr ToT convert_initial_value(FromT v) noexcept
{
  return initial_value_converter<ToT>::cast(v);
}

template <typename BinaryOpT, typename ValueT, typename FlagT = bool>
struct schwarz_scan_op
{
  using fv_t = augmented_value_t<ValueT, FlagT>;
  mutable BinaryOpT scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE fv_t operator()(fv_t o1, fv_t o2) const
  {
    if (o2.flag)
    {
      return o2;
    }
    const ValueT res_value = scan_op(o1.value, o2.value);

    return fv_t{res_value, o1.flag};
  }
};

template <typename V, typename F = bool>
struct packer
{
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return augmented_value_t<V, F>{v, f};
  }
};

template <typename ScanOp, typename V, typename F = bool>
struct packer_iv
{
  mutable ScanOp op;
  V init_v;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    V res = v;
    if (f)
    {
      res = op(init_v, v);
    }
    return augmented_value_t<V, F>{res, f};
  }
};

template <typename V, typename F = bool>
struct projector_iv
{
  V init_v;

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto operator()(V v, F f) const
  {
    return (f) ? init_v : v;
  }
};

template <typename PrefixT, typename BinaryOpT>
struct worker_prefix_callback_t
{
  PrefixT& m_exclusive_prefix;
  BinaryOpT m_scan_op;

  _CCCL_DEVICE _CCCL_FORCEINLINE worker_prefix_callback_t(PrefixT& prefix, BinaryOpT& op)
      : m_exclusive_prefix(prefix)
      , m_scan_op(op)
  {}

  _CCCL_DEVICE _CCCL_FORCEINLINE PrefixT operator()(PrefixT block_aggregate)
  {
    const PrefixT previous_prefix = m_exclusive_prefix;
    m_exclusive_prefix            = m_scan_op(m_exclusive_prefix, block_aggregate);
    return previous_prefix;
  }
};
} // namespace detail::segmented_scan

CUB_NAMESPACE_END
