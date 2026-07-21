// SPDX-FileCopyrightText: Copyright (c) 2024-2026, NVIDIA CORPORATION. All rights reserved.
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

#include <cub/agent/single_pass_scan_operators.cuh>
#include <cub/device/dispatch/tuning/common.cuh>

#include <cuda/std/__concepts/same_as.h>

CUB_NAMESPACE_BEGIN

namespace detail
{
template <typename DelayConstructor>
inline constexpr auto lookback_delay_policy_from_type = 0;

template <unsigned int L2WriteLatency>
inline constexpr auto lookback_delay_policy_from_type<no_delay_constructor_t<L2WriteLatency>> =
  LookbackDelayPolicy{LookbackDelayAlgorithm::no_delay, 0, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto lookback_delay_policy_from_type<fixed_delay_constructor_t<Delay, L2WriteLatency>> =
  LookbackDelayPolicy{LookbackDelayAlgorithm::fixed_delay, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto lookback_delay_policy_from_type<exponential_backoff_constructor_t<Delay, L2WriteLatency>> =
  LookbackDelayPolicy{LookbackDelayAlgorithm::exponential_backoff, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto lookback_delay_policy_from_type<exponential_backoff_jitter_constructor_t<Delay, L2WriteLatency>> =
  LookbackDelayPolicy{LookbackDelayAlgorithm::exponential_backoff_jitter, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  lookback_delay_policy_from_type<exponential_backoff_jitter_window_constructor_t<Delay, L2WriteLatency>> =
    LookbackDelayPolicy{LookbackDelayAlgorithm::exponential_backoff_jitter_window, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  lookback_delay_policy_from_type<exponential_backon_jitter_window_constructor_t<Delay, L2WriteLatency>> =
    LookbackDelayPolicy{LookbackDelayAlgorithm::exponential_backon_jitter_window, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto lookback_delay_policy_from_type<exponential_backon_jitter_constructor_t<Delay, L2WriteLatency>> =
  LookbackDelayPolicy{LookbackDelayAlgorithm::exponential_backon_jitter, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto lookback_delay_policy_from_type<exponential_backon_constructor_t<Delay, L2WriteLatency>> =
  LookbackDelayPolicy{LookbackDelayAlgorithm::exponential_backon, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency, unsigned int GridThreshold>
inline constexpr auto
  lookback_delay_policy_from_type<reduce_by_key_delay_constructor_t<Delay, L2WriteLatency, GridThreshold>> =
    LookbackDelayPolicy{LookbackDelayAlgorithm::__reduce_by_key, Delay, L2WriteLatency};

template <LookbackDelayAlgorithm Kind, unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for;

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::no_delay, Delay, L2WriteLatency>
{
  using type = no_delay_constructor_t<L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::fixed_delay, Delay, L2WriteLatency>
{
  using type = fixed_delay_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::exponential_backoff, Delay, L2WriteLatency>
{
  using type = exponential_backoff_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::exponential_backoff_jitter, Delay, L2WriteLatency>
{
  using type = exponential_backoff_jitter_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::exponential_backoff_jitter_window, Delay, L2WriteLatency>
{
  using type = exponential_backoff_jitter_window_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::exponential_backon_jitter_window, Delay, L2WriteLatency>
{
  using type = exponential_backon_jitter_window_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::exponential_backon_jitter, Delay, L2WriteLatency>
{
  using type = exponential_backon_jitter_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::exponential_backon, Delay, L2WriteLatency>
{
  using type = exponential_backon_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<LookbackDelayAlgorithm::__reduce_by_key, Delay, L2WriteLatency>
{
  using type = reduce_by_key_delay_constructor_t<Delay, L2WriteLatency>;
};

template <LookbackDelayAlgorithm Kind, unsigned int Delay, unsigned int L2WriteLatency>
using delay_constructor_t = typename delay_constructor_for<Kind, Delay, L2WriteLatency>::type;

_CCCL_HOST_DEVICE_API constexpr auto default_delay_constructor_policy(bool is_primitive_or_trivially_copyable)
{
  if (is_primitive_or_trivially_copyable)
  {
    return LookbackDelayPolicy{LookbackDelayAlgorithm::fixed_delay, 350, 450};
  }
  return LookbackDelayPolicy{LookbackDelayAlgorithm::no_delay, 0, 450};
}

_CCCL_HOST_DEVICE_API constexpr auto default_reduce_by_key_delay_constructor_policy(
  int key_size,
  int value_size,
  bool key_is_primitive_or_trivially_copyable,
  bool value_is_primitive_or_trivially_copyable)
{
  if (value_is_primitive_or_trivially_copyable && (value_size + key_size < 16))
  {
    return LookbackDelayPolicy{LookbackDelayAlgorithm::__reduce_by_key, 350, 450};
  }
  return default_delay_constructor_policy(key_is_primitive_or_trivially_copyable);
}
} // namespace detail

CUB_NAMESPACE_END
