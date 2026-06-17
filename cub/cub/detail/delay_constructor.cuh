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

#include <cuda/std/__host_stdlib/ostream>

CUB_NAMESPACE_BEGIN

//! The delay algorithm used by decoupled lookback
enum class LookbackDelayAlgorithm
{
  no_delay,
  fixed_delay,
  exponential_backoff,
  exponential_backoff_jitter,
  exponential_backoff_jitter_window,
  exponential_backon_jitter_window,
  exponential_backon_jitter,
  exponential_backon,
  __reduce_by_key //!< Internal
};

#if _CCCL_HOSTED()
inline ::std::ostream& operator<<(::std::ostream& os, LookbackDelayAlgorithm kind)
{
  switch (kind)
  {
    case LookbackDelayAlgorithm::no_delay:
      return os << "LookbackDelayAlgorithm::no_delay";
    case LookbackDelayAlgorithm::fixed_delay:
      return os << "LookbackDelayAlgorithm::fixed_delay";
    case LookbackDelayAlgorithm::exponential_backoff:
      return os << "LookbackDelayAlgorithm::exponential_backoff";
    case LookbackDelayAlgorithm::exponential_backoff_jitter:
      return os << "LookbackDelayAlgorithm::exponential_backoff_jitter";
    case LookbackDelayAlgorithm::exponential_backoff_jitter_window:
      return os << "LookbackDelayAlgorithm::exponential_backoff_jitter_window";
    case LookbackDelayAlgorithm::exponential_backon_jitter_window:
      return os << "LookbackDelayAlgorithm::exponential_backon_jitter_window";
    case LookbackDelayAlgorithm::exponential_backon_jitter:
      return os << "LookbackDelayAlgorithm::exponential_backon_jitter";
    case LookbackDelayAlgorithm::exponential_backon:
      return os << "LookbackDelayAlgorithm::exponential_backon";
    case LookbackDelayAlgorithm::__reduce_by_key:
      return os << "LookbackDelayAlgorithm::__reduce_by_key";
    default:
      return os << "<unknown LookbackDelayAlgorithm: " << static_cast<int>(kind) << ">";
  }
}
#endif // _CCCL_HOSTED()

//! The policy configuring the delay algorithm used by decoupled lookback
struct LookbackDelayPolicy
{
  LookbackDelayAlgorithm kind; //!< The algorithm used for delaying during decoupled lookback
  unsigned int delay; //!< The delay in nanoseconds
  unsigned int l2_write_latency; //!< The write latency of the L2 cache in nanoseconds

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator==(const LookbackDelayPolicy& lhs, const LookbackDelayPolicy& rhs) noexcept
  {
    return lhs.kind == rhs.kind && lhs.delay == rhs.delay && lhs.l2_write_latency == rhs.l2_write_latency;
  }

  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr friend bool
  operator!=(const LookbackDelayPolicy& lhs, const LookbackDelayPolicy& rhs) noexcept
  {
    return !(lhs == rhs);
  }

#if _CCCL_HOSTED()
  friend ::std::ostream& operator<<(::std::ostream& os, const LookbackDelayPolicy& p)
  {
    return os << "LookbackDelayPolicy { .kind = " << p.kind << ", .delay = " << p.delay
              << ", .l2_write_latency = " << p.l2_write_latency << " }";
  }
#endif // _CCCL_HOSTED()
};

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
