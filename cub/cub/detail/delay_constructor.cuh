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

#if !_CCCL_COMPILER(NVRTC)
#  include <ostream>
#endif

CUB_NAMESPACE_BEGIN

namespace detail
{
enum class delay_constructor_kind
{
  no_delay,
  fixed_delay,
  exponential_backoff,
  exponential_backoff_jitter,
  exponential_backoff_jitter_window,
  exponential_backon_jitter_window,
  exponential_backon_jitter,
  exponential_backon,
  reduce_by_key
};

#if !_CCCL_COMPILER(NVRTC)
inline ::std::ostream& operator<<(::std::ostream& os, delay_constructor_kind kind)
{
  switch (kind)
  {
    case delay_constructor_kind::no_delay:
      return os << "delay_constructor_kind::no_delay";
    case delay_constructor_kind::fixed_delay:
      return os << "delay_constructor_kind::fixed_delay";
    case delay_constructor_kind::exponential_backoff:
      return os << "delay_constructor_kind::exponential_backoff";
    case delay_constructor_kind::exponential_backoff_jitter:
      return os << "delay_constructor_kind::exponential_backoff_jitter";
    case delay_constructor_kind::exponential_backoff_jitter_window:
      return os << "delay_constructor_kind::exponential_backoff_jitter_window";
    case delay_constructor_kind::exponential_backon_jitter_window:
      return os << "delay_constructor_kind::exponential_backon_jitter_window";
    case delay_constructor_kind::exponential_backon_jitter:
      return os << "delay_constructor_kind::exponential_backon_jitter";
    case delay_constructor_kind::exponential_backon:
      return os << "delay_constructor_kind::exponential_backon";
    case delay_constructor_kind::reduce_by_key:
      return os << "delay_constructor_kind::reduce_by_key";
    default:
      return os << "<unknown delay_constructor_kind: " << static_cast<int>(kind) << ">";
  }
}
#endif // !_CCCL_COMPILER(NVRTC)

struct delay_constructor_policy
{
  delay_constructor_kind kind;
  unsigned int delay;
  unsigned int l2_write_latency;

  _CCCL_API constexpr friend bool operator==(const delay_constructor_policy& lhs, const delay_constructor_policy& rhs)
  {
    return lhs.kind == rhs.kind && lhs.delay == rhs.delay && lhs.l2_write_latency == rhs.l2_write_latency;
  }

  _CCCL_API constexpr friend bool operator!=(const delay_constructor_policy& lhs, const delay_constructor_policy& rhs)
  {
    return !(lhs == rhs);
  }

#if !_CCCL_COMPILER(NVRTC)
  friend ::std::ostream& operator<<(::std::ostream& os, const delay_constructor_policy& p)
  {
    return os << "delay_constructor_policy { .kind = " << p.kind << ", .delay = " << p.delay
              << ", .l2_write_latency = " << p.l2_write_latency << " }";
  }
#endif // !_CCCL_COMPILER(NVRTC)
};

template <typename DelayConstructor>
inline constexpr auto delay_constructor_policy_from_type = 0;

template <unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<no_delay_constructor_t<L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::no_delay, 0, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<fixed_delay_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::fixed_delay, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<exponential_backoff_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::exponential_backoff, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  delay_constructor_policy_from_type<exponential_backoff_jitter_constructor_t<Delay, L2WriteLatency>> =
    delay_constructor_policy{delay_constructor_kind::exponential_backoff_jitter, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  delay_constructor_policy_from_type<exponential_backoff_jitter_window_constructor_t<Delay, L2WriteLatency>> =
    delay_constructor_policy{delay_constructor_kind::exponential_backoff_jitter_window, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto
  delay_constructor_policy_from_type<exponential_backon_jitter_window_constructor_t<Delay, L2WriteLatency>> =
    delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter_window, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<exponential_backon_jitter_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::exponential_backon_jitter, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency>
inline constexpr auto delay_constructor_policy_from_type<exponential_backon_constructor_t<Delay, L2WriteLatency>> =
  delay_constructor_policy{delay_constructor_kind::exponential_backon, Delay, L2WriteLatency};

template <unsigned int Delay, unsigned int L2WriteLatency, unsigned int GridThreshold>
inline constexpr auto
  delay_constructor_policy_from_type<reduce_by_key_delay_constructor_t<Delay, L2WriteLatency, GridThreshold>> =
    delay_constructor_policy{delay_constructor_kind::reduce_by_key, Delay, L2WriteLatency};

template <delay_constructor_kind Kind, unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for;

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::no_delay, Delay, L2WriteLatency>
{
  using type = no_delay_constructor_t<L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::fixed_delay, Delay, L2WriteLatency>
{
  using type = fixed_delay_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::exponential_backoff, Delay, L2WriteLatency>
{
  using type = exponential_backoff_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::exponential_backoff_jitter, Delay, L2WriteLatency>
{
  using type = exponential_backoff_jitter_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::exponential_backoff_jitter_window, Delay, L2WriteLatency>
{
  using type = exponential_backoff_jitter_window_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::exponential_backon_jitter_window, Delay, L2WriteLatency>
{
  using type = exponential_backon_jitter_window_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::exponential_backon_jitter, Delay, L2WriteLatency>
{
  using type = exponential_backon_jitter_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::exponential_backon, Delay, L2WriteLatency>
{
  using type = exponential_backon_constructor_t<Delay, L2WriteLatency>;
};

template <unsigned int Delay, unsigned int L2WriteLatency>
struct delay_constructor_for<delay_constructor_kind::reduce_by_key, Delay, L2WriteLatency>
{
  using type = reduce_by_key_delay_constructor_t<Delay, L2WriteLatency>;
};

template <delay_constructor_kind Kind, unsigned int Delay, unsigned int L2WriteLatency>
using delay_constructor_t = typename delay_constructor_for<Kind, Delay, L2WriteLatency>::type;

_CCCL_API constexpr auto
default_reduce_by_key_delay_constructor_policy(int key_size, int value_size, bool value_is_primitive)
{
  if (value_is_primitive && (value_size + key_size < 16))
  {
    return delay_constructor_policy{delay_constructor_kind::reduce_by_key, 350, 450};
  }
  return delay_constructor_policy{delay_constructor_kind::no_delay, 0, 450};
}
} // namespace detail

CUB_NAMESPACE_END
