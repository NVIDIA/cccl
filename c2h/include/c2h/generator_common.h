// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>
#include <cuda/std/type_traits>
#include <cuda/stream>

#include <cstddef>
#include <stdexcept>

#include <c2h/custom_type.h>

namespace c2h
{
namespace detail
{
template <typename T>
[[nodiscard]] std::size_t checked_uniform_offsets_size(T total_elements)
{
  static_assert(::cuda::std::is_integral_v<T>, "Uniform offset types must be integral");

  if constexpr (::cuda::std::is_signed_v<T>)
  {
    if (total_elements < T{0})
    {
      throw std::invalid_argument{"total_elements must be non-negative"};
    }
  }

  // gen_uniform_offsets uses total_elements + 1 as a sentinel.
  if (total_elements == (::cuda::std::numeric_limits<T>::max)())
  {
    throw std::invalid_argument{"total_elements is too large to generate uniform offsets"};
  }

  constexpr auto max_size            = (::cuda::std::numeric_limits<std::size_t>::max)();
  const auto total_elements_unsigned = static_cast<::cuda::std::uintmax_t>(total_elements);
  if (total_elements_unsigned > static_cast<::cuda::std::uintmax_t>(max_size - 2))
  {
    throw std::invalid_argument{"total_elements is too large to generate uniform offsets"};
  }

  return static_cast<std::size_t>(total_elements) + 2;
}

template <class T>
class value_wrapper_t
{
  T m_val{};

public:
  using value_type = T;

  explicit value_wrapper_t(T val)
      : m_val(val)
  {}
  explicit value_wrapper_t(int val)
      : m_val(static_cast<T>(val))
  {}
  T get() const
  {
    return m_val;
  }
};
} // namespace detail

struct seed_t : detail::value_wrapper_t<unsigned long long int>
{
  using value_wrapper_t::value_wrapper_t;
};

struct modulo_t : detail::value_wrapper_t<std::size_t>
{
  using value_wrapper_t::value_wrapper_t;
};

namespace detail
{
void gen_custom_type_state(
  seed_t seed,
  char* data,
  custom_type_state_t min,
  custom_type_state_t max,
  std::size_t elements,
  std::size_t element_size);

void gen_custom_type_state(
  ::cuda::stream_ref stream,
  seed_t seed,
  char* data,
  custom_type_state_t min,
  custom_type_state_t max,
  std::size_t elements,
  std::size_t element_size);

template <typename OffsetT, typename KeyT>
void init_key_segments(::cuda::std::span<const OffsetT> segment_offsets, KeyT* d_out, std::size_t element_size);

template <typename T>
void gen_values_between(seed_t seed, ::cuda::std::span<T> data, T min, T max);

template <typename T>
void gen_values_between(::cuda::stream_ref stream, seed_t seed, ::cuda::std::span<T> data, T min, T max);

template <typename T>
void gen_values_cyclic(modulo_t mod, ::cuda::std::span<T> data);

template <typename T>
std::size_t gen_uniform_offsets(
  seed_t seed, ::cuda::std::span<T> segment_offsets, T total_elements, T min_segment_size, T max_segment_size);

template <typename T>
std::size_t gen_uniform_offsets(
  ::cuda::stream_ref stream,
  seed_t seed,
  ::cuda::std::span<T> segment_offsets,
  T total_elements,
  T min_segment_size,
  T max_segment_size);
} // namespace detail
} // namespace c2h
