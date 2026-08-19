// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cuda/std/span>

#include <cstddef>

#include <c2h/custom_type.h>

namespace c2h
{
namespace detail
{
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

template <typename OffsetT, typename KeyT>
void init_key_segments(::cuda::std::span<const OffsetT> segment_offsets, KeyT* d_out, std::size_t element_size);

template <typename T>
void gen_values_between(seed_t seed, ::cuda::std::span<T> data, T min, T max);

template <typename T>
void gen_values_cyclic(modulo_t mod, ::cuda::std::span<T> data);

template <typename T>
std::size_t gen_uniform_offsets(
  seed_t seed, ::cuda::std::span<T> segment_offsets, T total_elements, T min_segment_size, T max_segment_size);
} // namespace detail
} // namespace c2h
