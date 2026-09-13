// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/std/complex>
#include <cuda/stream>
#include <cuda/type_traits>

#include <cstddef>
#include <memory>

#include <c2h/generator_common.h>

namespace c2h::detail
{
class generator_state_t;

inline constexpr std::size_t max_cached_generator_states = 16;

class random_data_t
{
public:
  random_data_t(const random_data_t&)            = delete;
  random_data_t& operator=(const random_data_t&) = delete;

  random_data_t(random_data_t&&) noexcept;
  random_data_t& operator=(random_data_t&&) noexcept;

  ~random_data_t();

  [[nodiscard]] float* data() const noexcept
  {
    return m_data;
  }

private:
  friend class generator_t;

  random_data_t(float* data, std::shared_ptr<generator_state_t> state) noexcept;

  float* m_data = nullptr;
  std::shared_ptr<generator_state_t> m_state;
};

// called once from main to set up the generator state
void init_generator();

// Sets the seed and fills the per-device default-stream distribution. The returned object keeps the distribution alive;
// enqueue all consumers before destroying it.
[[nodiscard]] random_data_t prepare_random_data(seed_t seed, std::size_t num_items);

// Sets the seed and fills the per-device, per-stream distribution. The returned object keeps the distribution alive;
// enqueue all consumers on stream before destroying it.
[[nodiscard]] random_data_t prepare_random_data(::cuda::stream_ref stream, seed_t seed, std::size_t num_items);

[[nodiscard]] std::size_t cached_generator_state_count();

// called once before main returns to clean up the generator state
void cleanup_generator();

template <typename T, bool = ::cuda::is_floating_point_v<T>>
struct random_to_item_t
{
  float m_min;
  float m_max;

  __host__ __device__ random_to_item_t(T min, T max)
      : m_min(static_cast<float>(min))
      , m_max(static_cast<float>(max))
  {}

  __device__ T operator()(float random_value)
  {
    return static_cast<T>((m_max - m_min) * random_value + m_min);
  }
};

template <typename T>
struct random_to_item_t<T, true>
{
  using storage_t = ::cuda::std::_If<(sizeof(T) > 4), double, float>;
  storage_t m_min;
  storage_t m_max;

  __host__ __device__ random_to_item_t(T min, T max)
      : m_min(static_cast<storage_t>(min))
      , m_max(static_cast<storage_t>(max))
  {}

  __device__ T operator()(float random_value)
  {
    return static_cast<T>(m_max * random_value + m_min * (1.0f - random_value));
  }
};

template <typename T>
struct random_to_item_t<cuda::std::complex<T>, false>
{
  cuda::std::complex<T> m_min;
  cuda::std::complex<T> m_max;

  __host__ __device__ random_to_item_t(cuda::std::complex<T> min, cuda::std::complex<T> max)
      : m_min(min)
      , m_max(max)
  {}

  __device__ cuda::std::complex<T> operator()(float random_value) const
  {
    return (m_max - m_min) * cuda::std::complex<T>(random_value) + m_min;
  }
};
} // namespace c2h::detail
