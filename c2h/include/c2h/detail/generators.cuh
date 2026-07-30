// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda/std/complex>

#include <c2h/generators.h>

namespace c2h::detail
{
// called once from main to set up the generator state
void init_generator();

// sets the seed and resizes the distribution vector, fills it, and returns a pointer the start of the data
float* prepare_random_data(seed_t seed, std::size_t num_items);

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
