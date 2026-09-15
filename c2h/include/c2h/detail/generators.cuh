// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#pragma once

#include <cuda/std/complex>
#include <cuda/std/cstddef>
#include <cuda/type_traits>

#include <c2h/generator_common.h>

#if C2H_HAS_CURAND
#  include <curand_kernel.h>
#else
#  include <cuda/std/random>
#endif

namespace c2h::detail
{
// draws a single uniform float in (0, 1] from an independent stream per index, so many indices can be drawn
// concurrently without any shared state
struct index_to_random_uniform
{
  unsigned long long m_seed;

  __device__ float operator()(std::size_t i) const
  {
#if C2H_HAS_CURAND
    curandStatePhilox4_32_10_t state;
    curand_init(m_seed, i, 0, &state);
    return curand_uniform(&state);
#else
    cuda::std::philox4x32 engine(static_cast<cuda::std::philox4x32::result_type>(m_seed ^ (m_seed >> 32)));
    engine.set_counter(
      {0,
       0,
       static_cast<cuda::std::philox4x32::result_type>(i >> 32),
       static_cast<cuda::std::philox4x32::result_type>(i)});
    return cuda::std::uniform_real_distribution<float>{0.0f, 1.0f}(engine);
#endif // C2H_HAS_CURAND
  }
};

template <typename Op>
struct index_to_transformed_random_uniform
{
  unsigned long long m_seed;
  Op m_op;

  __device__ auto operator()(std::size_t i)
  {
    return m_op(index_to_random_uniform{m_seed}(i));
  }
};

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
