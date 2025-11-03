// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <cuda/std/complex>

#include <c2h/generators.h>
#include <c2h/vector.h>

#if C2H_HAS_CURAND
#  include <curand.h>
#else
#  include <thrust/random.h>
#endif

namespace c2h::detail
{
class generator_t
{
public:
  generator_t()
  {
#if C2H_HAS_CURAND
    curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
#endif
  }

  ~generator_t()
  {
#if C2H_HAS_CURAND
    curandDestroyGenerator(m_gen);
#endif
  }

  // sets the seed and resizes the distribution vector, fills it by calling generate(), and returns a pointer the start
  // of the data
  float* prepare_random_generator(seed_t seed, std::size_t num_items);

  // re-fills the currently held distribution vector with new random values
  void generate();

private:
#if C2H_HAS_CURAND
  curandGenerator_t m_gen;
#else
  thrust::default_random_engine m_re;
#endif
  c2h::device_vector<float> m_distribution;
};

inline generator_t generator;

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
