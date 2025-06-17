// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#define C2H_EXPORTS

#include <c2h/generators.h>
#include <c2h/vector.h>

#if C2H_HAS_CURAND
#  include <curand.h>
#else
#  include <thrust/random.h>
#endif

namespace c2h
{

class generator_t
{
private:
  generator_t();

public:
  static generator_t& instance();
  ~generator_t();

  template <typename T>
  void operator()(seed_t seed,
                  c2h::device_vector<T>& data,
                  T min = std::numeric_limits<T>::min(),
                  T max = std::numeric_limits<T>::max());

  template <typename T>
  void operator()(modulo_t modulo, c2h::device_vector<T>& data);

  float* distribution();

#if C2H_HAS_CURAND
  curandGenerator_t& gen()
  {
    return m_gen;
  }
#endif // C2H_HAS_CURAND

  float* prepare_random_generator(seed_t seed, std::size_t num_items);

  void generate();

private:
#if C2H_HAS_CURAND
  curandGenerator_t m_gen;
#else
  thrust::default_random_engine m_re;
#endif
  c2h::device_vector<float> m_distribution;
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

} // namespace c2h
