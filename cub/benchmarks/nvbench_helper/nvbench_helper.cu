#include <thrust/distance.h>
#include <thrust/binary_search.h>
#include <thrust/iterator/transform_output_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/execution_policy.h>
#include <thrust/fill.h>
#include <thrust/count.h>
#include <thrust/for_each.h>
#include <thrust/tabulate.h>

#include <cub/device/device_copy.cuh>

#include <cstdint>
#include <random>

#include "thrust/scan.h"
#include <curand.h>
#include <nvbench_helper.cuh>

class generator_t
{
public:
  generator_t();
  ~generator_t();

  template <typename T>
  void operator()(seed_t seed,
                  thrust::device_vector<T> &data,
                  bit_entropy entropy,
                  T min = std::numeric_limits<T>::lowest(),
                  T max = std::numeric_limits<T>::max());

  template <typename T>
  thrust::device_vector<T> power_law_segment_offsets(seed_t seed,
                                                     std::size_t total_elements,
                                                     std::size_t total_segments);

  double *distribution();
  curandGenerator_t &gen() { return m_gen; }

  double *prepare_random_generator(seed_t seed, std::size_t num_items);
  double *prepare_lognormal_random_generator(seed_t seed, std::size_t num_items);

private:
  curandGenerator_t m_gen;
  thrust::device_vector<double> m_distribution;
};

template <typename T>
struct random_to_item_t
{
  double m_min;
  double m_max;

  __host__ __device__ random_to_item_t(T min, T max)
      : m_min(static_cast<double>(min))
      , m_max(static_cast<double>(max))
  {}

  __host__ __device__ T operator()(double random_value)
  {
    return static_cast<T>((m_max - m_min) * random_value + m_min);
  }
};

generator_t::generator_t() { curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT); }

generator_t::~generator_t() { curandDestroyGenerator(m_gen); }

double *generator_t::distribution() { return thrust::raw_pointer_cast(m_distribution.data()); }

double *generator_t::prepare_random_generator(seed_t seed, std::size_t num_items)
{
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());

  m_distribution.resize(num_items);
  curandGenerateUniformDouble(m_gen, this->distribution(), num_items);

  return this->distribution();
}

double *generator_t::prepare_lognormal_random_generator(seed_t seed, std::size_t num_segments)
{
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());

  m_distribution.resize(num_segments);

  const double mean = 3.0;
  const double sigma = 1.2;
  curandGenerateLogNormalDouble(m_gen, this->distribution(), num_segments, mean, sigma);

  return this->distribution();
}

template <class T>
__global__ void and_kernel(T *d_in, T *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    d_in[i] = d_in[i] & d_tmp[i];
  }
}

__global__ void and_kernel(float *d_in, float *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    std::uint32_t result = reinterpret_cast<std::uint32_t &>(d_in[i]) &
                           reinterpret_cast<std::uint32_t &>(d_tmp[i]);
    d_in[i] = reinterpret_cast<float &>(result);
  }
}

__global__ void and_kernel(double *d_in, double *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    std::uint64_t result = reinterpret_cast<std::uint64_t &>(d_in[i]) &
                           reinterpret_cast<std::uint64_t &>(d_tmp[i]);
    d_in[i] = reinterpret_cast<double &>(result);
  }
}

__global__ void and_kernel(complex *d_in, complex *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    double in_real = d_in[i].real();
    double in_imag = d_in[i].imag();

    double tmp_real = d_tmp[i].real();
    double tmp_imag = d_tmp[i].imag();

    std::uint64_t result_real = reinterpret_cast<std::uint64_t &>(in_real) &
                                reinterpret_cast<std::uint64_t &>(tmp_real);

    std::uint64_t result_imag = reinterpret_cast<std::uint64_t &>(in_imag) &
                                reinterpret_cast<std::uint64_t &>(tmp_imag);

    d_in[i].real(reinterpret_cast<double &>(result_real));
    d_in[i].imag(reinterpret_cast<double &>(result_imag));
  }
}

__global__ void set_real_kernel(complex *d_in, complex min, complex max, double *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    d_in[i].real(random_to_item_t<double>{min.real(), max.real()}(d_tmp[i]));
  }
}

__global__ void set_imag_kernel(complex *d_in, complex min, complex max, double *d_tmp, std::size_t n)
{
  const std::size_t i = blockIdx.x * blockDim.x + threadIdx.x;

  if (i < n)
  {
    d_in[i].imag(random_to_item_t<double>{min.imag(), max.imag()}(d_tmp[i]));
  }
}

template <class T>
struct lognormal_transformer_t
{
  std::size_t total_elements;
  double sum;

  __device__ T operator()(double val)
  {
    return floor(val * total_elements / sum);
  }
};

template <class T>
__global__ void lognormal_adjust_kernel(T *segment_sizes, std::size_t diff)
{
  const unsigned tid = threadIdx.x + blockIdx.x * blockDim.x;

  if (tid < diff)
  {
    segment_sizes[tid]++;
  }
}

template <class T>
thrust::device_vector<T> generator_t::power_law_segment_offsets(seed_t seed,
                                                                std::size_t total_elements,
                                                                std::size_t total_segments)
{
  thrust::device_vector<T> segment_sizes(total_segments + 1);
  prepare_lognormal_random_generator(seed, total_segments);

  if (thrust::count(m_distribution.begin(), m_distribution.end(), 0.0) == total_segments)
  {
    thrust::fill_n(m_distribution.begin(), total_segments, 1.0);
  }

  double sum = thrust::reduce(m_distribution.begin(), m_distribution.end());
  thrust::transform(m_distribution.begin(),
                    m_distribution.end(),
                    segment_sizes.begin(),
                    lognormal_transformer_t<T>{total_elements, sum});

  const int diff = total_elements - thrust::reduce(segment_sizes.begin(), segment_sizes.end());
  const int block_size = 256;
  const int grid_size  = (std::abs(diff) + block_size - 1) / block_size;

  T *d_segment_sizes = thrust::raw_pointer_cast(segment_sizes.data());
  lognormal_adjust_kernel<T><<<grid_size, block_size>>>(d_segment_sizes, diff);

  thrust::exclusive_scan(segment_sizes.begin(), segment_sizes.end(), segment_sizes.begin());

  return segment_sizes;
}

template <class T>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<T> &data,
                             bit_entropy entropy,
                             T min,
                             T max)
{
  switch (entropy)
  {
    case bit_entropy::_1_000: {
      prepare_random_generator(seed, data.size());

      thrust::transform(m_distribution.begin(),
                        m_distribution.end(),
                        data.begin(),
                        random_to_item_t<T>(min, max));
      return;
    }
    case bit_entropy::_0_000: {
      std::mt19937 rng;
      rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      T random_value = random_to_item_t<T>(min, max)(dist(rng));
      thrust::fill(data.begin(), data.end(), random_value);
      return;
    }
    default: {
      prepare_random_generator(seed, data.size());

      thrust::transform(m_distribution.begin(),
                        m_distribution.end(),
                        data.begin(),
                        random_to_item_t<T>(min, max));

      const int number_of_steps = static_cast<int>(entropy);
      thrust::device_vector<T> tmp(data.size());
      const int threads_in_block = 256;
      const int blocks_in_grid   = (data.size() + threads_in_block - 1) / threads_in_block;

      for (int i = 0; i < number_of_steps; i++, ++seed)
      {
        (*this)(seed, tmp, bit_entropy::_1_000, min, max);
        and_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                         thrust::raw_pointer_cast(tmp.data()),
                                                         data.size());
        cudaStreamSynchronize(0);
      }
      return;
    }
  };
}

template <>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<complex> &data,
                             bit_entropy entropy,
                             complex min,
                             complex max)
{
  const int threads_in_block = 256;
  const int blocks_in_grid   = (data.size() + threads_in_block - 1) / threads_in_block;

  switch (entropy)
  {
    case bit_entropy::_1_000: {
      prepare_random_generator(seed, data.size()); ++seed;
      set_real_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());
      prepare_random_generator(seed, data.size()); ++seed;
      set_imag_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());
      return;
    }
    case bit_entropy::_0_000: {
      std::mt19937 rng;
      rng.seed(static_cast<std::mt19937::result_type>(seed.get()));
      std::uniform_real_distribution<float> dist(0.0f, 1.0f);
      double random_imag = random_to_item_t<double>(min.imag(), max.imag())(dist(rng));
      double random_real = random_to_item_t<double>(min.imag(), max.imag())(dist(rng));
      complex random_value(random_real, random_imag);
      thrust::fill(data.begin(), data.end(), random_value);
      return;
    }
    default: {
      prepare_random_generator(seed, data.size());

      prepare_random_generator(seed, data.size()); ++seed;
      set_real_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());
      prepare_random_generator(seed, data.size()); ++seed;
      set_imag_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                            min,
                                                            max,
                                                            thrust::raw_pointer_cast(m_distribution.data()),
                                                            data.size());

      const int number_of_steps = static_cast<int>(entropy);
      thrust::device_vector<complex> tmp(data.size());

      for (int i = 0; i < number_of_steps; i++, ++seed)
      {
        (*this)(seed, tmp, bit_entropy::_1_000, min, max);
        and_kernel<<<blocks_in_grid, threads_in_block>>>(thrust::raw_pointer_cast(data.data()),
                                                         thrust::raw_pointer_cast(tmp.data()),
                                                         data.size());
        cudaStreamSynchronize(0);
      }
      return;
    }
  };
}

struct random_to_probability_t
{
  double m_probability;

  __host__ __device__ bool operator()(double random_value)
  {
    return random_value < m_probability;
  }
};

template <>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<bool> &data,
                             bit_entropy entropy,
                             bool /* min */,
                             bool /* max */)
{
  if (entropy == bit_entropy::_0_000)
  {
    thrust::fill(data.begin(), data.end(), false);
  }
  else if (entropy == bit_entropy::_1_000)
  {
    thrust::fill(data.begin(), data.end(), true);
  }
  else
  {
    prepare_random_generator(seed, data.size());
    thrust::transform(m_distribution.begin(),
                      m_distribution.end(),
                      data.begin(),
                      random_to_probability_t{entropy_to_probability(entropy)});
  }
}

template <typename T>
void gen(seed_t seed, thrust::device_vector<T> &data, bit_entropy entropy, T min, T max)
{
  generator_t{}(seed, data, entropy, min, max);
}

#define INSTANTIATE_RND(TYPE)                                                                      \
  template void gen<TYPE>(seed_t,                                                                  \
                          thrust::device_vector<TYPE> & data,                                      \
                          bit_entropy,                                                             \
                          TYPE min,                                                                \
                          TYPE max)

#define INSTANTIATE(TYPE) INSTANTIATE_RND(TYPE);

INSTANTIATE(bool);

INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(uint128_t);

INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(int128_t);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(complex);

#undef INSTANTIATE
#undef INSTANTIATE_RND


template <typename T>
thrust::device_vector<T> gen_power_law_offsets(seed_t seed,
                                               std::size_t total_elements,
                                               std::size_t total_segments)
{
  return generator_t{}.power_law_segment_offsets<T>(seed, total_elements, total_segments);
}

#define INSTANTIATE(TYPE)                                                                          \
  template thrust::device_vector<TYPE> gen_power_law_offsets<TYPE>(seed_t, std::size_t, std::size_t)

INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
#undef INSTANTIATE


template <class T>
struct offset_to_iterator_t
{
  T *base_it;

  __host__ __device__ __forceinline__ T*
  operator()(std::size_t offset) const
  {
    return base_it + offset;
  }
};

template <class T>
struct repeat_index_t
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<T> operator()(std::size_t i)
  {
    return thrust::constant_iterator<T>(static_cast<T>(i));
  }
};

struct offset_to_size_t
{
  std::size_t *offsets = nullptr;

  __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
  {
    return offsets[i + 1] - offsets[i];
  }
};

template <typename T>
thrust::device_vector<T> gen_power_law_key_segments(seed_t seed,
                                                    std::size_t total_elements,
                                                    thrust::device_vector<std::size_t> &segment_offsets)
{
  std::size_t total_segments = segment_offsets.size() - 1;
  thrust::device_vector<T> out(total_elements);
  std::size_t *d_offsets = thrust::raw_pointer_cast(segment_offsets.data());
  T *d_out = thrust::raw_pointer_cast(out.data());

  thrust::counting_iterator<int> iota(0);
  offset_to_iterator_t<T> dst_transform_op{d_out};

  auto d_range_srcs = thrust::make_transform_iterator(iota, repeat_index_t<T>{});
  auto d_range_dsts = thrust::make_transform_iterator(d_offsets, dst_transform_op);
  auto d_range_sizes = thrust::make_transform_iterator(iota, offset_to_size_t{d_offsets});

  std::uint8_t *d_temp_storage = nullptr;
  std::size_t temp_storage_bytes = 0;
  cub::DeviceCopy::Batched(d_temp_storage,
                           temp_storage_bytes,
                           d_range_srcs,
                           d_range_dsts,
                           d_range_sizes,
                           total_segments);
 
  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceCopy::Batched(d_temp_storage,
                           temp_storage_bytes,
                           d_range_srcs,
                           d_range_dsts,
                           d_range_sizes,
                           total_segments);
  cudaDeviceSynchronize();

  return out;
}

template <typename T>
thrust::device_vector<T> gen_power_law_key_segments(seed_t seed,
                                                    std::size_t total_elements,
                                                    std::size_t total_segments)
{
  thrust::device_vector<std::size_t> segment_offsets =
    gen_power_law_offsets<std::size_t>(seed, total_elements, total_segments);
  return gen_power_law_key_segments<T>(seed, total_elements, segment_offsets);
}

#define INSTANTIATE(TYPE)                                                                          \
  template thrust::device_vector<TYPE> gen_power_law_key_segments<TYPE>(seed_t,                    \
                                                                        std::size_t,               \
                                                                        std::size_t)

INSTANTIATE(bool);

INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(uint128_t);

INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(int128_t);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(complex);
#undef INSTANTIATE

template <class T>
struct gt_t
{
  T val;

  __device__ bool operator()(T x)
  {
    return x > val;
  }
};

template <typename T>
thrust::device_vector<T> gen_uniform_offsets(seed_t seed,
                                             T total_elements,
                                             T min_segment_size,
                                             T max_segment_size)
{
  thrust::device_vector<T> segment_offsets(total_elements + 2);
  gen(seed, segment_offsets, bit_entropy::_1_000, min_segment_size, max_segment_size);
  segment_offsets[total_elements] = total_elements + 1;
  thrust::exclusive_scan(segment_offsets.begin(), segment_offsets.end(), segment_offsets.begin());
  typename thrust::device_vector<T>::iterator iter =
    thrust::find_if(segment_offsets.begin(), segment_offsets.end(), gt_t<T>{total_elements});
  *iter = total_elements;
  segment_offsets.erase(iter + 1, segment_offsets.end());
  return segment_offsets;
}

#define INSTANTIATE(TYPE)                                                                          \
  template thrust::device_vector<TYPE> gen_uniform_offsets<TYPE>(seed_t, TYPE, TYPE, TYPE)

INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);

#undef INSTANTIATE

/**
 * @brief Generates a vector of random key segments.
 *
 * Not all parameter combinations can be satisfied. For instance, if the total
 * elements is less than the minimal segment size, the function will return a
 * vector with a single element that is outside of the requested range. 
 * At most one segment can be out of the requested range.
 */
template <typename T>
thrust::device_vector<T> gen_uniform_key_segments(seed_t seed,
                                                  std::size_t total_elements,
                                                  std::size_t min_segment_size,
                                                  std::size_t max_segment_size)
{
  thrust::device_vector<std::size_t> segment_offsets =
    gen_uniform_offsets(seed, total_elements, min_segment_size, max_segment_size);
  return gen_power_law_key_segments<T>(seed, total_elements, segment_offsets);
}

#define INSTANTIATE(TYPE)                                                                          \
  template thrust::device_vector<TYPE> gen_uniform_key_segments<TYPE>(seed_t,                      \
                                                                      std::size_t,                 \
                                                                      std::size_t,                 \
                                                                      std::size_t)

INSTANTIATE(bool);

INSTANTIATE(uint8_t);
INSTANTIATE(uint16_t);
INSTANTIATE(uint32_t);
INSTANTIATE(uint64_t);
INSTANTIATE(uint128_t);

INSTANTIATE(int8_t);
INSTANTIATE(int16_t);
INSTANTIATE(int32_t);
INSTANTIATE(int64_t);
INSTANTIATE(int128_t);

INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(complex);
#undef INSTANTIATE
