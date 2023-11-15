#pragma once

#include <cuda/std/complex>

#include <thrust/device_vector.h>

#include <limits>
#include <stdexcept>

#include <nvbench/nvbench.cuh>
#include <cuda/std/span>

#if defined(_MSC_VER)
#define NVBENCH_HELPER_HAS_I128 0
#else
#define NVBENCH_HELPER_HAS_I128 1
#endif

#if NVBENCH_HELPER_HAS_I128
using int128_t = __int128_t;
using uint128_t = __uint128_t;

NVBENCH_DECLARE_TYPE_STRINGS(int128_t, "I128", "int128_t");
NVBENCH_DECLARE_TYPE_STRINGS(uint128_t, "U128", "uint128_t");
#endif

using complex = cuda::std::complex<float>;

NVBENCH_DECLARE_TYPE_STRINGS(complex, "C64", "complex");

namespace detail 
{

template <class T, class List>
struct push_back
{};

template <class T, class... As>
struct push_back<T, nvbench::type_list<As...>>
{
  using type = nvbench::type_list<As..., T>;
};

}

template <class T, class List>
using push_back_t = typename detail::push_back<T, List>::type;

#ifdef TUNE_OffsetT
using offset_types = nvbench::type_list<TUNE_OffsetT>;
#else
using offset_types = nvbench::type_list<int32_t, int64_t>;
#endif

#ifdef TUNE_T
using integral_types    = nvbench::type_list<TUNE_T>;
using fundamental_types = nvbench::type_list<TUNE_T>;
using all_types         = nvbench::type_list<TUNE_T>;
#else
using integral_types = nvbench::type_list<int8_t, int16_t, int32_t, int64_t>;

using fundamental_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#  if NVBENCH_HELPER_HAS_I128
                     int128_t,
#  endif
                     float,
                     double>;

using all_types =
  nvbench::type_list<int8_t,
                     int16_t,
                     int32_t,
                     int64_t,
#  if NVBENCH_HELPER_HAS_I128
                     int128_t,
#  endif
                     float,
                     double,
                     complex>;
#endif

template <class T>
class value_wrapper_t
{
  T m_val{};

public:
  explicit value_wrapper_t(T val)
      : m_val(val)
  {}

  T get() const { return m_val; }

  value_wrapper_t& operator++() {
    m_val++;
    return *this;
  }
};

class seed_t : public value_wrapper_t<unsigned long long int>
{
public:
  using value_wrapper_t::value_wrapper_t;
  using value_wrapper_t::operator++;

  seed_t()
      : value_wrapper_t(42)
  {}
};

enum class bit_entropy 
{
  _1_000 = 0,
  _0_811 = 1,
  _0_544 = 2,
  _0_337 = 3,
  _0_201 = 4,
  _0_000 = 4200
};
NVBENCH_DECLARE_TYPE_STRINGS(bit_entropy, "BE", "bit entropy");

[[nodiscard]]
inline double entropy_to_probability(bit_entropy entropy)
{
  switch (entropy)
  {
    case bit_entropy::_1_000: return 1.0;
    case bit_entropy::_0_811: return 0.811;
    case bit_entropy::_0_544: return 0.544;
    case bit_entropy::_0_337: return 0.337;
    case bit_entropy::_0_201: return 0.201;
    case bit_entropy::_0_000: return 0.0;
    default: return 0.0;
  }
}

[[nodiscard]] inline bit_entropy str_to_entropy(std::string str) 
{
  if (str == "1.000") 
  {
    return bit_entropy::_1_000;
  }
  else if (str == "0.811") 
  {
    return bit_entropy::_0_811;
  }
  else if (str == "0.544") 
  {
    return bit_entropy::_0_544;
  }
  else if (str == "0.337") 
  {
    return bit_entropy::_0_337;
  }
  else if (str == "0.201") 
  {
    return bit_entropy::_0_201;
  }
  else if (str == "0.000") 
  {
    return bit_entropy::_0_000;
  }

  throw std::runtime_error("Can't convert string to bit entropy");
}

namespace detail
{

void do_not_optimize(const void *ptr);

template <typename T>
void gen_host(seed_t seed, cuda::std::span<T> data, bit_entropy entropy, T min, T max);

template <typename T>
void gen_device(seed_t seed, cuda::std::span<T> data, bit_entropy entropy, T min, T max);

template <typename T>
void gen_uniform_key_segments_host(seed_t seed,
                                   cuda::std::span<T> data,
                                   std::size_t min_segment_size,
                                   std::size_t max_segment_size);

template <typename T>
void gen_uniform_key_segments_device(seed_t seed,
                                     cuda::std::span<T> data,
                                     std::size_t min_segment_size,
                                     std::size_t max_segment_size);

template <typename T>
std::size_t gen_uniform_segment_offsets_host(seed_t seed,
                                             cuda::std::span<T> segment_offsets,
                                             std::size_t min_segment_size,
                                             std::size_t max_segment_size);

template <typename T>
std::size_t gen_uniform_segment_offsets_device(seed_t seed,
                                               cuda::std::span<T> segment_offsets,
                                               std::size_t min_segment_size,
                                               std::size_t max_segment_size);

template <typename T>
void gen_power_law_segment_offsets_host(seed_t seed,
                                        cuda::std::span<T> segment_offsets,
                                        std::size_t elements);

template <typename T>
void gen_power_law_segment_offsets_device(seed_t seed,
                                          cuda::std::span<T> segment_offsets,
                                          std::size_t elements);

namespace 
{

struct generator_base_t
{
  seed_t m_seed{};
  const std::size_t m_elements{0};
  const bit_entropy m_entropy{bit_entropy::_1_000};

  template <typename T>
  thrust::device_vector<T> generate(T min, T max)
  {
    thrust::device_vector<T> vec(m_elements); 
    cuda::std::span<T> span(thrust::raw_pointer_cast(vec.data()), m_elements);
    #if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    gen_device(m_seed, span, m_entropy, min, max);
    #else
    gen_host(m_seed, span, m_entropy, min, max);
    #endif
    ++m_seed;
    return vec;
  }
};

template <class T>
struct vector_generator_t : generator_base_t
{
  const T m_min{std::numeric_limits<T>::min()};
  const T m_max{std::numeric_limits<T>::max()};

  operator thrust::device_vector<T>()
  {
    return generator_base_t::generate(m_min, m_max);
  }
};

template <>
struct vector_generator_t<void> : generator_base_t
{
  template <typename T>
  operator thrust::device_vector<T>()
  {
    return generator_base_t::generate(std::numeric_limits<T>::min(), std::numeric_limits<T>::max());
  }

  // This overload is needed because numeric limits is not specialized for complex, making 
  // the min and max values for complex equal zero. 
  operator thrust::device_vector<complex>()
  {
    const complex min = complex{std::numeric_limits<complex::value_type>::min(),
                                std::numeric_limits<complex::value_type>::min()};
    const complex max = complex{std::numeric_limits<complex::value_type>::max(),
                                std::numeric_limits<complex::value_type>::max()};

    return generator_base_t::generate(min, max);
  }
};

struct uniform_key_segments_generator_t
{
  seed_t m_seed{};
  const std::size_t m_total_elements{0};
  const std::size_t m_min_segment_size{0};
  const std::size_t m_max_segment_size{0};

  template <class KeyT>
  operator thrust::device_vector<KeyT>()
  {
    thrust::device_vector<KeyT> keys_vec(m_total_elements);
    cuda::std::span<KeyT> keys(thrust::raw_pointer_cast(keys_vec.data()), keys_vec.size());
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    gen_uniform_key_segments_device(m_seed,
                                    keys,
                                    m_min_segment_size,
                                    m_max_segment_size);
#else
    gen_uniform_key_segments_host(m_seed,
                                  keys,
                                  m_min_segment_size,
                                  m_max_segment_size);
#endif
    ++m_seed;
    return keys_vec;
  }
};

struct uniform_segment_offsets_generator_t
{
  seed_t m_seed{};
  const std::size_t m_total_elements{0};
  const std::size_t m_min_segment_size{0};
  const std::size_t m_max_segment_size{0};

  template <class OffsetT>
  operator thrust::device_vector<OffsetT>()
  {
    thrust::device_vector<OffsetT> offsets_vec(m_total_elements + 2);
    cuda::std::span<OffsetT> offsets(thrust::raw_pointer_cast(offsets_vec.data()),
                                     offsets_vec.size());
    const std::size_t offsets_size =
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
      gen_uniform_segment_offsets_device(m_seed, offsets, m_min_segment_size, m_max_segment_size);
#else
      gen_uniform_segment_offsets_host(m_seed, offsets, m_min_segment_size, m_max_segment_size);
#endif
    offsets_vec.resize(offsets_size);
    offsets_vec.shrink_to_fit();
    ++m_seed;
    return offsets_vec;
  }
};

struct power_law_segment_offsets_generator_t
{
  seed_t m_seed{};
  const std::size_t m_elements{0};
  const std::size_t m_segments{0};

  template <class OffsetT>
  operator thrust::device_vector<OffsetT>()
  {
    thrust::device_vector<OffsetT> offsets_vec(m_segments + 1);
    cuda::std::span<OffsetT> offsets(thrust::raw_pointer_cast(offsets_vec.data()),
                                     offsets_vec.size());
#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
    gen_power_law_segment_offsets_device(m_seed, offsets, m_elements);
#else
    gen_power_law_segment_offsets_host(m_seed, offsets, m_elements);
#endif
    ++m_seed;
    return offsets_vec;
  }
};

struct gen_uniform_key_segments_t
{
  uniform_key_segments_generator_t operator()(std::size_t total_elements,
                                              std::size_t min_segment_size,
                                              std::size_t max_segment_size) const
  {
    return {seed_t{}, total_elements, min_segment_size, max_segment_size};
  }
};

struct gen_uniform_segment_offsets_t
{
  uniform_segment_offsets_generator_t operator()(std::size_t total_elements,
                                                 std::size_t min_segment_size,
                                                 std::size_t max_segment_size) const
  {
    return {seed_t{}, total_elements, min_segment_size, max_segment_size};
  }
};

struct gen_uniform_t
{
  gen_uniform_key_segments_t key_segments{};
  gen_uniform_segment_offsets_t segment_offsets{};
};

struct gen_power_law_segment_offsets_t
{
  power_law_segment_offsets_generator_t operator()(std::size_t elements,
                                                   std::size_t segments) const
  {
    return {seed_t{}, elements, segments};
  }
};

struct gen_power_law_t
{
  gen_power_law_segment_offsets_t segment_offsets{}; 
};

struct gen_t
{
  vector_generator_t<void> operator()(std::size_t elements,
                                      bit_entropy entropy = bit_entropy::_1_000) const
  {
    return {seed_t{}, elements, entropy};
  }

  template <class T>
  vector_generator_t<T> operator()(std::size_t elements,
                                   bit_entropy entropy = bit_entropy::_1_000,
                                   T min               = std::numeric_limits<T>::min,
                                   T max               = std::numeric_limits<T>::max()) const
  {
    return {seed_t{}, elements, entropy, min, max};
  }

  gen_uniform_t uniform{};
  gen_power_law_t power_law{};
};
} // namespace

} // namespace detail

inline detail::gen_t generate;

template <class T>
void do_not_optimize(const T& val)
{
  detail::do_not_optimize(&val);
}

struct less_t
{
  template <typename DataType>
  __host__ __device__ bool operator()(const DataType &lhs, const DataType &rhs) const
  {
    return lhs < rhs;
  }
};

template <>
__host__ __device__ inline bool less_t::operator()(const complex &lhs, const complex &rhs) const {
  double magnitude_0 = cuda::std::abs(lhs);
  double magnitude_1 = cuda::std::abs(rhs);

  if (cuda::std::isnan(magnitude_0) || cuda::std::isnan(magnitude_1))
  {
    // NaN's are always equal.
    return false;
  }
  else if (cuda::std::isinf(magnitude_0) || cuda::std::isinf(magnitude_1))
  {
    // If the real or imaginary part of the complex number has a very large value 
    // (close to the maximum representable value for a double), it is possible that 
    // the magnitude computation can result in positive infinity:
    // ```cpp
    // const double large_number = std::numeric_limits<double>::max() / 2;
    // std::complex<double> z(large_number, large_number);
    // std::abs(z) == inf;
    // ```
    // Dividing both components by a constant before computing the magnitude prevents overflow.
    const complex::value_type scaler = 0.5;

    magnitude_0 = cuda::std::abs(lhs * scaler);
    magnitude_1 = cuda::std::abs(rhs * scaler);
  }

  const complex::value_type difference = cuda::std::abs(magnitude_0 - magnitude_1);
  const complex::value_type threshold = cuda::std::numeric_limits<complex::value_type>::epsilon() * 2;

  if (difference < threshold) {
    // Triangles with the same magnitude are sorted by their phase angle.
    const complex::value_type phase_angle_0 = cuda::std::arg(lhs);
    const complex::value_type phase_angle_1 = cuda::std::arg(rhs);

    return phase_angle_0 < phase_angle_1;
  } else {
    return magnitude_0 < magnitude_1;
  }
}

struct max_t
{
  template <typename DataType>
  __host__ __device__ DataType operator()(const DataType &lhs, const DataType &rhs)
  {
    less_t less{};
    return less(lhs, rhs) ? rhs : lhs;
  }
};
