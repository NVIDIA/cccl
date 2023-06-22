#pragma once

#include <cuda/std/complex>

#include <thrust/device_vector.h>

#include <limits>
#include <stdexcept>

#include <nvbench/nvbench.cuh>

using complex = cuda::std::complex<float>;
using int128_t = __int128_t;
using uint128_t = __uint128_t;

NVBENCH_DECLARE_TYPE_STRINGS(int128_t, "I128", "int128_t");
NVBENCH_DECLARE_TYPE_STRINGS(uint128_t, "U128", "uint128_t");
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
using fundamental_types = nvbench::type_list<TUNE_T>;
using all_types = nvbench::type_list<TUNE_T>;
#else
using fundamental_types = nvbench::type_list<int8_t,
                                             int16_t,
                                             int32_t,
                                             int64_t,
                                             int128_t,
                                             float,
                                             double>;
                                             
using all_types = nvbench::type_list<int8_t,
                                     int16_t,
                                     int32_t,
                                     int64_t,
                                     int128_t,
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
    case bit_entropy::_0_000: return 0.0;
    case bit_entropy::_0_811: return 0.811;
    case bit_entropy::_0_544: return 0.544;
    case bit_entropy::_0_337: return 0.337;
    case bit_entropy::_0_201: return 0.201;
    case bit_entropy::_1_000: return 1.0;
    default: return 0.0;
  }
}

[[nodiscard]] bit_entropy str_to_entropy(std::string str) 
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

template <typename T>
void gen(seed_t seed,
         thrust::device_vector<T> &data,
         bit_entropy entropy = bit_entropy::_1_000,
         T min = std::numeric_limits<T>::min(),
         T max = std::numeric_limits<T>::max());

template <typename T>
thrust::device_vector<T> gen_power_law_offsets(seed_t seed,
                                               std::size_t total_elements,
                                               std::size_t total_segments);

template <typename T>
thrust::device_vector<T> gen_power_law_key_segments(seed_t seed,
                                                    std::size_t total_elements,
                                                    std::size_t total_segments);

template <typename T>
thrust::device_vector<T> gen_uniform_offsets(seed_t seed,
                                             T total_elements,
                                             T min_segment_size,
                                             T max_segment_size);

template <typename T>
thrust::device_vector<T> gen_uniform_key_segments(seed_t seed,
                                                  std::size_t total_elements,
                                                  std::size_t min_segment_size,
                                                  std::size_t max_segment_size);

// #define DBG_ENTROPY

// This is very slow to compile, do not enable by default
#ifdef DBG_ENTROPY
#include <thrust/host_vector.h>
#include <thrust/sort.h>

#include <cub/device/device_run_length_encode.cuh>

template <class T>
double get_expected_entropy(bit_entropy in_entropy)
{
  if (in_entropy == bit_entropy::_0_000) {
    return 0.0;
  }

  if (in_entropy == bit_entropy::_1_000) {
    return sizeof(T) * 8;
  }

  const int samples = static_cast<int>(in_entropy) + 1;
  const double p1 = std::pow(0.5, samples);
  const double p2 = 1 - p1;
  const double entropy = (-p1 * std::log2(p1)) + (-p2 * std::log2(p2));
  return sizeof(T) * 8 * entropy;
}

template <class T>
double compute_actual_entropy(thrust::device_vector<T> in)
{
  const int n = static_cast<int>(in.size());
  thrust::device_vector<T> unique(n);
  thrust::device_vector<int> counts(n);
  thrust::device_vector<int> num_runs(1);

  thrust::sort(in.begin(), in.end());

  // RLE
  void *d_temp_storage           = nullptr;
  std::size_t temp_storage_bytes = 0;

  T *d_in             = thrust::raw_pointer_cast(in.data());
  T *d_unique_out     = thrust::raw_pointer_cast(unique.data());
  int *d_counts_out   = thrust::raw_pointer_cast(counts.data());
  int *d_num_runs_out = thrust::raw_pointer_cast(num_runs.data());

  cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_unique_out,
                                     d_counts_out,
                                     d_num_runs_out,
                                     n);

  thrust::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  cub::DeviceRunLengthEncode::Encode(d_temp_storage,
                                     temp_storage_bytes,
                                     d_in,
                                     d_unique_out,
                                     d_counts_out,
                                     d_num_runs_out,
                                     n);

  thrust::host_vector<int> h_counts   = counts;
  thrust::host_vector<int> h_num_runs = num_runs;

  // normalize counts
  thrust::host_vector<double> ps(h_num_runs[0]);
  for (std::size_t i = 0; i < ps.size(); i++)
  {
    ps[i] = static_cast<double>(h_counts[i]) / n;
  }

  double entropy = 0.0;

  if (ps.size())
  {
    for (double p : ps)
    {
      entropy -= p * std::log2(p);
    }
  }

  return entropy;
}

template <class T>
void report_entropy(thrust::device_vector<T>& buffer_1, bit_entropy entropy)
{
  std::cout << "Actual entropy: " << compute_actual_entropy(buffer_1) << std::endl;
  std::cout << "Expected entropy: " << get_expected_entropy<T>(entropy) << std::endl;
}
#else
template <class T>
void report_entropy(thrust::device_vector<T>&, bit_entropy)
{
}

struct less_t
{
  template <typename DataType>
  __device__ bool operator()(const DataType &lhs, const DataType &rhs)
  {
    return lhs < rhs;
  }
};

template <>
__device__ bool less_t::operator()(const complex &lhs, const complex &rhs) {
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
  __device__ DataType operator()(const DataType &lhs, const DataType &rhs)
  {
    less_t less{};
    return less(lhs, rhs) ? rhs : lhs;
  }
};

#endif
