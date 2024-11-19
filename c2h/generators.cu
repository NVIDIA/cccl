/******************************************************************************
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#define C2H_EXPORTS

#include <cub/device/device_copy.cuh>
#include <cub/util_type.cuh>

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/find.h>
#include <thrust/for_each.h>
#include <thrust/iterator/constant_iterator.h>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <thrust/scan.h>
#include <thrust/tabulate.h>

#include <cuda/std/type_traits>

#include <cstdint>

#include <c2h/custom_type.h>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/fill_striped.h>
#include <c2h/generators.h>
#include <c2h/vector.h>

#if C2H_HAS_CURAND
#  include <curand.h>
#else
#  include <thrust/random.h>
#endif

namespace c2h
{

#if !C2H_HAS_CURAND
struct i_to_rnd_t
{
  __host__ __device__ i_to_rnd_t(thrust::default_random_engine engine)
      : m_engine(engine)
  {}

  thrust::default_random_engine m_engine{};

  template <typename IndexType>
  __host__ __device__ float operator()(IndexType n)
  {
    m_engine.discard(n);
    return thrust::uniform_real_distribution<float>{0.0f, 1.0f}(m_engine);
  }
};
#endif // !C2H_HAS_CURAND

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

// TODO(bgruber): modelled after cub::Traits. We should generalize this somewhere into libcu++.
template <typename T>
struct is_floating_point : ::cuda::std::is_floating_point<T>
{};
#ifdef _CCCL_HAS_NVFP16
template <>
struct is_floating_point<__half> : ::cuda::std::true_type
{};
#endif // _CCCL_HAS_NVFP16
#ifdef _CCCL_HAS_NVBF16
template <>
struct is_floating_point<__nv_bfloat16> : ::cuda::std::true_type
{};
#endif // _CCCL_HAS_NVBF16
#ifdef __CUDA_FP8_TYPES_EXIST__
template <>
struct is_floating_point<__nv_fp8_e4m3> : ::cuda::std::true_type
{};
template <>
struct is_floating_point<__nv_fp8_e5m2> : ::cuda::std::true_type
{};
#endif // __CUDA_FP8_TYPES_EXIST__

template <typename T, bool = is_floating_point<T>::value>
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

template <typename T, int VecItem>
struct random_to_vec_item_t;

#define RANDOM_TO_VEC_ITEM_SPEC(VEC_ITEM, VEC_FIELD)                               \
  template <typename T>                                                            \
  struct random_to_vec_item_t<T, VEC_ITEM>                                         \
  {                                                                                \
    __device__ void operator()(std::size_t idx)                                    \
    {                                                                              \
      auto min             = m_min.VEC_FIELD;                                      \
      auto max             = m_max.VEC_FIELD;                                      \
      m_out[idx].VEC_FIELD = random_to_item_t<decltype(min)>(min, max)(m_in[idx]); \
    }                                                                              \
    random_to_vec_item_t(T min, T max, float* in, T* out)                          \
        : m_min(min)                                                               \
        , m_max(max)                                                               \
        , m_in(in)                                                                 \
        , m_out(out)                                                               \
    {}                                                                             \
    T m_min;                                                                       \
    T m_max;                                                                       \
    float* m_in{};                                                                 \
    T* m_out{};                                                                    \
  }

RANDOM_TO_VEC_ITEM_SPEC(0, x);
RANDOM_TO_VEC_ITEM_SPEC(1, y);
RANDOM_TO_VEC_ITEM_SPEC(2, z);
RANDOM_TO_VEC_ITEM_SPEC(3, w);
#undef RANDOM_TO_VEC_ITEM_SPEC

generator_t::generator_t()
{
#if C2H_HAS_CURAND
  curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
#endif
}

generator_t::~generator_t()
{
#if C2H_HAS_CURAND
  curandDestroyGenerator(m_gen);
#endif
}

float* generator_t::distribution()
{
  return thrust::raw_pointer_cast(m_distribution.data());
}

void generator_t::generate()
{
#if C2H_HAS_CURAND
  curandGenerateUniform(m_gen, this->distribution(), this->m_distribution.size());
#else
  thrust::tabulate(c2h::device_policy, this->m_distribution.begin(), this->m_distribution.end(), i_to_rnd_t{m_re});
  m_re.discard(this->m_distribution.size());
#endif
}

float* generator_t::prepare_random_generator(seed_t seed, std::size_t num_items)
{
  m_distribution.resize(num_items);

#if C2H_HAS_CURAND
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());
#else
  m_re.seed(seed.get());
#endif

  generate();

  return this->distribution();
}

template <bool SetKeys>
struct random_to_custom_t
{
  static constexpr std::size_t m_max_key = std::numeric_limits<std::size_t>::max();

  __device__ void operator()(std::size_t idx)
  {
    std::size_t in = static_cast<std::size_t>(static_cast<float>(m_max_key) * m_in[idx]);

    custom_type_state_t* out = reinterpret_cast<custom_type_state_t*>(m_out + idx * m_element_size);

    if (SetKeys)
    {
      out->key = in;
    }
    else
    {
      out->val = in;
    }
  }

  random_to_custom_t(float* in, char* out, std::size_t element_size)
      : m_in(in)
      , m_out(out)
      , m_element_size(element_size)
  {}

  float* m_in{};
  char* m_out{};
  std::size_t m_element_size{};
};

template <class T>
void generator_t::operator()(seed_t seed, c2h::device_vector<T>& data, T min, T max)
{
  prepare_random_generator(seed, data.size());

  thrust::transform(
    c2h::device_policy, m_distribution.begin(), m_distribution.end(), data.begin(), random_to_item_t<T>(min, max));
}

template <typename T>
struct count_to_item_t
{
  unsigned long long int n;

  count_to_item_t(unsigned long long int n)
      : n(n)
  {}

  template <typename CounterT>
  __device__ T operator()(CounterT id)
  {
    // This has to be a type for which extended floating point types like __nv_fp8_e5m2 provide an overload
    return static_cast<T>(static_cast<float>(static_cast<unsigned long long int>(id) % n));
  }
};

template <typename T>
void generator_t::operator()(modulo_t mod, c2h::device_vector<T>& data)
{
  thrust::tabulate(c2h::device_policy, data.begin(), data.end(), count_to_item_t<T>{mod.get()});
}

generator_t& generator_t::instance()
{
  static generator_t generator;
  return generator;
}

namespace detail
{

void gen(seed_t seed,
         char* d_out,
         custom_type_state_t /* min */,
         custom_type_state_t /* max */,
         std::size_t elements,
         std::size_t element_size)
{
  thrust::counting_iterator<std::size_t> cnt_begin(0);
  thrust::counting_iterator<std::size_t> cnt_end(elements);

  generator_t& generator = generator_t::instance();
  float* d_in            = generator.prepare_random_generator(seed, elements);

  thrust::for_each(c2h::device_policy, cnt_begin, cnt_end, random_to_custom_t<true>{d_in, d_out, element_size});

  generator.generate();

  thrust::for_each(c2h::device_policy, cnt_begin, cnt_end, random_to_custom_t<false>{d_in, d_out, element_size});
}

template <class T>
struct greater_equal_op
{
  T val;

  __device__ bool operator()(T x)
  {
    return x >= val;
  }
};

template <typename T>
struct spaced_out_it_op
{
  char* base_it;
  std::size_t element_size;

  __host__ __device__ __forceinline__ T& operator()(std::size_t offset) const
  {
    return *reinterpret_cast<T*>(base_it + (element_size * offset));
  }
};

template <typename T>
struct offset_to_iterator_t
{
  char* base_it;
  std::size_t element_size;

  __host__
    __device__ __forceinline__ thrust::transform_iterator<spaced_out_it_op<T>, thrust::counting_iterator<std::size_t>>
    operator()(std::size_t offset) const
  {
    // The pointer to the beginning of this "buffer" (aka a series of same "keys")
    auto base_ptr = base_it + (element_size * offset);

    // We need to make sure that the i-th element within this "buffer" is spaced out by
    // `element_size`
    auto counting_it = thrust::make_counting_iterator(std::size_t{0});
    spaced_out_it_op<T> space_out_op{base_ptr, element_size};
    return thrust::make_transform_iterator(counting_it, space_out_op);
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

template <>
struct repeat_index_t<custom_type_state_t>
{
  __host__ __device__ __forceinline__ thrust::constant_iterator<custom_type_state_t> operator()(std::size_t i)
  {
    custom_type_state_t item{};
    item.key = i;
    item.val = i;
    return thrust::constant_iterator<custom_type_state_t>(item);
  }
};

template <typename OffsetT>
struct offset_to_size_t
{
  const OffsetT* offsets;

  __host__ __device__ __forceinline__ std::size_t operator()(std::size_t i)
  {
    return offsets[i + 1] - offsets[i];
  }
};

/**
 * @brief Initializes key-segment ranges from an offsets-array like the one given by
 * `gen_uniform_offset`.
 */
template <typename OffsetT, typename KeyT>
void init_key_segments(const c2h::device_vector<OffsetT>& segment_offsets, KeyT* d_out, std::size_t element_size)
{
  OffsetT total_segments   = static_cast<OffsetT>(segment_offsets.size() - 1);
  const OffsetT* d_offsets = thrust::raw_pointer_cast(segment_offsets.data());

  thrust::counting_iterator<int> iota(0);
  offset_to_iterator_t<KeyT> dst_transform_op{reinterpret_cast<char*>(d_out), element_size};

  auto d_range_srcs  = thrust::make_transform_iterator(iota, repeat_index_t<KeyT>{});
  auto d_range_dsts  = thrust::make_transform_iterator(d_offsets, dst_transform_op);
  auto d_range_sizes = thrust::make_transform_iterator(iota, offset_to_size_t<OffsetT>{d_offsets});

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  std::uint8_t* d_temp_storage   = nullptr;
  std::size_t temp_storage_bytes = 0;
  // TODO(bgruber): replace by a non-CUB implementation
  cub::DeviceCopy::Batched(
    d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);

  c2h::device_vector<std::uint8_t> temp_storage(temp_storage_bytes);
  d_temp_storage = thrust::raw_pointer_cast(temp_storage.data());

  // TODO(bgruber): replace by a non-CUB implementation
  cub::DeviceCopy::Batched(
    d_temp_storage, temp_storage_bytes, d_range_srcs, d_range_dsts, d_range_sizes, total_segments);
  cudaDeviceSynchronize();
#else // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
  static_assert(sizeof(OffsetT) == 0, "Need to implement a non-CUB version of cub::DeviceCopy::Batched");
  // TODO(bgruber): implement and *test* a non-CUB version, here is a sketch:
  // thrust::for_each(
  //   thrust::device,
  //   thrust::counting_iterator<OffsetT>{0},
  //   thrust::counting_iterator<OffsetT>{total_segments},
  //   [&](OffsetT i) {
  //     const auto value = d_range_srcs[i];
  //     const auto start = d_range_sizes[i];
  //     const auto end   = d_range_sizes[i + 1];
  //     for (auto j = start; j < end; ++j)
  //     {
  //       d_range_dsts[j] = value;
  //     }
  //   });
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
}

template void init_key_segments(
  const c2h::device_vector<std::uint32_t>& segment_offsets, std::int32_t* out, std::size_t element_size);
template void init_key_segments(
  const c2h::device_vector<std::uint32_t>& segment_offsets, std::uint8_t* out, std::size_t element_size);
template void
init_key_segments(const c2h::device_vector<std::uint32_t>& segment_offsets, float* out, std::size_t element_size);
template void init_key_segments(
  const c2h::device_vector<std::uint32_t>& segment_offsets, custom_type_state_t* out, std::size_t element_size);
#ifdef _CCCL_HAS_NVFP16
template void
init_key_segments(const c2h::device_vector<std::uint32_t>& segment_offsets, half_t* out, std::size_t element_size);
#endif // _CCCL_HAS_NVFP16

#ifdef _CCCL_HAS_NVBF16
template void
init_key_segments(const c2h::device_vector<std::uint32_t>& segment_offsets, bfloat16_t* out, std::size_t element_size);
#endif // _CCCL_HAS_NVBF16
} // namespace detail

template <typename T>
c2h::device_vector<T> gen_uniform_offsets(seed_t seed, T total_elements, T min_segment_size, T max_segment_size)
{
  c2h::device_vector<T> segment_offsets(total_elements + 2);
  gen(seed, segment_offsets, min_segment_size, max_segment_size);
  segment_offsets[total_elements] = total_elements + 1;
  thrust::exclusive_scan(c2h::device_policy, segment_offsets.begin(), segment_offsets.end(), segment_offsets.begin());
  typename c2h::device_vector<T>::iterator iter = thrust::find_if(
    c2h::device_policy, segment_offsets.begin(), segment_offsets.end(), detail::greater_equal_op<T>{total_elements});
  *iter = total_elements;
  segment_offsets.erase(iter + 1, segment_offsets.end());
  return segment_offsets;
}

template c2h::device_vector<int32_t>
gen_uniform_offsets(seed_t seed, int32_t total_elements, int32_t min_segment_size, int32_t max_segment_size);

template c2h::device_vector<uint32_t>
gen_uniform_offsets(seed_t seed, uint32_t total_elements, uint32_t min_segment_size, uint32_t max_segment_size);
template c2h::device_vector<int64_t>
gen_uniform_offsets(seed_t seed, int64_t total_elements, int64_t min_segment_size, int64_t max_segment_size);
template c2h::device_vector<uint64_t>
gen_uniform_offsets(seed_t seed, uint64_t total_elements, uint64_t min_segment_size, uint64_t max_segment_size);

template <typename T>
void gen(seed_t seed, c2h::device_vector<T>& data, T min, T max)
{
  generator_t::instance()(seed, data, min, max);
}

template <typename T>
void gen(modulo_t mod, c2h::device_vector<T>& data)
{
  generator_t::instance()(mod, data);
}

#define INSTANTIATE_RND(TYPE) template void gen<TYPE>(seed_t, c2h::device_vector<TYPE> & data, TYPE min, TYPE max)
#define INSTANTIATE_MOD(TYPE) template void gen<TYPE>(modulo_t, c2h::device_vector<TYPE> & data)

#define INSTANTIATE(TYPE) \
  INSTANTIATE_RND(TYPE);  \
  INSTANTIATE_MOD(TYPE)

INSTANTIATE(std::uint8_t);
INSTANTIATE(std::uint16_t);
INSTANTIATE(std::uint32_t);
INSTANTIATE(std::uint64_t);

INSTANTIATE(std::int8_t);
INSTANTIATE(std::int16_t);
INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);

#if defined(__CUDA_FP8_TYPES_EXIST__)
INSTANTIATE(__nv_fp8_e5m2);
INSTANTIATE(__nv_fp8_e4m3);
#endif // defined(__CUDA_FP8_TYPES_EXIST__)
INSTANTIATE(float);
INSTANTIATE(double);

INSTANTIATE(bool);
INSTANTIATE(char);

#ifdef _CCCL_HAS_NVFP16
INSTANTIATE(half_t);
INSTANTIATE(__half);
#endif // _CCCL_HAS_NVFP16

#ifdef _CCCL_HAS_NVBF16
INSTANTIATE(bfloat16_t);
INSTANTIATE(__nv_bfloat16);
#endif // _CCCL_HAS_NVBF16

#undef INSTANTIATE_RND
#undef INSTANTIATE_MOD
#undef INSTANTIATE

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T, int VecItem>
struct vec_gen_helper_t;

template <typename T>
struct vec_gen_helper_t<T, -1>
{
  static void gen(c2h::device_vector<T>&, T, T) {}
};

template <typename T, int VecItem>
struct vec_gen_helper_t
{
  static void gen(c2h::device_vector<T>& data, T min, T max)
  {
    thrust::counting_iterator<std::size_t> cnt_begin(0);
    thrust::counting_iterator<std::size_t> cnt_end(data.size());

    generator_t& generator = generator_t::instance();
    float* d_in            = generator.distribution();
    T* d_out               = thrust::raw_pointer_cast(data.data());

    generator.generate();

    thrust::for_each(c2h::device_policy, cnt_begin, cnt_end, random_to_vec_item_t<T, VecItem>{min, max, d_in, d_out});

    vec_gen_helper_t<T, VecItem - 1>::gen(data, min, max);
  }
};

#  define VEC_SPECIALIZATION(TYPE, SIZE)                                                                     \
    template <>                                                                                              \
    void gen<TYPE##SIZE>(seed_t seed, c2h::device_vector<TYPE##SIZE> & data, TYPE##SIZE min, TYPE##SIZE max) \
    {                                                                                                        \
      generator_t& generator = generator_t::instance();                                                      \
      generator.prepare_random_generator(seed, data.size());                                                 \
      vec_gen_helper_t<TYPE##SIZE, SIZE - 1>::gen(data, min, max);                                           \
    }

VEC_SPECIALIZATION(char, 2);
VEC_SPECIALIZATION(char, 3);
VEC_SPECIALIZATION(char, 4);

// VEC_SPECIALIZATION(uchar, 2);
VEC_SPECIALIZATION(uchar, 3);
// VEC_SPECIALIZATION(uchar, 4);

VEC_SPECIALIZATION(short, 2);
VEC_SPECIALIZATION(short, 3);
VEC_SPECIALIZATION(short, 4);

// VEC_SPECIALIZATION(ushort, 2);
// VEC_SPECIALIZATION(ushort, 3);
// VEC_SPECIALIZATION(ushort, 4);

VEC_SPECIALIZATION(int, 2);
VEC_SPECIALIZATION(int, 3);
VEC_SPECIALIZATION(int, 4);

// VEC_SPECIALIZATION(uint, 2);
// VEC_SPECIALIZATION(uint, 3);
// VEC_SPECIALIZATION(uint, 4);

VEC_SPECIALIZATION(long, 2);
VEC_SPECIALIZATION(long, 3);
VEC_SPECIALIZATION(long, 4);

// VEC_SPECIALIZATION(ulong, 2);
// VEC_SPECIALIZATION(ulong, 3);
// VEC_SPECIALIZATION(ulong, 4);

VEC_SPECIALIZATION(longlong, 2);
VEC_SPECIALIZATION(longlong, 3);
VEC_SPECIALIZATION(longlong, 4);

VEC_SPECIALIZATION(ulonglong, 2);
// VEC_SPECIALIZATION(ulonglong, 3);
VEC_SPECIALIZATION(ulonglong, 4);

VEC_SPECIALIZATION(float, 2);
VEC_SPECIALIZATION(float, 3);
VEC_SPECIALIZATION(float, 4);

VEC_SPECIALIZATION(double, 2);
VEC_SPECIALIZATION(double, 3);
VEC_SPECIALIZATION(double, 4);

template <typename VecType, typename Type>
struct vec_gen_t
{
  std::size_t n;
  scalar_to_vec_t<VecType> convert;

  vec_gen_t(std::size_t n)
      : n(n)
  {}

  template <typename CounterT>
  __device__ VecType operator()(CounterT id)
  {
    return convert(static_cast<Type>(id) % n);
  }
};

#  define VEC_GEN_MOD_SPECIALIZATION(VEC_TYPE, SCALAR_TYPE)                                                        \
    template <>                                                                                                    \
    void gen<VEC_TYPE>(modulo_t mod, c2h::device_vector<VEC_TYPE> & data)                                          \
    {                                                                                                              \
      thrust::tabulate(c2h::device_policy, data.begin(), data.end(), vec_gen_t<VEC_TYPE, SCALAR_TYPE>{mod.get()}); \
    }

VEC_GEN_MOD_SPECIALIZATION(short2, short);
VEC_GEN_MOD_SPECIALIZATION(uchar3, unsigned char);
VEC_GEN_MOD_SPECIALIZATION(ulonglong4, unsigned long long);
VEC_GEN_MOD_SPECIALIZATION(ushort4, unsigned short);
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace c2h
