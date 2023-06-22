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

#include <thrust/distance.h>
#include <thrust/execution_policy.h>
#include <thrust/for_each.h>
#include <thrust/tabulate.h>

#include <cstdint>

#include <c2h/custom_type.cuh>
#include <c2h/generators.cuh>
#include <curand.h>
#include <fill_striped.cuh>

namespace c2h
{

class generator_t
{
private:
  generator_t();

public:

  static generator_t &instance();
  ~generator_t();

  template <typename T>
  void operator()(seed_t seed,
                  thrust::device_vector<T> &data,
                  T min = std::numeric_limits<T>::min(),
                  T max = std::numeric_limits<T>::max());

  template <typename T>
  void operator()(modulo_t modulo,
                  thrust::device_vector<T> &data);

  float* distribution();
  curandGenerator_t &gen() { return m_gen; }

  float* prepare_random_generator(
      seed_t seed,
      std::size_t num_items);

private:
  curandGenerator_t m_gen;
  thrust::device_vector<float> m_distribution;
};


template <typename T>
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

template <typename T, int VecItem>
struct random_to_vec_item_t;

#define RANDOM_TO_VEC_ITEM_SPEC(VEC_ITEM, VEC_FIELD)                           \
  template <typename T>                                                        \
  struct random_to_vec_item_t<T, VEC_ITEM>                                     \
  {                                                                            \
    __device__ void operator()(std::size_t idx)                                \
    {                                                                          \
      auto min             = m_min.VEC_FIELD;                                  \
      auto max             = m_max.VEC_FIELD;                                  \
      m_out[idx].VEC_FIELD = random_to_item_t<decltype(min)>(min,              \
                                                             max)(m_in[idx]);  \
    }                                                                          \
    random_to_vec_item_t(T min, T max, float *in, T *out)                      \
        : m_min(min)                                                           \
        , m_max(max)                                                           \
        , m_in(in)                                                             \
        , m_out(out)                                                           \
    {}                                                                         \
    T m_min;                                                                   \
    T m_max;                                                                   \
    float *m_in{};                                                             \
    T *m_out{};                                                                \
  }

RANDOM_TO_VEC_ITEM_SPEC(0, x);
RANDOM_TO_VEC_ITEM_SPEC(1, y);
RANDOM_TO_VEC_ITEM_SPEC(2, z);
RANDOM_TO_VEC_ITEM_SPEC(3, w);

generator_t::generator_t()
{
  curandCreateGenerator(&m_gen, CURAND_RNG_PSEUDO_DEFAULT);
}

generator_t::~generator_t()
{
  curandDestroyGenerator(m_gen);
}

float* generator_t::distribution()
{
  return thrust::raw_pointer_cast(m_distribution.data());
}

float *generator_t::prepare_random_generator(seed_t seed, 
                                             std::size_t num_items)
{
  curandSetPseudoRandomGeneratorSeed(m_gen, seed.get());

  m_distribution.resize(num_items);
  curandGenerateUniform(m_gen,
                        this->distribution(),
                        num_items);

  return this->distribution();
}

template <bool SetKeys> 
struct random_to_custom_t
{
  static constexpr std::size_t m_max_key = 
    std::numeric_limits<std::size_t>::max();

  __device__ void operator()(std::size_t idx)
  {
    std::size_t in = 
      static_cast<std::size_t>(static_cast<float>(m_max_key) * m_in[idx]);

    custom_type_state_t* out = 
      reinterpret_cast<custom_type_state_t*>(m_out + idx * m_element_size);

    if (SetKeys)
    {
      out->key = in;
    }
    else 
    {
      out->val = in;
    }
  }

  random_to_custom_t(
      float *in, 
      char *out,
      std::size_t element_size)
    : m_in(in)
    , m_out(out)
    , m_element_size(element_size)
  {}

  float *m_in{};
  char *m_out{};
  std::size_t m_element_size{};
};

template <class T>
void generator_t::operator()(seed_t seed,
                             thrust::device_vector<T> &data,
                             T min,
                             T max)
{
  prepare_random_generator(seed, data.size());

  thrust::transform(m_distribution.begin(),
                    m_distribution.end(),
                    data.begin(),
                    random_to_item_t<T>(min, max));
}

template <typename T>
struct count_to_item_t
{
  std::size_t n;

  count_to_item_t(std::size_t n)
    : n(n)
  {}

  template <typename CounterT>
  __device__ T operator()(CounterT id)
  {
    return static_cast<T>(static_cast<std::size_t>(id) % n);
  }
};

template <typename T>
void generator_t::operator()(modulo_t mod,
                             thrust::device_vector<T> &data)
{
  thrust::tabulate(data.begin(), data.end(), count_to_item_t<T>{mod.get()});
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
  float *d_in = generator.prepare_random_generator(seed, elements);

  thrust::for_each(
    thrust::device,
    cnt_begin,
    cnt_end,
    random_to_custom_t<true>{d_in, d_out, element_size});

  curandGenerateUniform(generator.gen(),
                        generator.distribution(),
                        elements);

  thrust::for_each(
    thrust::device,
    cnt_begin,
    cnt_end,
    random_to_custom_t<false>{d_in, d_out, element_size});
}

}


template <typename T>
void gen(seed_t seed, 
         thrust::device_vector<T> &data,
         T min,
         T max)
{
  generator_t::instance()(seed, data, min, max);
}

template <typename T>
void gen(modulo_t mod, 
         thrust::device_vector<T> &data)
{
  generator_t::instance()(mod, data);
}

#define INSTANTIATE_RND(TYPE) \
template \
void gen<TYPE>( \
    seed_t, \
    thrust::device_vector<TYPE> &data, \
    TYPE min, \
    TYPE max)

#define INSTANTIATE_MOD(TYPE) \
template \
void gen<TYPE>( \
    modulo_t, \
    thrust::device_vector<TYPE> &data)

#define INSTANTIATE(TYPE) \
  INSTANTIATE_RND(TYPE); \
  INSTANTIATE_MOD(TYPE)

INSTANTIATE(std::uint8_t);
INSTANTIATE(std::uint16_t);
INSTANTIATE(std::uint32_t);
INSTANTIATE(std::uint64_t);

INSTANTIATE(std::int8_t);
INSTANTIATE(std::int16_t);
INSTANTIATE(std::int32_t);
INSTANTIATE(std::int64_t);

INSTANTIATE(float);
INSTANTIATE(double);

template <typename T, int VecItem>
struct vec_gen_helper_t;

template <typename T>
struct vec_gen_helper_t<T, -1>
{
  static void gen(thrust::device_vector<T> &, T , T )
  {
  }
};

template <typename T, int VecItem>
struct vec_gen_helper_t
{
  static void gen(thrust::device_vector<T> &data,
                  T min,
                  T max)
  {
    thrust::counting_iterator<std::size_t> cnt_begin(0);
    thrust::counting_iterator<std::size_t> cnt_end(data.size());

    generator_t& generator = generator_t::instance();
    float *d_in = generator.distribution();
    T *d_out = thrust::raw_pointer_cast(data.data());

    curandGenerateUniform(generator.gen(), d_in, data.size());

    thrust::for_each(
      thrust::device,
      cnt_begin,
      cnt_end,
      random_to_vec_item_t<T, VecItem>{min, max, d_in, d_out});

    vec_gen_helper_t<T, VecItem - 1>::gen(data, min, max);
  }
};


#define VEC_SPECIALIZATION(TYPE, SIZE) \
template<> void gen<TYPE##SIZE>(seed_t seed, \
                                thrust::device_vector<TYPE##SIZE> &data, \
                                TYPE##SIZE min, \
                                TYPE##SIZE max) \
{ \
  generator_t& generator = generator_t::instance(); \
  generator.prepare_random_generator(seed, data.size()); \
  vec_gen_helper_t<TYPE##SIZE, SIZE - 1>::gen(data, min, max); \
}

VEC_SPECIALIZATION(int, 2);
VEC_SPECIALIZATION(long, 2);
VEC_SPECIALIZATION(longlong, 2);
VEC_SPECIALIZATION(longlong, 4);

VEC_SPECIALIZATION(char, 2);
VEC_SPECIALIZATION(char, 4);

VEC_SPECIALIZATION(short, 2);

VEC_SPECIALIZATION(double, 2);

VEC_SPECIALIZATION(uchar, 3);

VEC_SPECIALIZATION(ulonglong, 4);

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

#define VEC_GEN_MOD_SPECIALIZATION(VEC_TYPE, SCALAR_TYPE)                                          \
  template <>                                                                                      \
  void gen<VEC_TYPE>(modulo_t mod, thrust::device_vector<VEC_TYPE> & data)                         \
  {                                                                                                \
    thrust::tabulate(data.begin(), data.end(), vec_gen_t<VEC_TYPE, SCALAR_TYPE>{mod.get()});       \
  }

VEC_GEN_MOD_SPECIALIZATION(short2, short);

VEC_GEN_MOD_SPECIALIZATION(uchar3, unsigned char);

VEC_GEN_MOD_SPECIALIZATION(ulonglong4, unsigned long long);

VEC_GEN_MOD_SPECIALIZATION(ushort4, unsigned short);

} // c2h

