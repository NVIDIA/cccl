// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/tabulate.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>

#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/fill_striped.h>
#include <c2h/generators.cuh>
#include <c2h/generators.h>
#include <c2h/vector.h>

namespace c2h
{

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
    thrust::counting_iterator<size_t> cnt_begin(0);
    thrust::counting_iterator<size_t> cnt_end(data.size());

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

VEC_SPECIALIZATION(ushort, 2);

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

#  if TEST_HALF_T()
VEC_SPECIALIZATION(__half, 2);
#  endif // TEST_HALF_T()
#  if TEST_BF_T()
VEC_SPECIALIZATION(__nv_bfloat16, 2);
#  endif // TEST_BF_T()

template <typename VecType, typename Type>
struct vec_gen_t
{
  size_t n;
  scalar_to_vec_t<VecType> convert;

  vec_gen_t(size_t n)
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
