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

template <typename T, int VecSize>
struct random_to_vec_item_t
{
  __device__ void operator()(std::size_t idx)
  {
#define SET_FIELD(VEC_FIELD) \
  m_out[idx].VEC_FIELD = random_to_item_t<decltype(m_min.VEC_FIELD)>(m_min.VEC_FIELD, m_max.VEC_FIELD)(m_in[idx]);

    if constexpr (VecSize >= 4)
    {
      SET_FIELD(w);
    }
    if constexpr (VecSize >= 3)
    {
      SET_FIELD(z);
    }
    if constexpr (VecSize >= 2)
    {
      SET_FIELD(y);
    }
    if constexpr (VecSize >= 1)
    {
      SET_FIELD(x);
    }
#undef SET_FIELD
  }

  T m_min;
  T m_max;
  float* m_in{};
  T* m_out{};
};

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
template <typename T, int VecSize>
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

    thrust::for_each(c2h::device_policy, cnt_begin, cnt_end, random_to_vec_item_t<T, VecSize>{min, max, d_in, d_out});
  }
};

#  define VEC_SPECIALIZATION(TYPE, SIZE, ...)                               \
    template <>                                                             \
    void gen<TYPE##SIZE##__VA_ARGS__>(                                      \
      seed_t seed,                                                          \
      c2h::device_vector<TYPE##SIZE##__VA_ARGS__> & data,                   \
      TYPE##SIZE##__VA_ARGS__ min,                                          \
      TYPE##SIZE##__VA_ARGS__ max)                                          \
    {                                                                       \
      generator_t& generator = generator_t::instance();                     \
      generator.prepare_random_generator(seed, data.size());                \
      vec_gen_helper_t<TYPE##SIZE##__VA_ARGS__, SIZE>::gen(data, min, max); \
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
_CCCL_SUPPRESS_DEPRECATED_PUSH
VEC_SPECIALIZATION(long, 4);
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(long, 4, _16a);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

// VEC_SPECIALIZATION(ulong, 2);
// VEC_SPECIALIZATION(ulong, 3);
// VEC_SPECIALIZATION(ulong, 4);

VEC_SPECIALIZATION(longlong, 2);
VEC_SPECIALIZATION(longlong, 3);
_CCCL_SUPPRESS_DEPRECATED_PUSH
VEC_SPECIALIZATION(longlong, 4);
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(longlong, 4, _16a);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

VEC_SPECIALIZATION(ulonglong, 2);
// VEC_SPECIALIZATION(ulonglong, 3);
_CCCL_SUPPRESS_DEPRECATED_PUSH
VEC_SPECIALIZATION(ulonglong, 4);
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(ulonglong, 4, _16a);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

VEC_SPECIALIZATION(float, 2);
VEC_SPECIALIZATION(float, 3);
VEC_SPECIALIZATION(float, 4);

VEC_SPECIALIZATION(double, 2);
VEC_SPECIALIZATION(double, 3);
_CCCL_SUPPRESS_DEPRECATED_PUSH
VEC_SPECIALIZATION(double, 4);
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(double, 4, _16a);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

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
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_GEN_MOD_SPECIALIZATION(ulonglong4_16a, unsigned long long);
#  else
VEC_GEN_MOD_SPECIALIZATION(ulonglong4, unsigned long long);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
VEC_GEN_MOD_SPECIALIZATION(ushort4, unsigned short);
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace c2h
