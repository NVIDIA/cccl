// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/tabulate.h>

#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/stream>

#include <c2h/detail/generators.cuh>
#include <c2h/detail/scoped_current_device.cuh>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/fill_striped.h>
#include <c2h/generators.h>
#include <c2h/vector.h>

namespace c2h::detail
{
template <typename T, int VecSize>
struct random_to_vec_item_t
{
  unsigned long long m_seed;
  T m_min;
  T m_max;

  __device__ T operator()(std::size_t idx)
  {
    T out{};
    index_to_random_uniform rng{m_seed};
    int field = 0;
#define SET_FIELD(VEC_FIELD) \
  out.VEC_FIELD =            \
    random_to_item_t<decltype(m_min.VEC_FIELD)>(m_min.VEC_FIELD, m_max.VEC_FIELD)(rng(idx * VecSize + field++));

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
    return out;
  }
};

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
#  define VEC_SPECIALIZATION(T)                                                                              \
    template <>                                                                                              \
    void gen_values_between(seed_t seed, ::cuda::std::span<T> data, T min, T max)                            \
    {                                                                                                        \
      thrust::tabulate(device_policy,                                                                        \
                       data.begin(),                                                                         \
                       data.end(),                                                                           \
                       random_to_vec_item_t<T, ::cuda::std::tuple_size_v<T>>{seed.get(), min, max});         \
    }                                                                                                        \
    template <>                                                                                              \
    void gen_values_between(::cuda::stream_ref stream, seed_t seed, ::cuda::std::span<T> data, T min, T max) \
    {                                                                                                        \
      const scoped_current_device device_scope{stream.device().get()};                                       \
      thrust::tabulate(device_policy.on(stream.get()),                                                       \
                       data.begin(),                                                                         \
                       data.end(),                                                                           \
                       random_to_vec_item_t<T, ::cuda::std::tuple_size_v<T>>{seed.get(), min, max});         \
    }

VEC_SPECIALIZATION(char2);
VEC_SPECIALIZATION(char3);
VEC_SPECIALIZATION(char4);

// VEC_SPECIALIZATION(uchar2);
VEC_SPECIALIZATION(uchar3);
// VEC_SPECIALIZATION(uchar4);

VEC_SPECIALIZATION(short2);
VEC_SPECIALIZATION(short3);
VEC_SPECIALIZATION(short4);

VEC_SPECIALIZATION(ushort2);

VEC_SPECIALIZATION(int2);
VEC_SPECIALIZATION(int3);
VEC_SPECIALIZATION(int4);

// VEC_SPECIALIZATION(uint2);
// VEC_SPECIALIZATION(uint3);
// VEC_SPECIALIZATION(uint4);

VEC_SPECIALIZATION(long2);
VEC_SPECIALIZATION(long3);
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(long4_16a);
VEC_SPECIALIZATION(long4_32a);
#  else
VEC_SPECIALIZATION(long4);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

// VEC_SPECIALIZATION(ulong2);
// VEC_SPECIALIZATION(ulong3);
// VEC_SPECIALIZATION(ulong4);

VEC_SPECIALIZATION(longlong2);
VEC_SPECIALIZATION(longlong3);
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(longlong4_16a);
VEC_SPECIALIZATION(longlong4_32a);
#  else
VEC_SPECIALIZATION(longlong4);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

VEC_SPECIALIZATION(ulonglong2);
// VEC_SPECIALIZATION(ulonglong3);
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(ulonglong4_16a);
VEC_SPECIALIZATION(ulonglong4_32a);
#  else
VEC_SPECIALIZATION(ulonglong4);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

VEC_SPECIALIZATION(float2);
VEC_SPECIALIZATION(float3);
VEC_SPECIALIZATION(float4);

VEC_SPECIALIZATION(double2);
VEC_SPECIALIZATION(double3);
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_SPECIALIZATION(double4_16a);
VEC_SPECIALIZATION(double4_32a);
#  else
VEC_SPECIALIZATION(double4);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

#  if CCCL_VERSION > 3001000
#    if TEST_HALF_T()
VEC_SPECIALIZATION(__half2);
#    endif // TEST_HALF_T()
#    if TEST_BF_T()
VEC_SPECIALIZATION(__nv_bfloat162);
#    endif // TEST_BF_T()
#  endif // CCCL_VERSION > 3001000

template <typename VecType, typename Type>
struct counter_to_cyclic_vector_t
{
  std::size_t n;

  template <typename CounterT>
  __device__ VecType operator()(CounterT id) const
  {
    return scalar_to_vec_t<VecType>{}(static_cast<Type>(id) % n);
  }
};

#  define VEC_GEN_MOD_SPECIALIZATION(VEC_TYPE, SCALAR_TYPE)                                                     \
    template <>                                                                                                 \
    void gen_values_cyclic<VEC_TYPE>(modulo_t mod, ::cuda::std::span<VEC_TYPE> data)                            \
    {                                                                                                           \
      thrust::tabulate(                                                                                         \
        device_policy, data.begin(), data.end(), counter_to_cyclic_vector_t<VEC_TYPE, SCALAR_TYPE>{mod.get()}); \
    }

VEC_GEN_MOD_SPECIALIZATION(short2, short);
VEC_GEN_MOD_SPECIALIZATION(uchar3, unsigned char);
#  if _CCCL_CTK_AT_LEAST(13, 0)
VEC_GEN_MOD_SPECIALIZATION(ulonglong4_16a, unsigned long long);
VEC_GEN_MOD_SPECIALIZATION(ulonglong4_32a, unsigned long long);
#  else
VEC_GEN_MOD_SPECIALIZATION(ulonglong4, unsigned long long);
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
VEC_GEN_MOD_SPECIALIZATION(ushort4, unsigned short);
#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
} // namespace c2h::detail
