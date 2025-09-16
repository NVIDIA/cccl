// SPDX-FileCopyrightText: Copyright (c) 2011-2025, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3-Clause

#include <thrust/tabulate.h>
#include <thrust/transform.h>

#include <c2h/bfloat16.cuh>
#include <c2h/detail/generators.cuh>
#include <c2h/device_policy.h>
#include <c2h/extended_types.h>
#include <c2h/generators.h>
#include <c2h/half.cuh>

namespace c2h::detail
{
template <typename T>
void gen_values_between(seed_t seed, ::cuda::std::span<T> data, T min, T max)
{
  const auto* dist = generator.prepare_random_generator(seed, data.size());
  thrust::transform(device_policy, dist, dist + data.size(), data.begin(), random_to_item_t<T>(min, max));
}

template <typename T>
struct counter_to_cyclic_item_t
{
  std::size_t n;

  template <typename CounterT>
  __device__ T operator()(CounterT id)
  {
    // This has to be a type for which extended floating point types like __nv_fp8_e5m2 provide an overload
    return static_cast<T>(static_cast<float>(static_cast<uint64_t>(id) % n));
  }
};

template <typename T>
void gen_values_cyclic(modulo_t mod, ::cuda::std::span<T> data)
{
  thrust::tabulate(device_policy, data.begin(), data.end(), counter_to_cyclic_item_t<T>{mod.get()});
}

#define INSTANTIATE_RND(TYPE) \
  template void gen_values_between<TYPE>(seed_t, ::cuda::std::span<TYPE> data, TYPE min, TYPE max)
#define INSTANTIATE_MOD(TYPE) template void gen_values_cyclic<TYPE>(modulo_t, ::cuda::std::span<TYPE> data)

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

#if _CCCL_HAS_NVFP8()
INSTANTIATE(__nv_fp8_e5m2);
INSTANTIATE(__nv_fp8_e4m3);
#endif // _CCCL_HAS_NVFP8()
INSTANTIATE(float);
INSTANTIATE(double);
INSTANTIATE(cuda::std::complex<float>);
INSTANTIATE(cuda::std::complex<double>);

INSTANTIATE(bool);
INSTANTIATE(char);

#if TEST_HALF_T()
INSTANTIATE(half_t);
INSTANTIATE(__half);
#  if _CCCL_CUDACC_AT_LEAST(12, 2)
INSTANTIATE(cuda::std::complex<__half>);
#  endif
#endif // TEST_HALF_T()

#if TEST_BF_T()
INSTANTIATE(bfloat16_t);
INSTANTIATE(__nv_bfloat16);
#  if _CCCL_CUDACC_AT_LEAST(12, 2)
INSTANTIATE(cuda::std::complex<__nv_bfloat16>);
#  endif
#endif // TEST_BF_T()

#if TEST_INT128()
INSTANTIATE(__int128_t);
INSTANTIATE(__uint128_t);
#endif // TEST_INT128()

#undef INSTANTIATE_RND
#undef INSTANTIATE_MOD
#undef INSTANTIATE
} // namespace c2h::detail
