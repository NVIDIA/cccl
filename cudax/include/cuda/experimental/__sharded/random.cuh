//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Sharding-invariant random fill over sharded views (cuRAND tier).
 *
 * `generate_uniform(data, envs, seed)` / `generate_normal(...)` fill a
 * sharded view so that the result is BITWISE IDENTICAL to a whole-array
 * generation with the same seed, for any sharding: resharding never changes
 * the data. This is the reproducibility contract sharded workloads need
 * (reshard-stable experiments, checkpoint-stable initialization); fill is
 * placement-neutral by expectation, so the value is correctness, not
 * bandwidth.
 *
 * Contract: counter-based Philox (`CURAND_RNG_PSEUDO_PHILOX4_32_10`, draw k
 * is a pure function of (seed, k)), non-DYNAMIC ordering, and per shard the
 * generation starts at the shard's global element index. Poisson is excluded
 * by design: its data-dependent draw count breaks the position mapping.
 *
 * Paths per element type:
 *  - float: the host API with `curandSetGeneratorOffset(global_offset)` —
 *    bitwise identical to a stock whole-array `curandGenerateUniform/Normal`.
 *  - double: a per-element device-API kernel (`curand_init` with
 *    subsequence = the global element index), position-pure by construction.
 *    WORKAROUND: the host API's FP64 offset positioning measured inconsistent
 *    at large offsets on current toolkits, so this path deliberately selects
 *    the device-API mapping, which is exact at every scale. The
 *    double-precision sequence of THIS interface is therefore defined by that
 *    mapping (not bitwise-comparable to a stock host-API whole-array run);
 *    the invariance test is the gate to re-run per toolkit.
 *
 * Synchronous convenience contract (the `__generic_map` no-stream form):
 * refuses under CUDA graph capture and under `sync_policy::forbid`, joins
 * every lane before returning.
 *
 * Vendor tier: opt-in header (not part of `<cuda/experimental/sharded.cuh>`);
 * consumers link `CUDA::curand`.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__sharded/concepts.cuh>
#include <cuda/experimental/__sharded/cuda_safe_call.cuh>
#include <cuda/experimental/__sharded/stream_scope.cuh>

#include <stdexcept>
#include <string>
#include <vector>

#include <curand.h>
#include <curand_kernel.h>

namespace cuda::experimental::sharded
{
namespace reserved
{
inline void __curand_check(curandStatus_t __s, const char* __what)
{
  if (__s != CURAND_STATUS_SUCCESS)
  {
    _CCCL_THROW(::std::runtime_error,
                ::std::string(__what) + ": cuRAND error " + ::std::to_string(static_cast<int>(__s)));
  }
}

//! Owns the per-shard host-API generators of one call. Destroyed only after
//! the call's join: generation kernels must not outlive their generator.
class __philox_set
{
public:
  //! Create a Philox generator for the CURRENT device, seeded and positioned
  //! per the sharding-invariance contract, bound to @p __stream.
  curandGenerator_t __make(unsigned long long __seed, unsigned long long __offset, cudaStream_t __stream)
  {
    curandGenerator_t __g = nullptr;
    __curand_check(curandCreateGenerator(&__g, CURAND_RNG_PSEUDO_PHILOX4_32_10), "create");
    __gens_.push_back(__g); // owned from this point
    __curand_check(curandSetGeneratorOrdering(__g, CURAND_ORDERING_PSEUDO_DEFAULT), "ordering");
    __curand_check(curandSetPseudoRandomGeneratorSeed(__g, __seed), "seed");
    __curand_check(curandSetGeneratorOffset(__g, __offset), "offset");
    __curand_check(curandSetStream(__g, __stream), "stream");
    return __g;
  }

  ~__philox_set()
  {
    for (curandGenerator_t __g : __gens_)
    {
      (void) curandDestroyGenerator(__g);
    }
  }

private:
  ::std::vector<curandGenerator_t> __gens_;
};

//! Position-pure FP64 fill: element gi is a pure function of (seed, gi) via a
//! per-element Philox subsequence (see the file-level path note).
template <bool _IsNormal>
__global__ void __positional_fill_double(
  double* __out,
  unsigned long long __n,
  unsigned long long __global_offset,
  unsigned long long __seed,
  double __mean,
  double __stddev)
{
  const unsigned long long __stride = static_cast<unsigned long long>(gridDim.x) * blockDim.x;
  for (unsigned long long __i = static_cast<unsigned long long>(blockIdx.x) * blockDim.x + threadIdx.x; __i < __n;
       __i += __stride)
  {
    curandStatePhilox4_32_10_t __st;
    curand_init(__seed, __global_offset + __i, 0, &__st);
    if constexpr (_IsNormal)
    {
      __out[__i] = __mean + __stddev * curand_normal_double(&__st);
    }
    else
    {
      __out[__i] = curand_uniform_double(&__st);
    }
  }
}

template <bool _IsNormal>
inline void __launch_positional_fill_double(
  double* __out,
  unsigned long long __n,
  unsigned long long __global_offset,
  unsigned long long __seed,
  double __mean,
  double __stddev,
  cudaStream_t __stream)
{
  constexpr unsigned __threads = 256;
  const unsigned __blocks =
    static_cast<unsigned>(::cuda::std::min<unsigned long long>((__n + __threads - 1) / __threads, 65535));
  __positional_fill_double<_IsNormal>
    <<<__blocks, __threads, 0, __stream>>>(__out, __n, __global_offset, __seed, __mean, __stddev);
  cuda_safe_call(cudaGetLastError());
}

template <class _Tp>
inline curandStatus_t __host_generate_uniform(curandGenerator_t __g, _Tp* __p, ::std::size_t __n)
{
  if constexpr (::cuda::std::is_same_v<_Tp, float>)
  {
    return curandGenerateUniform(__g, __p, __n);
  }
  else
  {
    return curandGenerateUniformDouble(__g, __p, __n);
  }
}

template <class _Tp>
inline curandStatus_t
__host_generate_normal(curandGenerator_t __g, _Tp* __p, ::std::size_t __n, _Tp __mean, _Tp __stddev)
{
  if constexpr (::cuda::std::is_same_v<_Tp, float>)
  {
    return curandGenerateNormal(__g, __p, __n, __mean, __stddev);
  }
  else
  {
    return curandGenerateNormalDouble(__g, __p, __n, __mean, __stddev);
  }
}
} // namespace reserved

//! @brief Sharding-invariant uniform fill: bitwise identical for any
//! sharding of the same index space (see the file-level contract).
template <class _S, class _Envs>
void generate_uniform(_S&& __data, const _Envs& __envs, unsigned long long __seed)
{
  using _Tp = view_element_t<_S>;
  static_assert(::cuda::std::is_same_v<_Tp, float> || ::cuda::std::is_same_v<_Tp, double>,
                "generate_uniform: float/double elements only");
  reserved::__philox_set __gens;
  __detail::__generic_map(
    __data, __envs, default_call_env{}, "sharded::generate_uniform", [&](const auto& __d, cudaStream_t __stream) {
      if constexpr (::cuda::std::is_same_v<_Tp, double>)
      {
        reserved::__launch_positional_fill_double<false>(
          __d.data, __d.size, static_cast<unsigned long long>(__d.global_offset), __seed, 0.0, 0.0, __stream);
      }
      else
      {
        curandGenerator_t __g = __gens.__make(__seed, static_cast<unsigned long long>(__d.global_offset), __stream);
        reserved::__curand_check(
          reserved::__host_generate_uniform<_Tp>(__g, __d.data, static_cast<::std::size_t>(__d.size)),
          "generate_uniform");
      }
    });
  // __generic_map's synchronous contract joined every lane: generators may die.
}

//! @brief Sharding-invariant normal fill (same contract).
template <class _S, class _Envs>
void generate_normal(
  _S&& __data, const _Envs& __envs, unsigned long long __seed, view_element_t<_S> __mean, view_element_t<_S> __stddev)
{
  using _Tp = view_element_t<_S>;
  static_assert(::cuda::std::is_same_v<_Tp, float> || ::cuda::std::is_same_v<_Tp, double>,
                "generate_normal: float/double elements only");
  reserved::__philox_set __gens;
  __detail::__generic_map(
    __data, __envs, default_call_env{}, "sharded::generate_normal", [&](const auto& __d, cudaStream_t __stream) {
      if constexpr (::cuda::std::is_same_v<_Tp, double>)
      {
        reserved::__launch_positional_fill_double<true>(
          __d.data, __d.size, static_cast<unsigned long long>(__d.global_offset), __seed, __mean, __stddev, __stream);
      }
      else
      {
        curandGenerator_t __g = __gens.__make(__seed, static_cast<unsigned long long>(__d.global_offset), __stream);
        reserved::__curand_check(
          reserved::__host_generate_normal<_Tp>(__g, __d.data, static_cast<::std::size_t>(__d.size), __mean, __stddev),
          "generate_normal");
      }
    });
}
} // namespace cuda::experimental::sharded
