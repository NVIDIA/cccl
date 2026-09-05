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
 *    Normal generation honors the documented even-count requirement at any
 *    shard boundary by decomposing into an even-aligned main run plus
 *    boundary elements from 2-element runs (same global sequence).
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

#include <functional>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
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

//! Process-wide cache of host-API Philox generators, one per (device, stream).
//!
//! Creating a host-API generator is cheap, but its FIRST generate call pays
//! a lazy state setup (0.2-4 ms, host-blocking) and `curandDestroyGenerator`
//! is an implicit device sync (0.2-1 ms). Done per shard per call, as the
//! first version of this tier did, those two serialize on the host and grow
//! linearly with the shard count: on a 4x GB200 node a 2^26 float fill went
//! from 0.85 ms at one place to 2.5 ms at four. Seed and offset, by contrast,
//! are host-side state applied at the next generate — free to reset per call.
//! So generators are created once per (device, stream) and reused; the
//! sharding-invariance contract only depends on (seed, offset), never on the
//! generator's history.
//!
//! Keyed by stream as well as device because a generator's offset is per
//! object: two shards generating concurrently on different streams of one
//! device need distinct generators. Sequential reuse on one stream is safe
//! (the offset is captured at enqueue time). The cache is intentionally never
//! destroyed: tearing down cuRAND/CUDA state from a static destructor races
//! the runtime's own shutdown.
class __philox_cache
{
public:
  static __philox_cache& __instance()
  {
    static __philox_cache* __c = new __philox_cache(); // deliberately leaked, see above
    return *__c;
  }

  //! A generator for the CURRENT device bound to @p __stream, seeded and
  //! positioned per the sharding-invariance contract.
  curandGenerator_t __get(unsigned long long __seed, unsigned long long __offset, cudaStream_t __stream)
  {
    int __dev = 0;
    cuda_safe_call(cudaGetDevice(&__dev));
    curandGenerator_t __g = nullptr;
    {
      ::std::lock_guard<::std::mutex> __lock(__mutex_);
      auto __it = __gens_.find(__key{__dev, __stream});
      if (__it == __gens_.end())
      {
        __curand_check(curandCreateGenerator(&__g, CURAND_RNG_PSEUDO_PHILOX4_32_10), "create");
        __curand_check(curandSetGeneratorOrdering(__g, CURAND_ORDERING_PSEUDO_DEFAULT), "ordering");
        __curand_check(curandSetStream(__g, __stream), "stream");
        __gens_.emplace(__key{__dev, __stream}, __g);
      }
      else
      {
        __g = __it->second;
      }
    }
    __curand_check(curandSetPseudoRandomGeneratorSeed(__g, __seed), "seed");
    __curand_check(curandSetGeneratorOffset(__g, __offset), "offset");
    return __g;
  }

private:
  struct __key
  {
    int __dev;
    cudaStream_t __stream;
    bool operator==(const __key& __o) const
    {
      return __dev == __o.__dev && __stream == __o.__stream;
    }
  };
  struct __key_hash
  {
    size_t operator()(const __key& __k) const
    {
      return ::std::hash<void*>{}(static_cast<void*>(__k.__stream))
           ^ (static_cast<size_t>(__k.__dev) * 0x9E3779B97F4A7C15ull);
    }
  };
  ::std::mutex __mutex_;
  ::std::unordered_map<__key, curandGenerator_t, __key_hash> __gens_;
};

//! Per-call facade over the cache (keeps the call sites' shape).
struct __philox_set
{
  curandGenerator_t __make(unsigned long long __seed, unsigned long long __offset, cudaStream_t __stream)
  {
    return __philox_cache::__instance().__get(__seed, __offset, __stream);
  }
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
        // Documented cuRAND contract: pseudo-random normal generation
        // requires an EVEN count (CURAND_STATUS_LENGTH_NOT_MULTIPLE
        // otherwise; some toolkits accept odd n, but the documented contract
        // is even — review r3940236061). Decompose the shard into an
        // even-offset/even-count main run plus up to two boundary elements,
        // each taken from a 2-element run at an even offset: every host-API
        // call is contract-conforming and the global sequence is unchanged.
        const unsigned long long __o    = static_cast<unsigned long long>(__d.global_offset);
        const unsigned long long __n    = static_cast<unsigned long long>(__d.size);
        const unsigned long long __head = __o & 1ull; // shard starts mid-pair
        const unsigned long long __main = (__n - __head) & ~1ull; // even
        const bool __tail               = (__head + __main) != __n; // one trailing element

        if (__main != 0)
        {
          curandGenerator_t __g = __gens.__make(__seed, __o + __head, __stream);
          reserved::__curand_check(
            reserved::__host_generate_normal<_Tp>(
              __g, __d.data + __head, static_cast<::std::size_t>(__main), __mean, __stddev),
            "generate_normal");
        }
        if (__head || __tail)
        {
          // Boundary elements land in a small stream-ordered scratch first:
          // the 2-run also produces the neighboring element, which belongs
          // to the adjacent shard and must not be written here.
          _Tp* __scratch = nullptr;
          cuda_safe_call(cudaMallocAsync(&__scratch, 4 * sizeof(_Tp), __stream));
          if (__head) // element __o, second of the pair starting at __o - 1
          {
            curandGenerator_t __gh = __gens.__make(__seed, __o - 1, __stream);
            reserved::__curand_check(
              reserved::__host_generate_normal<_Tp>(__gh, __scratch, 2, __mean, __stddev), "generate_normal head");
            cuda_safe_call(cudaMemcpyAsync(__d.data, __scratch + 1, sizeof(_Tp), cudaMemcpyDeviceToDevice, __stream));
          }
          if (__tail) // element __o + __n - 1, first of the pair starting there (even by construction)
          {
            curandGenerator_t __gt = __gens.__make(__seed, __o + __n - 1, __stream);
            reserved::__curand_check(
              reserved::__host_generate_normal<_Tp>(__gt, __scratch + 2, 2, __mean, __stddev), "generate_normal tail");
            cuda_safe_call(
              cudaMemcpyAsync(__d.data + (__n - 1), __scratch + 2, sizeof(_Tp), cudaMemcpyDeviceToDevice, __stream));
          }
          cuda_safe_call(cudaFreeAsync(__scratch, __stream));
        }
      }
    });
}
} // namespace cuda::experimental::sharded
