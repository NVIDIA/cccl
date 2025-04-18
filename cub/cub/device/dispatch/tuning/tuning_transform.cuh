/******************************************************************************
 * Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 *AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 *IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/util_device.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>
#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/std/__cccl/execution_space.h>
#include <cuda/std/bit>

// The ublkcp kernel needs PTX features that are only available and understood by nvcc >=12.
// Also, cooperative groups do not support NVHPC yet.
#if !_CCCL_CUDA_COMPILER(NVHPC)
#  ifndef _CUB_HAS_TRANSFORM_UBLKCP
#    define _CUB_HAS_TRANSFORM_UBLKCP 1
#  endif // !_CUB_HAS_TRANSFORM_UBLKCP
#endif // !_CCCL_CUDA_COMPILER(NVHPC)

CUB_NAMESPACE_BEGIN

namespace detail
{
namespace transform
{
enum class Algorithm
{
  // We previously had a fallback algorithm that would use cub::DeviceFor. Benchmarks showed that the prefetch algorithm
  // is always superior to that fallback, so it was removed.
  prefetch,
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  ublkcp,
#endif // _CUB_HAS_TRANSFORM_UBLKCP
};

template <int BlockThreads>
struct prefetch_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int items_per_thread_no_input = 2; // when there are no input iterators, the kernel is just filling
  static constexpr int min_items_per_thread      = 1;
  static constexpr int max_items_per_thread      = 32;
};

template <int BlockThreads>
struct async_copy_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int min_items_per_thread = 1;
  static constexpr int max_items_per_thread = 32;
};

// mult must be a power of 2
template <typename Integral>
_CCCL_HOST_DEVICE _CCCL_FORCEINLINE constexpr auto round_up_to_po2_multiple(Integral x, Integral mult) -> Integral
{
  _CCCL_ASSERT(::cuda::std::has_single_bit(static_cast<::cuda::std::make_unsigned_t<Integral>>(mult)), "");
  return (x + mult - 1) & ~(mult - 1);
}

_CCCL_HOST_DEVICE constexpr int sum()
{
  return 0;
}

// TODO(bgruber): remove with C++17
template <typename... Ts>
_CCCL_HOST_DEVICE constexpr int sum(int head, Ts... tail)
{
  return head + sum(tail...);
}

template <typename... Its>
_CCCL_HOST_DEVICE constexpr auto loaded_bytes_per_iteration() -> int
{
  return (int{sizeof(it_value_t<Its>)} + ... + 0);
}

constexpr int bulk_copy_alignment     = 128;
constexpr int bulk_copy_size_multiple = 16;

template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE constexpr auto bulk_copy_smem_for_tile_size(int tile_size) -> int
{
  return round_up_to_po2_multiple(int{sizeof(int64_t)}, bulk_copy_alignment) /* bar */
       // 128 bytes of padding for each input tile (handles before + after)
       + tile_size * loaded_bytes_per_iteration<RandomAccessIteratorsIn...>()
       + sizeof...(RandomAccessIteratorsIn) * bulk_copy_alignment;
}

_CCCL_HOST_DEVICE constexpr int arch_to_min_bytes_in_flight(int sm_arch)
{
  // TODO(bgruber): use if-else in C++14 for better readability
  return sm_arch >= 900 ? 48 * 1024 // 32 for H100, 48 for H200
       : sm_arch >= 800 ? 16 * 1024 // A100
                        : 12 * 1024; // V100 and below
}

template <typename PolicyT, typename = void>
struct TransformPolicyWrapper : PolicyT
{
  _CCCL_HOST_DEVICE TransformPolicyWrapper(PolicyT base)
      : PolicyT(base)
  {}
};

template <typename StaticPolicyT>
struct TransformPolicyWrapper<StaticPolicyT, ::cuda::std::void_t<decltype(StaticPolicyT::algorithm)>> : StaticPolicyT
{
  _CCCL_HOST_DEVICE TransformPolicyWrapper(StaticPolicyT base)
      : StaticPolicyT(base)
  {}

  _CCCL_HOST_DEVICE static constexpr Algorithm GetAlgorithm()
  {
    return StaticPolicyT::algorithm;
  }

  _CCCL_HOST_DEVICE static constexpr int BlockThreads()
  {
    return StaticPolicyT::algo_policy::block_threads;
  }

  _CCCL_HOST_DEVICE static constexpr int ItemsPerThreadNoInput()
  {
    return StaticPolicyT::algo_policy::items_per_thread_no_input;
  }

  _CCCL_HOST_DEVICE static constexpr int MinItemsPerThread()
  {
    return StaticPolicyT::algo_policy::min_items_per_thread;
  }

  _CCCL_HOST_DEVICE static constexpr int MaxItemsPerThread()
  {
    return StaticPolicyT::algo_policy::max_items_per_thread;
  }
};

template <typename PolicyT>
_CCCL_HOST_DEVICE TransformPolicyWrapper<PolicyT> MakeTransformPolicyWrapper(PolicyT base)
{
  return TransformPolicyWrapper<PolicyT>(base);
}

template <bool RequiresStableAddress, typename RandomAccessIteratorTupleIn>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress, typename... RandomAccessIteratorsIn>
struct policy_hub<RequiresStableAddress, ::cuda::std::tuple<RandomAccessIteratorsIn...>>
{
  static constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
  static constexpr bool all_contiguous =
    ::cuda::std::conjunction_v<THRUST_NS_QUALIFIER::is_contiguous_iterator<RandomAccessIteratorsIn>...>;
  static constexpr bool all_values_trivially_reloc =
    ::cuda::std::conjunction_v<THRUST_NS_QUALIFIER::is_trivially_relocatable<it_value_t<RandomAccessIteratorsIn>>...>;

  static constexpr bool can_memcpy = all_contiguous && all_values_trivially_reloc;

  // TODO(bgruber): consider a separate kernel for just filling

  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(300);
    // TODO(bgruber): we don't need algo, because we can just detect the type of algo_policy
    static constexpr auto algorithm = Algorithm::prefetch;
    using algo_policy               = prefetch_policy_t<256>;
  };

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  template <int BlockSize, int PtxVersion>
  struct bulkcopy_policy
  {
    static constexpr int min_bif = arch_to_min_bytes_in_flight(PtxVersion);
    using async_policy           = async_copy_policy_t<BlockSize>;
    static constexpr bool exhaust_smem =
      bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(
        async_policy::block_threads * async_policy::min_items_per_thread)
      > int{max_smem_per_block};
    static constexpr bool any_type_is_overalinged =
      ((alignof(it_value_t<RandomAccessIteratorsIn>) > bulk_copy_alignment) || ...);

    static constexpr bool use_fallback =
      RequiresStableAddress || !can_memcpy || no_input_streams || exhaust_smem || any_type_is_overalinged;
    static constexpr auto algorithm = use_fallback ? Algorithm::prefetch : Algorithm::ublkcp;
    using algo_policy               = ::cuda::std::_If<use_fallback, prefetch_policy_t<BlockSize>, async_policy>;
  };

  struct policy900
      : bulkcopy_policy<256, 900>
      , ChainedPolicy<900, policy900, policy300>
  {};

  struct policy1000
      : bulkcopy_policy<128, 1000>
      , ChainedPolicy<1000, policy1000, policy900>
  {};

  using max_policy = policy1000;
#else // _CUB_HAS_TRANSFORM_UBLKCP
  using max_policy = policy300;
#endif // _CUB_HAS_TRANSFORM_UBLKCP
};

} // namespace transform
} // namespace detail

CUB_NAMESPACE_END
