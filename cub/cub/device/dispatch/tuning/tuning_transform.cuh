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

#include <cuda/cmath>
#include <cuda/numeric>
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

namespace detail::transform
{
enum class Algorithm
{
  // We previously had a fallback algorithm that would use cub::DeviceFor. Benchmarks showed that the prefetch algorithm
  // is always superior to that fallback, so it was removed.
  prefetch,
  vectorized,
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

template <int BlockThreads, int BulkCopyAlignment>
struct async_copy_policy_t
{
  static constexpr int block_threads = BlockThreads;
  // items per tile are determined at runtime. these (inclusive) bounds allow overriding that value via a tuning policy
  static constexpr int min_items_per_thread = 1;
  static constexpr int max_items_per_thread = 32;

  static constexpr int bulk_copy_alignment = BulkCopyAlignment;
};

template <int BlockThreads, int ItemsPerThread, int LoadStoreWordSize>
struct vectorized_policy_t : prefetch_policy_t<BlockThreads>
{
  static constexpr int items_per_thread     = ItemsPerThread;
  static constexpr int load_store_word_size = LoadStoreWordSize;
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

constexpr int bulk_copy_size_multiple = 16;

_CCCL_HOST_DEVICE constexpr auto bulk_copy_alignment(int ptx_version) -> int
{
  return ptx_version < 1000 ? 128 : 16;
}

template <typename... RandomAccessIteratorsIn>
_CCCL_HOST_DEVICE constexpr auto bulk_copy_smem_for_tile_size(int tile_size, int bulk_copy_align) -> int
{
  return round_up_to_po2_multiple(int{sizeof(int64_t)}, bulk_copy_align) /* bar */
       // 128 bytes of padding for each input tile (handles before + after)
       + tile_size * loaded_bytes_per_iteration<RandomAccessIteratorsIn...>()
       + sizeof...(RandomAccessIteratorsIn) * bulk_copy_align;
}

_CCCL_HOST_DEVICE constexpr int arch_to_min_bytes_in_flight(int sm_arch)
{
  if (sm_arch >= 1000)
  {
    return 64 * 1024; // B200
  }
  if (sm_arch >= 900)
  {
    return 48 * 1024; // 32 for H100, 48 for H200
  }
  if (sm_arch >= 800)
  {
    return 16 * 1024; // A100
  }
  return 12 * 1024; // V100 and below
}

template <typename T, typename... Ts>
_CCCL_HOST_DEVICE constexpr bool are_equal([[maybe_unused]] T head, Ts... tail)
{
  return ((head == tail) && ...);
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

  template <typename = void>
  _CCCL_HOST_DEVICE static constexpr int ItemsPerThread()
  {
    return StaticPolicyT::algo_policy::items_per_thread;
  }

  _CCCL_HOST_DEVICE static constexpr int LoadStoreWordSize()
  {
    return StaticPolicyT::algo_policy::load_store_word_size;
  }
};

template <typename PolicyT>
_CCCL_HOST_DEVICE TransformPolicyWrapper<PolicyT> MakeTransformPolicyWrapper(PolicyT base)
{
  return TransformPolicyWrapper<PolicyT>(base);
}

template <typename T>
inline constexpr size_t size_of = sizeof(T);

template <>
inline constexpr size_t size_of<void> = 0;

template <bool RequiresStableAddress, typename RandomAccessIteratorTupleIn, typename RandomAccessIteratorOut>
struct policy_hub
{
  static_assert(sizeof(RandomAccessIteratorTupleIn) == 0, "Second parameter must be a tuple");
};

template <bool RequiresStableAddress, typename... RandomAccessIteratorsIn, typename RandomAccessIteratorOut>
struct policy_hub<RequiresStableAddress, ::cuda::std::tuple<RandomAccessIteratorsIn...>, RandomAccessIteratorOut>
{
  static constexpr bool no_input_streams = sizeof...(RandomAccessIteratorsIn) == 0;
  static constexpr bool all_contiguous =
    (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIteratorsIn> && ...);
  static constexpr bool all_values_trivially_reloc =
    (THRUST_NS_QUALIFIER::is_trivially_relocatable_v<it_value_t<RandomAccessIteratorsIn>> && ...);
  static constexpr bool can_memcpy = all_contiguous && all_values_trivially_reloc;
  static constexpr bool all_values_same_size =
    are_equal(size_of<it_value_t<RandomAccessIteratorsIn>>..., size_of<it_value_t<RandomAccessIteratorOut>>);

  // for vectorized policy:
  static constexpr int load_store_word_size = 8;
  static constexpr int value_type_size      = ::cuda::std::max(int{size_of<it_value_t<RandomAccessIteratorOut>>}, 1);
  static constexpr int bytes_per_tile = ::cuda::round_up(32, ::cuda::std::lcm(load_store_word_size, value_type_size));
  static_assert(bytes_per_tile % value_type_size == 0);
  static constexpr int items_per_thread = bytes_per_tile / value_type_size;
  static_assert((items_per_thread * value_type_size) % load_store_word_size == 0);
  using default_vectorized_policy_t = vectorized_policy_t<256, items_per_thread, load_store_word_size>;

  // TODO(bgruber): consider a separate kernel for just filling

  struct policy300 : ChainedPolicy<300, policy300, policy300>
  {
    static constexpr int min_bif       = arch_to_min_bytes_in_flight(300);
    static constexpr bool use_fallback = RequiresStableAddress || !can_memcpy || !all_values_same_size;
    // TODO(bgruber): we don't need algo, because we can just detect the type of algo_policy
    static constexpr auto algorithm = use_fallback ? Algorithm::prefetch : Algorithm::vectorized;
    using algo_policy = ::cuda::std::_If<use_fallback, prefetch_policy_t<256>, default_vectorized_policy_t>;
  };

#ifdef _CUB_HAS_TRANSFORM_UBLKCP
  template <int BlockSize, int PtxVersion>
  struct bulkcopy_policy
  {
  private:
    static constexpr int bulk_copy_align = bulk_copy_alignment(PtxVersion);
    using async_policy                   = async_copy_policy_t<BlockSize, bulk_copy_align>;
    static constexpr bool exhaust_smem =
      bulk_copy_smem_for_tile_size<RandomAccessIteratorsIn...>(
        async_policy::block_threads * async_policy::min_items_per_thread, bulk_copy_align)
      > int{max_smem_per_block};
    static constexpr bool any_type_is_overalinged =
      ((alignof(it_value_t<RandomAccessIteratorsIn>) > bulk_copy_align) || ...);

    static constexpr bool use_fallback =
      RequiresStableAddress || !can_memcpy || no_input_streams || exhaust_smem || any_type_is_overalinged;

  public:
    static constexpr int min_bif    = arch_to_min_bytes_in_flight(PtxVersion);
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
#endif // _CUB_HAS_TRANSFORM_UBLKCP

  // UBLKCP is disabled on sm120 for now
  struct policy1200
      : ChainedPolicy<1200,
                      policy1200,
#ifdef _CUB_HAS_TRANSFORM_UBLKCP
                      policy1000
#else // _CUB_HAS_TRANSFORM_UBLKCP
                      policy300
#endif // _CUB_HAS_TRANSFORM_UBLKCP
                      >
  {
    static constexpr int min_bif    = arch_to_min_bytes_in_flight(1200);
    static constexpr auto algorithm = Algorithm::prefetch;
    using algo_policy               = prefetch_policy_t<256>;
  };

  using max_policy = policy1200;
};

} // namespace detail::transform

CUB_NAMESPACE_END
