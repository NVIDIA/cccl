// SPDX-FileCopyrightText: Copyright (c) 2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/arch_dispatch.cuh>
#include <cub/detail/detect_cuda_runtime.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/uninitialized_copy.cuh>
#include <cub/device/dispatch/kernels/kernel_transform.cuh>
#include <cub/util_arch.cuh>
#include <cub/util_device.cuh>
#include <cub/util_math.cuh>
#include <cub/util_type.cuh>

#include <thrust/system/cuda/detail/core/triple_chevron_launch.h>
#include <thrust/type_traits/is_trivially_relocatable.h>
#include <thrust/type_traits/unwrap_contiguous_iterator.h>

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/std/__algorithm/clamp.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/void_t.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/move.h>
#include <cuda/std/array>
#include <cuda/std/cassert>
#include <cuda/std/cstdint>
#include <cuda/std/expected>
#include <cuda/std/optional>
#include <cuda/std/tuple>

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
#  include <sstream>
#endif

// On Windows, the `if CUB_DETAIL_CONSTEXPR_ISH` results in `warning C4702: unreachable code`.
_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702)
_CCCL_DIAG_SUPPRESS_GCC("-Wattributes") // __visibility__ attribute ignored

CUB_NAMESPACE_BEGIN

namespace detail::transform
{
template <typename T>
using cuda_expected = ::cuda::std::expected<T, cudaError_t>;

struct async_config
{
  int items_per_thread;
  int max_occupancy;
  int sm_count;
};

struct prefetch_config
{
  int max_occupancy;
  int sm_count;
};

template <typename PolicySelector,
          typename Offset,
          typename RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename Predicate,
          typename TransformOp>
struct TransformKernelSource;

template <typename PolicySelector,
          typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename Predicate,
          typename TransformOp>
struct TransformKernelSource<PolicySelector,
                             Offset,
                             ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                             RandomAccessIteratorOut,
                             Predicate,
                             TransformOp>
{
  // PolicySelector must be stateless, so we can pass the type to the kernel
  static_assert(::cuda::std::is_empty_v<PolicySelector>);

  CUB_DEFINE_KERNEL_GETTER(
    TransformKernel,
    transform_kernel<PolicySelector,
                     Offset,
                     Predicate,
                     TransformOp,
                     THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorOut>,
                     THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator_t<RandomAccessIteratorsIn>...>);

  template <class ActionT>
  CUB_RUNTIME_FUNCTION cuda_expected<async_config> CacheAsyncConfiguration(const ActionT& action)
  {
    NV_IF_TARGET(NV_IS_HOST, (static auto cached_config = action(); return cached_config;), (return action();))
  }

  template <class ActionT>
  CUB_RUNTIME_FUNCTION cuda_expected<prefetch_config> CachePrefetchConfiguration(const ActionT& action)
  {
    NV_IF_TARGET(NV_IS_HOST, (static auto cached_config = action(); return cached_config;), (return action();))
  }

  CUB_RUNTIME_FUNCTION static constexpr int LoadedBytesPerIteration()
  {
    return loaded_bytes_per_iteration<RandomAccessIteratorsIn...>();
  }

  CUB_RUNTIME_FUNCTION static constexpr auto InputIteratorInfos()
  {
    return ::cuda::std::array<iterator_info, sizeof...(RandomAccessIteratorsIn)>{
      make_iterator_info<RandomAccessIteratorsIn>()...};
  }

  template <typename It>
  CUB_RUNTIME_FUNCTION static constexpr kernel_arg<It> MakeIteratorKernelArg(It it)
  {
    return detail::transform::make_iterator_kernel_arg(it);
  }

  template <typename It>
  CUB_RUNTIME_FUNCTION static constexpr kernel_arg<It> MakeAlignedBasePtrKernelArg(It it, int align)
  {
    return detail::transform::make_aligned_base_ptr_kernel_arg(it, align);
  }

private:
  template <typename T>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static auto is_pointer_aligned(T it, [[maybe_unused]] int alignment)
  {
    if constexpr (THRUST_NS_QUALIFIER::is_contiguous_iterator_v<decltype(it)>)
    {
      return ::cuda::is_aligned(THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(it), alignment);
    }
    else
    {
      return true; // fancy iterators are aligned, since the vectorized kernel chooses a different code path
    }
  }

public:
  CUB_RUNTIME_FUNCTION constexpr static bool
  CanVectorize(int vec_size, const RandomAccessIteratorOut& out, const RandomAccessIteratorsIn&... in)
  {
    return is_pointer_aligned(out, size_of<it_value_t<RandomAccessIteratorOut>> * vec_size)
        && (is_pointer_aligned(in, sizeof(it_value_t<RandomAccessIteratorsIn>) * vec_size) && ...);
  }
};

enum class requires_stable_address
{
  no,
  yes
};

// Reduces the items_per_thread when necessary to generate enough blocks to reach the maximum occupancy.
template <typename Offset, typename Policy>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE int
spread_out_items_per_thread(Offset num_items, Policy policy, int items_per_thread, int sm_count, int max_occupancy)
{
  const int items_per_thread_evenly_spread = static_cast<int>(
    (::cuda::std::min) (Offset{items_per_thread},
                        ::cuda::ceil_div(num_items, sm_count * policy.block_threads * max_occupancy)));
  const int items_per_thread_clamped =
    ::cuda::std::clamp(items_per_thread_evenly_spread, policy.min_items_per_thread, policy.max_items_per_thread);
  return items_per_thread_clamped;
}

template <bool NoInputs,
          typename Offset,
          typename SMemFunc,
          typename PolicyGetter,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE auto configure_async_kernel(
  Offset num_items,
  int alignment,
  SMemFunc dyn_smem_for_tile_size,
  cudaStream_t stream,
  PolicyGetter policy_getter,
  KernelSource kernel_source,
  KernelLauncherFactory launcher_factory)
  -> cuda_expected<
    ::cuda::std::tuple<decltype(launcher_factory(0, 0, 0, 0)), decltype(kernel_source.TransformKernel()), int>>
{
  CUB_DETAIL_CONSTEXPR_ISH const transform_policy policy = policy_getter();
  CUB_DETAIL_CONSTEXPR_ISH int block_threads             = policy.async_copy.block_threads;

  _CCCL_ASSERT(block_threads % alignment == 0, "block_threads needs to be a multiple of the copy alignment");
  // ^ then tile_size is a multiple of it

  CUB_DETAIL_CONSTEXPR_ISH auto min_items_per_thread = policy.async_copy.min_items_per_thread;
  CUB_DETAIL_CONSTEXPR_ISH auto max_items_per_thread = policy.async_copy.max_items_per_thread;

  // ensures the loop below runs at least once
  // pulled outside of the lambda below to make MSVC happy
  CUB_DETAIL_STATIC_ISH_ASSERT(min_items_per_thread <= max_items_per_thread, "invalid policy");

  auto determine_element_counts = [&]() -> cuda_expected<async_config> {
    int sm_count = 0;
    auto error   = CubDebug(launcher_factory.MultiProcessorCount(sm_count));
    if (error != cudaSuccess)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
    }

    // Increase the number of output elements per thread until we reach the required bytes in flight.
    // Benchmarking shows that even for a few iteration, this loop takes around 4-7 us, so should not be a concern.
    // This computation MUST NOT depend on any runtime state of the current API invocation (like num_items), since the
    // result will be cached.
    async_config last_config{};
    for (int items_per_thread = +min_items_per_thread; items_per_thread <= +max_items_per_thread; ++items_per_thread)
    {
      const int tile_size     = block_threads * items_per_thread;
      const int dyn_smem_size = dyn_smem_for_tile_size(tile_size, alignment);
      int max_occupancy       = 0;
      error                   = CubDebug(
        launcher_factory.MaxSmOccupancy(max_occupancy, kernel_source.TransformKernel(), block_threads, dyn_smem_size));
      if (error != cudaSuccess)
      {
        return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
      }
      if (max_occupancy == 0)
      {
        // assert should be prevented by smem check in policy
        _CCCL_ASSERT(last_config.items_per_thread > 0, "min_items_per_thread exceeds available shared memory");
        return last_config;
      }

      const auto config = async_config{items_per_thread, max_occupancy, sm_count};

      const int bytes_in_flight_SM = max_occupancy * tile_size * kernel_source.LoadedBytesPerIteration();
      if (policy.min_bytes_in_flight <= bytes_in_flight_SM)
      {
        return config;
      }

      last_config = config;
    }
    return last_config;
  };
  cuda_expected<async_config> config = kernel_source.CacheAsyncConfiguration(determine_element_counts);
  if (!config)
  {
    return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(config.error());
  }
  _CCCL_ASSERT(config->items_per_thread > 0, "");
  _CCCL_ASSERT((config->items_per_thread * block_threads) % alignment == 0, "");

  const int ipt = spread_out_items_per_thread(
    num_items, policy.async_copy, config->items_per_thread, config->sm_count, config->max_occupancy);
  const int tile_size     = block_threads * ipt;
  const int dyn_smem_size = dyn_smem_for_tile_size(tile_size, alignment);
  _CCCL_ASSERT(NoInputs != (dyn_smem_size != 0), ""); // logical xor

  const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{tile_size}));
  // config->smem_size is 16 bytes larger than needed for UBLKCP because it's the total SMEM size, but 16 bytes are
  // occupied by static shared memory and padding. But let's not complicate things.
  return ::cuda::std::make_tuple(
    launcher_factory(grid_dim, block_threads, dyn_smem_size, stream, true), kernel_source.TransformKernel(), ipt);
}

template <typename Offset,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename Predicate,
          typename TransformOp,
          typename SMemFunc,
          std::size_t... Is,
          typename PolicyGetter,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_async_algorithm(
  ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
  RandomAccessIteratorOut out,
  Offset num_items,
  Predicate pred,
  TransformOp op,
  cudaStream_t stream,
  int alignment,
  SMemFunc dyn_smem_for_tile_size,
  cuda::std::index_sequence<Is...>,
  PolicyGetter policy_getter,
  KernelSource kernel_source,
  KernelLauncherFactory launcher_factory)
{
  auto ret = configure_async_kernel<(sizeof...(RandomAccessIteratorsIn) == 0)>(
    num_items, alignment, dyn_smem_for_tile_size, stream, policy_getter, kernel_source, launcher_factory);
  if (!ret)
  {
    return ret.error();
  }

  auto [launcher, kernel, items_per_thread] = *ret;
  return launcher.doit(
    kernel,
    num_items,
    items_per_thread,
    false,
    pred,
    op,
    THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(out),
    kernel_source.MakeAlignedBasePtrKernelArg(
      THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)), alignment)...);
}

template <typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename Offset,
          typename Predicate,
          typename TransformOp,
          size_t... Is,
          typename PolicyGetter,
          typename KernelSource,
          typename KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_prefetch_or_vectorized_algorithm(
  [[maybe_unused]] ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
  RandomAccessIteratorOut out,
  Offset num_items,
  Predicate pred,
  TransformOp op,
  cudaStream_t stream,
  ::cuda::std::index_sequence<Is...>,
  PolicyGetter policy_getter,
  KernelSource kernel_source,
  KernelLauncherFactory launcher_factory)
{
  CUB_DETAIL_CONSTEXPR_ISH const transform_policy policy = policy_getter();
  CUB_DETAIL_CONSTEXPR_ISH const int block_threads =
    policy.algorithm == Algorithm::vectorized ? policy.vectorized.block_threads : policy.prefetch.block_threads;

  auto determine_config = [&]() -> cuda_expected<prefetch_config> {
    int max_occupancy = 0;
    auto error =
      CubDebug(launcher_factory.MaxSmOccupancy(max_occupancy, kernel_source.TransformKernel(), block_threads, 0));
    if (error != cudaSuccess)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
    }
    int sm_count = 0;
    error        = CubDebug(launcher_factory.MultiProcessorCount(sm_count));
    if (error != cudaSuccess)
    {
      return ::cuda::std::unexpected<cudaError_t /* nvcc 12.0 fails CTAD here */>(error);
    }
    return prefetch_config{max_occupancy, sm_count};
  };

  cuda_expected<prefetch_config> config = kernel_source.CachePrefetchConfiguration(determine_config);
  if (!config)
  {
    return config.error();
  }

  auto can_vectorize = false;
  ::cuda::std::optional<int> ipt;

  // the policy already handles the compile-time checks if we can vectorize. Do the remaining alignment check here
  if CUB_DETAIL_CONSTEXPR_ISH (Algorithm::vectorized == policy.algorithm)
  {
    const int vs  = policy.vectorized.vec_size;
    can_vectorize = kernel_source.CanVectorize(vs, out, ::cuda::std::get<Is>(in)...);
    if (can_vectorize)
    {
      ipt = policy.vectorized.items_per_thread;
    }
  }

  if (!ipt)
  {
    // otherwise, set up the prefetch kernel
    const auto fallback_prefetch_policy = prefetch_policy{
      policy.vectorized.block_threads,
      policy.vectorized.prefetch_items_per_thread_no_input,
      policy.vectorized.prefetch_min_items_per_thread,
      policy.vectorized.prefetch_max_items_per_thread};
    const auto prefetch_policy = policy.algorithm == Algorithm::prefetch ? policy.prefetch : fallback_prefetch_policy;

    auto loaded_bytes_per_iter           = kernel_source.LoadedBytesPerIteration();
    const auto items_per_thread_no_input = prefetch_policy.items_per_thread_no_input;
    // choose items per thread to reach minimum bytes in flight
    const int items_per_thread =
      loaded_bytes_per_iter == 0
        ? items_per_thread_no_input
        : ::cuda::ceil_div(policy.min_bytes_in_flight, config->max_occupancy * block_threads * loaded_bytes_per_iter);

    // but also generate enough blocks for full occupancy to optimize small problem sizes, e.g., 2^16/2^20 elements
    ipt = spread_out_items_per_thread(
      num_items, prefetch_policy, items_per_thread, config->sm_count, config->max_occupancy);
  }
  _CCCL_ASSERT(ipt, "");
  const int tile_size = block_threads * ipt.value();
  const auto grid_dim = static_cast<unsigned int>(::cuda::ceil_div(num_items, Offset{tile_size}));
  return CubDebug(
    launcher_factory(grid_dim, block_threads, 0, stream, true)
      .doit(kernel_source.TransformKernel(),
            num_items,
            ipt.value(),
            can_vectorize,
            pred,
            op,
            THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(out),
            kernel_source.MakeIteratorKernelArg(
              THRUST_NS_QUALIFIER::try_unwrap_contiguous_iterator(::cuda::std::get<Is>(in)))...));
}

// This should ideally have been a lambda, but MSVC < 14.44 ICEs if we put a `(static) constexpr` variable inside
template <typename RandomAccessIteratorTupleIn,
          typename RandomAccessIteratorOut,
          typename Offset,
          typename Predicate,
          typename TransformOp,
          typename KernelSource,
          typename KernelLauncherFactory>
struct invoke_for_arch;

template <typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename Offset,
          typename Predicate,
          typename TransformOp,
          typename KernelSource,
          typename KernelLauncherFactory>
struct invoke_for_arch<::cuda::std::tuple<RandomAccessIteratorsIn...>,
                       RandomAccessIteratorOut,
                       Offset,
                       Predicate,
                       TransformOp,
                       KernelSource,
                       KernelLauncherFactory>
{
  ::cuda::std::tuple<RandomAccessIteratorsIn...> in;
  RandomAccessIteratorOut out;
  Offset num_items;
  Predicate pred;
  TransformOp op;
  cudaStream_t stream;
  KernelSource kernel_source;
  KernelLauncherFactory launcher_factory;
  ::cuda::arch_id arch_id;

  template <typename PolicyGetter>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t operator()(PolicyGetter policy_getter) const
  {
    CUB_DETAIL_CONSTEXPR_ISH transform_policy active_policy = policy_getter();
    const auto seq = ::cuda::std::index_sequence_for<RandomAccessIteratorsIn...>{};

#if !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)
    NV_IF_TARGET(
      NV_IS_HOST,
      (std::stringstream ss; ss << active_policy;
       _CubLog("Dispatching DeviceTransform to arch %d with tuning: %s\n", (int) arch_id, ss.str().c_str());))
#endif // !_CCCL_COMPILER(NVRTC) && defined(CUB_DEBUG_LOG)

    if CUB_DETAIL_CONSTEXPR_ISH (Algorithm::ublkcp == active_policy.algorithm)
    {
      return invoke_async_algorithm(
        ::cuda::std::move(in),
        ::cuda::std::move(out),
        num_items,
        ::cuda::std::move(pred),
        ::cuda::std::move(op),
        stream,
        bulk_copy_alignment(arch_id),
        [&](int tile_size, int alignment) {
          return bulk_copy_dyn_smem_for_tile_size<sizeof...(RandomAccessIteratorsIn)>(
            kernel_source.InputIteratorInfos(), tile_size, alignment);
        },
        seq,
        policy_getter,
        kernel_source,
        launcher_factory);
    }
    else if CUB_DETAIL_CONSTEXPR_ISH (Algorithm::memcpy_async == active_policy.algorithm)
    {
      return invoke_async_algorithm(
        ::cuda::std::move(in),
        ::cuda::std::move(out),
        num_items,
        ::cuda::std::move(pred),
        ::cuda::std::move(op),
        stream,
        ldgsts_size_and_align,
        [&](int tile_size, int alignment) {
          return memcpy_async_dyn_smem_for_tile_size<sizeof...(RandomAccessIteratorsIn)>(
            kernel_source.InputIteratorInfos(), tile_size, alignment);
        },
        seq,
        policy_getter,
        kernel_source,
        launcher_factory);
    }
    else
    {
      return invoke_prefetch_or_vectorized_algorithm(
        ::cuda::std::move(in),
        ::cuda::std::move(out),
        num_items,
        ::cuda::std::move(pred),
        ::cuda::std::move(op),
        stream,
        seq,
        policy_getter,
        kernel_source,
        launcher_factory);
    }
  }
};

template <requires_stable_address StableAddress,
          typename... RandomAccessIteratorsIn,
          typename RandomAccessIteratorOut,
          typename Offset,
          typename Predicate,
          typename TransformOp,
          typename PolicySelector        = policy_selector_from_types<StableAddress == requires_stable_address::yes,
                                                                      ::cuda::std::is_same_v<Predicate, always_true_predicate>,
                                                                      ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                                                      RandomAccessIteratorOut>,
          typename KernelSource          = TransformKernelSource<PolicySelector,
                                                                 Offset,
                                                                 ::cuda::std::tuple<RandomAccessIteratorsIn...>,
                                                                 RandomAccessIteratorOut,
                                                                 Predicate,
                                                                 TransformOp>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires transform_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t dispatch(
  ::cuda::std::tuple<RandomAccessIteratorsIn...> in,
  RandomAccessIteratorOut out,
  Offset num_items,
  Predicate pred,
  TransformOp op,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  static_assert(
    ::cuda::std::is_same_v<Offset, ::cuda::std::int32_t> || ::cuda::std::is_same_v<Offset, ::cuda::std::int64_t>,
    "cub::DeviceTransform is only tested and tuned for 32-bit or 64-bit signed offset types");

  if (num_items == 0)
  {
    return cudaSuccess;
  }

  ::cuda::arch_id arch_id{};
  if (const auto error = CubDebug(launcher_factory.PtxArchId(arch_id)))
  {
    return error;
  }

  return dispatch_arch(
    policy_selector,
    arch_id,
    invoke_for_arch<::cuda::std::tuple<RandomAccessIteratorsIn...>,
                    RandomAccessIteratorOut,
                    Offset,
                    Predicate,
                    TransformOp,
                    KernelSource,
                    KernelLauncherFactory>{
      ::cuda::std::move(in),
      ::cuda::std::move(out),
      num_items,
      ::cuda::std::move(pred),
      ::cuda::std::move(op),
      stream,
      kernel_source,
      launcher_factory,
      arch_id});
}
} // namespace detail::transform
CUB_NAMESPACE_END

_CCCL_DIAG_POP
