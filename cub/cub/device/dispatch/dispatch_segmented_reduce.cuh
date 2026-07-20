// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/choose_offset.cuh>
#include <cub/detail/launcher/cuda_runtime.cuh>
#include <cub/detail/logging.cuh>
#include <cub/device/dispatch/dispatch_common.cuh>
#include <cub/device/dispatch/kernels/kernel_segmented_reduce.cuh>
#include <cub/device/dispatch/tuning/tuning_segmented_reduce.cuh>
#include <cub/util_debug.cuh>
#include <cub/util_device.cuh>
#include <cub/util_temporary_storage.cuh>
#include <cub/util_type.cuh> // for cub::detail::non_void_value_t, cub::detail::it_value_t

#include <cuda/__cmath/ceil_div.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__host_stdlib/sstream>
#include <cuda/std/__iterator/iterator_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_arithmetic.h>
#include <cuda/std/__type_traits/is_empty.h>
#include <cuda/std/__type_traits/is_pointer.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

CUB_NAMESPACE_BEGIN

namespace detail::segmented_reduce
{
template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename BeginOffsetIteratorT,
          typename EndOffsetIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT>
struct DeviceSegmentedReduceKernelSource
{
  // PolicySelector must be stateless, so we can pass the type to the kernel
  static_assert(::cuda::std::is_empty_v<PolicySelector>);

  CUB_DEFINE_KERNEL_GETTER(
    SegmentedReduceKernel,
    DeviceSegmentedReduceKernel<
      PolicySelector,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      OffsetT,
      ReductionOpT,
      InitValueT,
      AccumT>)
};

template <typename PolicyHub, typename AccumT, typename OffsetT, typename ReductionOpT>
struct policy_selector_from_hub
{
private:
  template <typename ActivePolicyT>
  _CCCL_HOST_DEVICE_API static constexpr auto convert_policy() -> SegmentedReducePolicy
  {
    using rp  = typename ActivePolicyT::ReducePolicy;
    using srp = typename segmented_reduce::policy_hub<AccumT, OffsetT, ReductionOpT>::MaxPolicy;
    using sp  = typename srp::SmallReducePolicy;
    using mp  = typename srp::MediumReducePolicy;
    return SegmentedReducePolicy{
      {rp::BLOCK_THREADS, rp::ITEMS_PER_THREAD, rp::VECTOR_LOAD_LENGTH, rp::BLOCK_ALGORITHM, rp::LOAD_MODIFIER},
      {rp::BLOCK_THREADS, mp::WARP_THREADS, rp::ITEMS_PER_THREAD, rp::VECTOR_LOAD_LENGTH, rp::LOAD_MODIFIER},
      {rp::BLOCK_THREADS, sp::WARP_THREADS, rp::ITEMS_PER_THREAD, rp::VECTOR_LOAD_LENGTH, rp::LOAD_MODIFIER}};
  }

  struct extract_policy_dispatch_t
  {
    SegmentedReducePolicy& policy;

    template <typename ActivePolicyT>
    _CCCL_HOST_DEVICE_API constexpr cudaError_t Invoke()
    {
      policy = convert_policy<ActivePolicyT>();
      return cudaSuccess;
    }
  };

public:
  [[nodiscard]] _CCCL_HOST_DEVICE_API constexpr auto operator()(::cuda::compute_capability cc) const
    -> SegmentedReducePolicy
  {
    NV_IF_ELSE_TARGET(NV_IS_HOST,
                      ({
                        const int ptx_version = cc.get() * 10;
                        SegmentedReducePolicy policy{};
                        extract_policy_dispatch_t dispatch{policy};
                        PolicyHub::MaxPolicy::Invoke(ptx_version, dispatch);
                        return policy;
                      }),
                      (return convert_policy<typename PolicyHub::MaxPolicy::ActivePolicy>();));
  }
};
} // namespace detail::segmented_reduce

// TODO(bgruber): drop in CCCL 4.0
/**
 * @brief Utility class for dispatching the appropriately-tuned kernels for
 *        device-wide reduction
 *
 * Deprecated [Since 3.5]
 *
 * @tparam InputIteratorT
 *   Random-access input iterator type for reading input items @iterator
 *
 * @tparam OutputIteratorT
 *   Output iterator type for recording the reduced aggregate @iterator
 *
 * @tparam BeginOffsetIteratorT
 *   Random-access input iterator type for reading segment beginning offsets
 *   @iterator
 *
 * @tparam EndOffsetIteratorT
 *   Random-access input iterator type for reading segment ending offsets
 *   @iterator
 *
 * @tparam OffsetT
 *   Signed integer type for global offsets
 *
 * @tparam ReductionOpT
 *   Binary reduction functor type having member
 *   `auto operator()(const T &a, const U &b)`
 *
 * @tparam InitValueT
 *   value type
 */
template <
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT,
  typename OffsetT,
  typename ReductionOpT,
  typename InitValueT = cub::detail::non_void_value_t<OutputIteratorT, cub::detail::it_value_t<InputIteratorT>>,
  typename AccumT     = ::cuda::std::__accumulator_t<ReductionOpT, cub::detail::it_value_t<InputIteratorT>, InitValueT>,
  typename PolicyHub  = detail::segmented_reduce::policy_hub<AccumT, OffsetT, ReductionOpT>,
  typename KernelSource = detail::segmented_reduce::DeviceSegmentedReduceKernelSource<
    detail::segmented_reduce::policy_selector_from_hub<PolicyHub, AccumT, OffsetT, ReductionOpT>,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT,
    OffsetT,
    ReductionOpT,
    InitValueT,
    AccumT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
struct CCCL_DEPRECATED_BECAUSE("Use the tuning API for DeviceSegmentedReduce") DispatchSegmentedReduce
{
  //---------------------------------------------------------------------------
  // Problem state
  //---------------------------------------------------------------------------

  /// Device-accessible allocation of temporary storage. When `nullptr`, the
  /// required allocation size is written to `temp_storage_bytes` and no work
  /// is done.
  void* d_temp_storage;

  /// Reference to size in bytes of `d_temp_storage` allocation
  size_t& temp_storage_bytes;

  /// Pointer to the input sequence of data items
  InputIteratorT d_in;

  /// Pointer to the output aggregate
  OutputIteratorT d_out;

  /// The number of segments that comprise the segmented reduction data
  ::cuda::std::int64_t num_segments;

  /// Random-access input iterator to the sequence of beginning offsets of
  /// length `num_segments`, such that `d_begin_offsets[i]` is the first
  /// element of the *i*<sup>th</sup> data segment in `d_keys_*` and
  /// `d_values_*`
  BeginOffsetIteratorT d_begin_offsets;

  /// Random-access input iterator to the sequence of ending offsets of length
  /// `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
  /// the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
  /// If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
  /// considered empty.
  EndOffsetIteratorT d_end_offsets;

  /// Binary reduction functor
  ReductionOpT reduction_op;

  /// The initial value of the reduction
  InitValueT init;

  /// CUDA stream to launch kernels within. Default is stream<sub>0</sub>.
  cudaStream_t stream;

  int ptx_version;

  // Source getter
  KernelSource kernel_source;

  KernelLauncherFactory launcher_factory;

  //---------------------------------------------------------------------------
  // Constructor
  //---------------------------------------------------------------------------

  /// Constructor
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE DispatchSegmentedReduce(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitValueT init,
    cudaStream_t stream,
    int ptx_version,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {})
      : d_temp_storage(d_temp_storage)
      , temp_storage_bytes(temp_storage_bytes)
      , d_in(d_in)
      , d_out(d_out)
      , num_segments(num_segments)
      , d_begin_offsets(d_begin_offsets)
      , d_end_offsets(d_end_offsets)
      , reduction_op(reduction_op)
      , init(init)
      , stream(stream)
      , ptx_version(ptx_version)
      , kernel_source(kernel_source)
      , launcher_factory(launcher_factory)
  {}

  //---------------------------------------------------------------------------
  // Chained policy invocation
  //---------------------------------------------------------------------------

  /**
   * @brief Invocation
   *
   * @tparam ActivePolicyT
   *   Umbrella policy active for the target device
   *
   * @tparam DeviceSegmentedReduceKernelT
   *   Function type of cub::DeviceSegmentedReduceKernel
   *
   * @param[in] segmented_reduce_kernel
   *   Kernel function pointer to instantiation of
   *   cub::DeviceSegmentedReduceKernel
   */
  template <typename ActivePolicyT, typename DeviceSegmentedReduceKernelT>
  CUB_RUNTIME_FUNCTION _CCCL_VISIBILITY_HIDDEN _CCCL_FORCEINLINE cudaError_t
  InvokePasses(DeviceSegmentedReduceKernelT segmented_reduce_kernel, ActivePolicyT policy = {})
  {
    cudaError error = cudaSuccess;

    do
    {
      // Return if the caller is simply requesting the size of the storage
      // allocation
      if (d_temp_storage == nullptr)
      {
        temp_storage_bytes = 1;
        return cudaSuccess;
      }

      // Init kernel configuration (computes kernel occupancy)
      [[maybe_unused]] detail::KernelConfig segmented_reduce_config;
      error =
        CubDebug(segmented_reduce_config.Init(segmented_reduce_kernel, policy.SegmentedReduce(), launcher_factory));
      if (cudaSuccess != error)
      {
        break;
      }

      const auto num_segments_per_invocation =
        static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<::cuda::std::int32_t>::max());
      const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, num_segments_per_invocation);

      for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
      {
        const auto current_seg_offset = invocation_index * num_segments_per_invocation;
        const auto num_current_segments =
          ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);

        // Log device_reduce_sweep_kernel configuration
        _CUB_LOG_KERNEL_LAUNCH(
          "SegmentedDeviceReduceKernel",
          static_cast<int>(num_current_segments),
          1,
          1,
          policy.SegmentedReduce().ThreadsPerBlock(),
          0,
          stream,
          ", SM occupancy: %d",
          segmented_reduce_config.sm_occupancy);

        // Invoke DeviceSegmentedReduceKernel
        launcher_factory(static_cast<::cuda::std::uint32_t>(num_current_segments),
                         policy.SegmentedReduce().ThreadsPerBlock(),
                         0,
                         stream)
          .doit(segmented_reduce_kernel,
                d_in,
                d_out,
                d_begin_offsets,
                d_end_offsets,
                static_cast<int>(num_current_segments),
                reduction_op,
                init,
                0);

        // Check for failure to launch
        error = CubDebug(cudaPeekAtLastError());
        if (cudaSuccess != error)
        {
          break;
        }

        if (invocation_index + 1 < num_invocations)
        {
          d_out += num_current_segments;
          d_begin_offsets += num_current_segments;
          d_end_offsets += num_current_segments;
        }

        // Sync the stream if specified to flush runtime errors
        error = CubDebug(detail::DebugSyncStream(stream));
        if (cudaSuccess != error)
        {
          break;
        }
      }
    } while (false);

    return error;
  }

  /// Invocation
  template <typename ActivePolicyT>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t Invoke(ActivePolicyT policy = {})
  {
    auto wrapped_policy = detail::reduce::MakeReducePolicyWrapper(policy);
    // Force kernel code-generation in all compiler passes
    return InvokePasses(kernel_source.SegmentedReduceKernel(), wrapped_policy);
  }

  //---------------------------------------------------------------------------
  // Dispatch entrypoints
  //---------------------------------------------------------------------------

  /**
   * @brief Internal dispatch routine for computing a device-wide reduction
   *
   * @param[in] d_temp_storage
   *   Device-accessible allocation of temporary storage. When `nullptr`, the
   *   required allocation size is written to `temp_storage_bytes` and no work
   *   is done.
   *
   * @param[in,out] temp_storage_bytes
   *   Reference to size in bytes of `d_temp_storage` allocation
   *
   * @param[in] d_in
   *   Pointer to the input sequence of data items
   *
   * @param[out] d_out
   *   Pointer to the output aggregate
   *
   * @param[in] num_segments
   *   The number of segments that comprise the sorting data
   *
   * @param[in] d_begin_offsets
   *   Random-access input iterator to the sequence of beginning offsets of
   *   length `num_segments`, such that `d_begin_offsets[i]` is the first
   *   element of the *i*<sup>th</sup> data segment in `d_keys_*` and
   *   `d_values_*`
   *
   * @param[in] d_end_offsets
   *   Random-access input iterator to the sequence of ending offsets of length
   *   `num_segments`, such that `d_end_offsets[i] - 1` is the last element of
   *   the *i*<sup>th</sup> data segment in `d_keys_*` and `d_values_*`.
   *   If `d_end_offsets[i] - 1 <= d_begin_offsets[i]`, the *i*<sup>th</sup> is
   *   considered empty.
   *
   * @param[in] reduction_op
   *   Binary reduction functor
   *
   * @param[in] init
   *   The initial value of the reduction
   *
   * @param[in] stream
   *   **[optional]** CUDA stream to launch kernels within.
   *   Default is stream<sub>0</sub>.
   */
  template <typename MaxPolicyT = typename PolicyHub::MaxPolicy>
  CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE static cudaError_t Dispatch(
    void* d_temp_storage,
    size_t& temp_storage_bytes,
    InputIteratorT d_in,
    OutputIteratorT d_out,
    ::cuda::std::int64_t num_segments,
    BeginOffsetIteratorT d_begin_offsets,
    EndOffsetIteratorT d_end_offsets,
    ReductionOpT reduction_op,
    InitValueT init,
    cudaStream_t stream,
    KernelSource kernel_source             = {},
    KernelLauncherFactory launcher_factory = {},
    MaxPolicyT max_policy                  = {})
  {
    if (num_segments <= 0)
    {
      return cudaSuccess;
    }

    cudaError error = cudaSuccess;

    do
    {
      // Get PTX version
      int ptx_version = 0;
      error           = CubDebug(launcher_factory.PtxVersion(ptx_version));
      if (cudaSuccess != error)
      {
        break;
      }

      // Create dispatch functor
      DispatchSegmentedReduce dispatch(
        d_temp_storage,
        temp_storage_bytes,
        d_in,
        d_out,
        num_segments,
        d_begin_offsets,
        d_end_offsets,
        reduction_op,
        init,
        stream,
        ptx_version,
        kernel_source,
        launcher_factory);

      // Dispatch to chained policy
      error = CubDebug(max_policy.Invoke(ptx_version, dispatch));
      if (cudaSuccess != error)
      {
        break;
      }
    } while (false);

    return error;
  }
};

namespace detail::segmented_reduce
{
// DeviceAdaptiveSegmentedReduceKernel currently requires:
//  - sizeof(T) == 4 or 8 (16-byte coalesced load via cuda::vector_type)
//  - T is arithmetic (vector loads, atomics)
//  - OutputIteratorT == T* (cuda::atomic_ref needs an lvalue T)
//  - ReductionOpT is plus<>, minimum<>, or maximum<> (atomic fallback handles only these)
//  - AccumT == T (the kernel accumulates in T throughout)
//  - sizeof(OffsetT) == 4 (the kernel uses int for offsets internally)
template <typename T,
          typename OutputIteratorT,
          typename ReductionOpT,
          typename AccumT,
          typename OffsetT>
inline constexpr bool is_adaptive_segmented_reduce_compatible_v =
  (sizeof(T) == 4 || sizeof(T) == 8) //
  && ::cuda::std::is_arithmetic_v<T> //
  && ::cuda::std::is_same_v<OutputIteratorT, T*> //
  && (::cuda::__is_cuda_std_plus_v<ReductionOpT> //
      || ::cuda::__is_cuda_minimum_v<ReductionOpT> //
      || ::cuda::__is_cuda_maximum_v<ReductionOpT>) //
  &&::cuda::std::is_same_v<AccumT, T> //
  && (sizeof(OffsetT) == 4);

template <class PolicySelector,
          class InputIteratorT,
          class OutputIteratorT,
          class BeginOffsetIteratorT,
          class EndOffsetIteratorT,
          class ReductionOpT,
          class InitValueT>
struct DeviceAdaptiveSegmentedReduceKernelSource
{
  using T = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;

  _CCCL_HOST_DEVICE_API static constexpr adaptive_segmented_reduce_policy GetPolicy(::cuda::compute_capability cc)
  {
    return select_policy<PolicySelector>(cc);
  }

  CUB_DEFINE_KERNEL_GETTER(
    ReduceKernel,
    DeviceAdaptiveSegmentedReduceKernel<
      PolicySelector,
      InputIteratorT,
      OutputIteratorT,
      BeginOffsetIteratorT,
      EndOffsetIteratorT,
      ReductionOpT,
      InitValueT>)

  static constexpr int FillItemsPerThread =
    adaptive_segmented_reduce_default_policy_selector{}(::cuda::compute_capability{}).items_per_thread;

  CUB_DEFINE_KERNEL_GETTER(FillKernel, AdaptiveSegmentedReduceFillKernel<T, FillItemsPerThread>)
};

template <class AdaptiveKernelSource,
          class InputIteratorT,
          class OutputIteratorT,
          class BeginOffsetIteratorT,
          class EndOffsetIteratorT,
          class ReductionOpT,
          class InitValueT,
          class KernelLauncherFactory>
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE cudaError_t invoke_adaptive_segmented_reduce(
  InputIteratorT d_in,
  OutputIteratorT d_out,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  int num_segments,
  ReductionOpT reduction_op,
  InitValueT init,
  ::cuda::compute_capability cc,
  cudaStream_t stream,
  KernelLauncherFactory& launcher_factory,
  AdaptiveKernelSource /*kernel_source*/ = {})
{
  using KernelSource = AdaptiveKernelSource;
  using T            = typename KernelSource::T;
  const auto policy  = KernelSource::GetPolicy(cc);
  const int THREADS  = policy.threads_per_block;

  int num_sms = 0;
  if (const auto error = CubDebug(launcher_factory.MultiProcessorCount(num_sms)))
  {
    return error;
  }

  int blocks_per_sm = 0;
  if (const auto error =
        CubDebug(launcher_factory.MaxSmOccupancy(blocks_per_sm, KernelSource::ReduceKernel(), THREADS)))
  {
    return error;
  }

  const int grid = ::cuda::std::max(1, blocks_per_sm * num_sms);

  // Initialize the output to cover two cases:
  // 1) Empty segments not visited by the kernel; and
  // 2) Segments shared across threads, which will use an atomic.
  constexpr int V              = 16 / sizeof(T);
  constexpr int ItemsPerThread = KernelSource::FillItemsPerThread;
  const int fill_grid =
    ::cuda::std::max(1, static_cast<int>(::cuda::ceil_div(num_segments, THREADS * ItemsPerThread * V)));
  if (const auto error = CubDebug(
        launcher_factory(static_cast<::cuda::std::uint32_t>(fill_grid), THREADS, 0, stream)
          .doit(KernelSource::FillKernel(), static_cast<T*>(d_out), num_segments, static_cast<T>(init))))
  {
    return error;
  }

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  // Launch the reduction as a PDL-dependent kernel to overlap with the fill.
  if (const auto error = CubDebug(
        launcher_factory(static_cast<::cuda::std::uint32_t>(grid), THREADS, 0, stream, /*dependent_launch=*/true)
          .doit(
            KernelSource::ReduceKernel(), d_in, d_out, d_begin_offsets, d_end_offsets, num_segments, reduction_op, init)))
  {
    return error;
  }

  if (const auto error = CubDebug(cudaPeekAtLastError()))
  {
    return error;
  }

  return CubDebug(detail::DebugSyncStream(stream));
}

// select the accumulator type using an overload set, so __accumulator_t is not instantiated when
// an overriding accumulator type is present. This is needed by CCCL.C.
template <typename InputIteratorT,
          typename InitValueT,
          typename ReductionOpT,
          typename InputT = ::cuda::std::iter_value_t<InputIteratorT>>
_CCCL_HOST_DEVICE_API auto select_segmented_accum_t(use_default*) -> ::cuda::std::__accumulator_t<
  ReductionOpT,
  InputT,
  ::cuda::std::conditional_t<::cuda::std::is_same_v<InitValueT, reduce::no_init_t>, InputT, InitValueT>>;

template <typename InputIteratorT,
          typename InitValueT,
          typename ReductionOpT,
          typename OverrideAccumT,
          ::cuda::std::enable_if_t<!::cuda::std::is_same_v<OverrideAccumT, use_default>, int> = 0>
_CCCL_HOST_DEVICE_API auto select_segmented_accum_t(OverrideAccumT*) -> OverrideAccumT;

template <
  typename OverrideAccumT  = use_default,
  typename OverrideOffsetT = use_default,
  typename InputIteratorT,
  typename OutputIteratorT,
  typename BeginOffsetIteratorT,
  typename EndOffsetIteratorT,
  // need to evaluate common_iterator_value lazily. This is needed by CCCL.C.
  typename OffsetT = typename ::cuda::std::conditional_t<::cuda::std::is_same_v<OverrideOffsetT, use_default>,
                                                         common_iterator_value<BeginOffsetIteratorT, EndOffsetIteratorT>,
                                                         ::cuda::std::type_identity<OverrideOffsetT>>::type,
  typename ReductionOpT,
  typename InitValueT = non_void_value_t<OutputIteratorT, it_value_t<InputIteratorT>>,
  typename AccumT =
    decltype(select_segmented_accum_t<InputIteratorT, InitValueT, ReductionOpT>(static_cast<OverrideAccumT*>(nullptr))),
  typename PolicySelector = policy_selector_from_types<AccumT, OffsetT, ReductionOpT>,
  typename KernelSource   = DeviceSegmentedReduceKernelSource<
    PolicySelector,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT,
    OffsetT,
    ReductionOpT,
    InitValueT,
    AccumT>,
  typename AdaptiveKernelSource = DeviceAdaptiveSegmentedReduceKernelSource<
    adaptive_segmented_reduce_default_policy_selector,
    InputIteratorT,
    OutputIteratorT,
    BeginOffsetIteratorT,
    EndOffsetIteratorT,
    ReductionOpT,
    InitValueT>,
  typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  BeginOffsetIteratorT d_begin_offsets,
  EndOffsetIteratorT d_end_offsets,
  ReductionOpT reduction_op,
  InitValueT init,
  size_t max_segment_size,
  cudaStream_t stream,
  PolicySelector policy_selector              = {},
  KernelSource kernel_source                  = {},
  AdaptiveKernelSource adaptive_kernel_source = {},
  KernelLauncherFactory launcher_factory      = {})
{
  if (num_segments <= 0)
  {
    return cudaSuccess;
  }

  // Get CC
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  // Use the adaptive kernel for compatible types/operators and assumptions.
  //
  // The implementation requires that d_end_offsets is monotonically increasing,
  // and there is currently no way for a user to assert this is true. For now,
  // dispatch only when d_begin_offsets + 1 == d_end_offsets.
  using InputT = typename ::cuda::std::iterator_traits<InputIteratorT>::value_type;
  if constexpr (is_adaptive_segmented_reduce_compatible_v<InputT, OutputIteratorT, ReductionOpT, AccumT, OffsetT>)
  {
    // Force NVCC to register adaptive kernel symbols at dispatch() instantiation time.
    [[maybe_unused]] auto _adaptive_reduce_kern = AdaptiveKernelSource::ReduceKernel();
    [[maybe_unused]] auto _adaptive_fill_kern   = AdaptiveKernelSource::FillKernel();

    NV_IF_TARGET(
      NV_IS_HOST,
      (if (cc.get() >= 90
           && num_segments <= static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<int>::max())) {
        if constexpr (::cuda::std::is_pointer_v<BeginOffsetIteratorT> && ::cuda::std::is_pointer_v<EndOffsetIteratorT>)
        {
          if (d_begin_offsets + 1 == d_end_offsets)
          {
            if (d_temp_storage == nullptr)
            {
              temp_storage_bytes = 1;
              return cudaSuccess;
            }
            return invoke_adaptive_segmented_reduce<AdaptiveKernelSource>(
              d_in,
              d_out,
              d_begin_offsets,
              d_end_offsets,
              static_cast<int>(num_segments),
              reduction_op,
              init,
              cc,
              stream,
              launcher_factory,
              adaptive_kernel_source);
          }
        }
      }))
  }

  const SegmentedReducePolicy active_policy = policy_selector(cc);
  detail::log_dispatch("DeviceSegmentedReduce (variable size)", cc, active_policy);

  // Compute segments_per_block based on max_segment_size hint
  int segments_per_block = 1;
  if (max_segment_size != 0)
  {
    if (::cuda::in_range(
          max_segment_size, static_cast<size_t>(1), static_cast<size_t>(active_policy.small_reduce.items_per_tile())))
    {
      segments_per_block = active_policy.small_reduce.segments_per_block();
    }
    else if (::cuda::in_range(max_segment_size,
                              static_cast<size_t>(1),
                              static_cast<size_t>(active_policy.medium_reduce.items_per_tile())))
    {
      segments_per_block = active_policy.medium_reduce.segments_per_block();
    }
  }
  if (d_temp_storage == nullptr)
  {
    temp_storage_bytes = 1;
    return cudaSuccess;
  }

  // Get SM occupancy for the segmented reduce kernel (only needed for logging)
  [[maybe_unused]] int sm_occupancy{};
#ifndef CUB_DEBUG_LOG
  if (logging_enabled())
#endif // CUB_DEBUG_LOG
  {
    if (const auto error = CubDebug(launcher_factory.MaxSmOccupancy(
          sm_occupancy, kernel_source.SegmentedReduceKernel(), active_policy.large_reduce.threads_per_block)))
    {
      return error;
    }
  }

  const auto num_segments_per_invocation =
    static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<::cuda::std::int32_t>::max());
  const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, num_segments_per_invocation);

  for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
  {
    const auto current_seg_offset   = invocation_index * num_segments_per_invocation;
    const auto num_current_segments = ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);

    // Log device_reduce_sweep_kernel configuration
    _CUB_LOG_KERNEL_LAUNCH(
      "SegmentedDeviceReduceKernel",
      static_cast<int>(num_current_segments),
      1,
      1,
      active_policy.large_reduce.threads_per_block,
      0,
      stream,
      ", SM occupancy: %d",
      sm_occupancy);

    // Invoke DeviceSegmentedReduceKernel
    const auto num_blocks =
      ::cuda::ceil_div(num_current_segments, static_cast<::cuda::std::int64_t>(segments_per_block));
    if (const auto error = CubDebug(
          launcher_factory(
            static_cast<::cuda::std::uint32_t>(num_blocks), active_policy.large_reduce.threads_per_block, 0, stream)
            .doit(kernel_source.SegmentedReduceKernel(),
                  d_in,
                  d_out,
                  d_begin_offsets,
                  d_end_offsets,
                  static_cast<int>(num_current_segments),
                  reduction_op,
                  init,
                  max_segment_size)))
    {
      return error;
    }

    // Check for failure to launch
    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    if (invocation_index + 1 < num_invocations)
    {
      d_out += num_current_segments;
      d_begin_offsets += num_current_segments;
      d_end_offsets += num_current_segments;
    }

    // Sync the stream if specified to flush runtime errors
    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
  }

  return cudaSuccess;
}

// @brief Functor to generate a key-value pair from an index and value
template <typename Iterator, typename OutputValueT>
struct generate_idx_value
{
private:
  Iterator it;
  int segment_size;

public:
  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE generate_idx_value(Iterator it, int segment_size)
      : it(it)
      , segment_size(segment_size)
  {}

  _CCCL_HOST_DEVICE _CCCL_FORCEINLINE auto operator()(::cuda::std::int64_t idx) const
  {
    return ::cuda::std::pair<int, OutputValueT>(static_cast<int>(idx % segment_size), it[idx]);
  }
};

template <typename PolicySelector,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT,
          typename AccumT>
struct DeviceFixedSizeSegmentedReduceKernelSource
{
  // PolicySelector must be stateless, so we can pass the type to the kernel
  static_assert(::cuda::std::is_empty_v<PolicySelector>);

  CUB_DEFINE_KERNEL_GETTER(
    FixedSizeSegmentedReduceKernel,
    DeviceFixedSizeSegmentedReduceKernel<
      PolicySelector,
      InputIteratorT,
      OutputIteratorT,
      OffsetT,
      ReductionOpT,
      InitValueT,
      AccumT>)

  CUB_DEFINE_KERNEL_GETTER(
    FixedSizeSegmentedReduceKernelFinal,
    DeviceFixedSizeSegmentedReduceKernel<PolicySelector, AccumT*, OutputIteratorT, OffsetT, ReductionOpT, InitValueT, AccumT>)

  CUB_RUNTIME_FUNCTION static constexpr ::cuda::std::size_t AccumSize()
  {
    return sizeof(AccumT);
  }
};

template <typename OverrideAccumT = use_default,
          typename InputIteratorT,
          typename OutputIteratorT,
          typename OffsetT,
          typename ReductionOpT,
          typename InitValueT     = non_void_value_t<OutputIteratorT, it_value_t<InputIteratorT>>,
          typename AccumT         = decltype(select_segmented_accum_t<InputIteratorT, InitValueT, ReductionOpT>(
            static_cast<OverrideAccumT*>(nullptr))),
          typename PolicySelector = policy_selector_from_types<AccumT, OffsetT, ReductionOpT>,
          typename KernelSource   = DeviceFixedSizeSegmentedReduceKernelSource<
            PolicySelector,
            InputIteratorT,
            OutputIteratorT,
            OffsetT,
            ReductionOpT,
            InitValueT,
            AccumT>,
          typename KernelLauncherFactory = CUB_DETAIL_DEFAULT_KERNEL_LAUNCHER_FACTORY,
          ::cuda::std::enable_if_t<::cuda::std::is_arithmetic_v<OffsetT>, int> = 0>
#if _CCCL_HAS_CONCEPTS()
  requires segmented_reduce_policy_selector<PolicySelector>
#endif // _CCCL_HAS_CONCEPTS()
CUB_RUNTIME_FUNCTION _CCCL_FORCEINLINE auto dispatch_fixed_size(
  void* d_temp_storage,
  size_t& temp_storage_bytes,
  InputIteratorT d_in,
  OutputIteratorT d_out,
  ::cuda::std::int64_t num_segments,
  OffsetT segment_size,
  ReductionOpT reduction_op,
  InitValueT init,
  cudaStream_t stream,
  PolicySelector policy_selector         = {},
  KernelSource kernel_source             = {},
  KernelLauncherFactory launcher_factory = {})
{
  if (num_segments <= 0)
  {
    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
    }
    return cudaSuccess;
  }

  // Get CC
  ::cuda::compute_capability cc{};
  if (const auto error = CubDebug(launcher_factory.PtxComputeCap(cc)))
  {
    return error;
  }

  const SegmentedReducePolicy active_policy = policy_selector(cc);
  detail::log_dispatch("DeviceSegmentedReduce (fixed size)", cc, active_policy);

  const auto tile_size = active_policy.large_reduce.threads_per_block * active_policy.large_reduce.items_per_thread;

  // Single-phase: segment fits in one tile
  if (segment_size < tile_size)
  {
    int segments_per_block = 1;

    if (segment_size <= active_policy.small_reduce.items_per_tile())
    {
      segments_per_block = active_policy.small_reduce.segments_per_block();
    }
    else if (segment_size <= active_policy.medium_reduce.items_per_tile())
    {
      segments_per_block = active_policy.medium_reduce.segments_per_block();
    }

    if (d_temp_storage == nullptr)
    {
      temp_storage_bytes = 1;
      return cudaSuccess;
    }

    constexpr auto num_segments_per_invocation =
      static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<::cuda::std::int32_t>::max());
    const ::cuda::std::int64_t num_invocations = ::cuda::ceil_div(num_segments, num_segments_per_invocation);

    for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
    {
      const auto current_seg_offset = invocation_index * num_segments_per_invocation;
      const auto num_current_segments =
        ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);
      const auto num_current_blocks = ::cuda::ceil_div(num_current_segments, segments_per_block);

      if (const auto error = CubDebug(
            launcher_factory(static_cast<::cuda::std::int32_t>(num_current_blocks),
                             active_policy.large_reduce.threads_per_block,
                             0,
                             stream)
              .doit(kernel_source.FixedSizeSegmentedReduceKernel(),
                    d_in,
                    d_out,
                    segment_size,
                    static_cast<::cuda::std::int32_t>(num_current_segments),
                    reduction_op,
                    init,
                    static_cast<AccumT*>(nullptr),
                    0,
                    0)))
      {
        return error;
      }

      d_in += num_segments_per_invocation * segment_size; // NOLINT(bugprone-misplaced-widening-cast)
      d_out += num_segments_per_invocation;

      if (const auto error = CubDebug(cudaPeekAtLastError()))
      {
        return error;
      }

      if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
      {
        return error;
      }
    }

    return cudaSuccess;
  }

  // Two-phase: segment spans multiple tiles
  const auto tiles_per_segment = static_cast<int>(::cuda::ceil_div(segment_size, tile_size));

  const auto max_tiles_per_invocation =
    static_cast<::cuda::std::int64_t>(::cuda::std::numeric_limits<::cuda::std::int32_t>::max());
  const auto max_segments_per_invocation = max_tiles_per_invocation / tiles_per_segment;
  const auto num_invocations             = ::cuda::ceil_div(num_segments, max_segments_per_invocation);
  const auto num_segments_per_invocation = ::cuda::std::min(max_segments_per_invocation, num_segments);
  const auto tiles_per_invocation        = num_segments_per_invocation * tiles_per_segment;

  // Temporary storage allocation requirements
  void* allocations[1]             = {};
  const size_t allocation_sizes[1] = {static_cast<size_t>(tiles_per_invocation) * kernel_source.AccumSize()};

  if (const auto error =
        CubDebug(detail::alias_temporaries(d_temp_storage, temp_storage_bytes, allocations, allocation_sizes)))
  {
    return error;
  }

  if (d_temp_storage == nullptr)
  {
    return cudaSuccess;
  }

  AccumT* d_block_reductions = static_cast<AccumT*>(allocations[0]); // NOLINT(misc-const-correctness)

  for (::cuda::std::int64_t invocation_index = 0; invocation_index < num_invocations; invocation_index++)
  {
    const auto current_seg_offset   = invocation_index * num_segments_per_invocation;
    const auto num_current_segments = ::cuda::std::min(num_segments_per_invocation, num_segments - current_seg_offset);
    const auto num_current_blocks   = static_cast<::cuda::std::int32_t>(num_current_segments * tiles_per_segment);

    // Phase 1: partial reductions
    if (const auto error = CubDebug(
          launcher_factory(num_current_blocks, active_policy.large_reduce.threads_per_block, 0, stream)
            .doit(kernel_source.FixedSizeSegmentedReduceKernel(),
                  d_in,
                  d_out,
                  segment_size,
                  num_current_blocks,
                  reduction_op,
                  init,
                  d_block_reductions,
                  tile_size,
                  tiles_per_segment)))
    {
      return error;
    }

    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }

    // Phase 2: final reduction of partial results
    const int final_segment_size = tiles_per_segment;
    int final_segments_per_block = 1;

    if (final_segment_size <= active_policy.small_reduce.items_per_tile())
    {
      final_segments_per_block = active_policy.small_reduce.segments_per_block();
    }
    else if (final_segment_size <= active_policy.medium_reduce.items_per_tile())
    {
      final_segments_per_block = active_policy.medium_reduce.segments_per_block();
    }

    const auto final_num_current_blocks = ::cuda::ceil_div(num_current_segments, final_segments_per_block);

    if (const auto error = CubDebug(
          launcher_factory(static_cast<::cuda::std::int32_t>(final_num_current_blocks),
                           active_policy.large_reduce.threads_per_block,
                           0,
                           stream)
            .doit(kernel_source.FixedSizeSegmentedReduceKernelFinal(),
                  d_block_reductions,
                  d_out,
                  final_segment_size,
                  static_cast<::cuda::std::int32_t>(num_current_segments),
                  reduction_op,
                  init,
                  static_cast<AccumT*>(nullptr),
                  0,
                  0)))
    {
      return error;
    }

    d_in += num_segments_per_invocation * segment_size; // NOLINT(bugprone-misplaced-widening-cast)
    d_out += num_segments_per_invocation;

    if (const auto error = CubDebug(cudaPeekAtLastError()))
    {
      return error;
    }

    if (const auto error = CubDebug(detail::DebugSyncStream(stream)))
    {
      return error;
    }
  }

  return cudaSuccess;
}
} // namespace detail::segmented_reduce

CUB_NAMESPACE_END
