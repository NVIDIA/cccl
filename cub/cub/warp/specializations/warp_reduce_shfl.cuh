// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/detail/key_value_pair.cuh>
#include <cub/detail/null_type.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>
#include <cub/warp/specializations/warp_reduce_shfl_ptx.cuh>
#include <cub/warp/specializations/warp_redux.cuh>

#include <cuda/__cmath/ilog.h>
#include <cuda/__cmath/pow2.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__ptx/instructions/get_sreg.h>
#include <cuda/__type_traits/is_trivially_copyable.h>
#include <cuda/__utility/static_for.h>
#include <cuda/__warp/warp_shuffle.h>
#include <cuda/std/__bit/countr.h>
#include <cuda/std/__functional/operations.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/cstdint>

#include <nv/target>

CUB_NAMESPACE_BEGIN
namespace detail
{
template <typename ReductionOp, typename T>
inline constexpr bool is_warp_vote_any_op =
  ::cuda::std::is_same_v<T, bool> && ::cuda::__is_cuda_std_logical_or_v<ReductionOp, T>;

template <typename ReductionOp, typename T>
inline constexpr bool is_warp_vote_all_op =
  ::cuda::std::is_same_v<T, bool> && ::cuda::__is_cuda_std_logical_and_v<ReductionOp, T>;

// WarpReduceShfl provides SHFL-based reduction of items partitioned across a warp
template <typename T, int LogicalWarpThreads>
struct WarpReduceShfl
{
  static_assert(::cuda::is_power_of_two(LogicalWarpThreads), "LogicalWarpThreads must be a power of two");

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool is_full_warp = (LogicalWarpThreads == warp_threads);

  /// The number of warp reduction steps
  static constexpr int steps = ::cuda::ilog2(LogicalWarpThreads);

  /// Shared memory storage layout type
  using TempStorage = NullType;

  /// Lane index in logical warp
  int lane_id;

  /// Logical warp index in 32-thread physical warp
  int warp_id;

  /// 32-thread physical warp member mask of logical warp
  ::cuda::std::uint32_t member_mask;

  _CCCL_DEVICE_API _CCCL_FORCEINLINE WarpReduceShfl(TempStorage& /*temp_storage*/)
      : lane_id(static_cast<int>(::cuda::ptx::get_sreg_laneid()))
      , warp_id(is_full_warp ? 0 : (lane_id / LogicalWarpThreads))
      , member_mask(CUB_NS_QUALIFIER::WarpMask<LogicalWarpThreads>(warp_id))
  {
    if constexpr (!is_full_warp)
    {
      lane_id = lane_id % LogicalWarpThreads;
    }
  }

  /**
   * @brief Reduction (specialized for swizzled ReduceByKeyOp<::cuda::std::plus<>> across
   *        KeyValuePair<KeyT, ValueT> types)
   *
   * @param[in] input
   *   Calling thread's input item
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename ValueT, typename KeyT>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE KeyValuePair<KeyT, ValueT> ReduceStep(
    const KeyValuePair<KeyT, ValueT> input,
    const SwizzleScanOp<ReduceByKeyOp<::cuda::std::plus<>>> /*reduction_op*/,
    const int last_lane,
    const int offset)
  {
    KeyValuePair<KeyT, ValueT> output;

    KeyT other_key = CUB_NS_QUALIFIER::ShuffleDown<LogicalWarpThreads>(input.key, offset, last_lane, member_mask);

    output.key   = input.key;
    output.value = ReduceStep(input.value, ::cuda::std::plus<>{}, last_lane, offset);

    if (input.key != other_key)
    {
      output.value = input.value;
    }

    return output;
  }

  /**
   * @brief Reduction (specialized for swizzled ReduceBySegmentOp<cuda::std::plus<>> across
   *        KeyValuePair<OffsetT, ValueT> types)
   *
   * @param[in] input
   *   Calling thread's input item.
   *
   * @param[in] reduction_op
   *   Binary reduction operator
   *
   * @param[in] last_lane
   *   Index of last lane in segment
   *
   * @param[in] offset
   *   Up-offset to pull from
   */
  template <typename ValueT, typename OffsetT>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE KeyValuePair<OffsetT, ValueT> ReduceStep(
    const KeyValuePair<OffsetT, ValueT> input,
    const SwizzleScanOp<ReduceBySegmentOp<::cuda::std::plus<>>> /*reduction_op*/,
    const int last_lane,
    const int offset)
  {
    // The reduction supplies both output fields.
    // NOLINTNEXTLINE(cppcoreguidelines-pro-type-member-init)
    KeyValuePair<OffsetT, ValueT> output;

    output.value = ReduceStep(input.value, ::cuda::std::plus<>{}, last_lane, offset);
    output.key   = ReduceStep(input.key, ::cuda::std::plus<>{}, last_lane, offset);

    if (input.key > 0)
    {
      output.value = input.value;
    }

    return output;
  }

  // Single step of a warp reduction
  // last_lane: Index of last lane in segment
  // offset: Up-offset to pull from
  template <typename Tp, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE Tp
  ReduceStep(const Tp input, ReductionOp reduction_op, const int last_lane, const int offset)
  {
    if constexpr (is_shfl_down_op_supported_v<ReductionOp, Tp>)
    {
      constexpr unsigned shfl_c_base = (warp_threads - LogicalWarpThreads) << 8;
      const int shfl_c               = last_lane | shfl_c_base;
      return CUB_NS_QUALIFIER::detail::shfl_down_op(input, reduction_op, offset, shfl_c, member_mask);
    }
    else
    {
      Tp output = input;
      Tp tmp    = CUB_NS_QUALIFIER::ShuffleDown<LogicalWarpThreads>(output, offset, last_lane, member_mask);
      // Perform reduction op if valid
      if (offset + lane_id <= last_lane)
      {
        output = reduction_op(input, tmp);
      }
      return output;
    }
  }

  // Unrolls the reduction steps
  template <typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void ShuffleReduce(T& input, ReductionOp reduction_op, const int last_lane)
  {
    // capture clause prevents regressions
    ::cuda::static_for<steps>([&, value = input, reduction_op](const auto step) mutable {
      value = ReduceStep(value, reduction_op, last_lane, 1u << step);
      input = value;
    });
  }

  // Unrolls the reduction steps when all lanes are valid: no range check, only lane 0 holds the result
  template <typename ReductionOp>
  _CCCL_DEVICE_API _CCCL_FORCEINLINE void FullShuffleReduce(T& input, ReductionOp reduction_op)
  {
    ::cuda::static_for<steps>([&](const auto step) {
      const T tmp = ::cuda::device::warp_shuffle_down<LogicalWarpThreads>(input, 1u << step, member_mask).data;
      input       = reduction_op(input, tmp);
    });
  }

  // AllValidLanes: Whether all lanes in each warp are contributing a valid fold of items
  template <bool AllValidLanes, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
  Reduce(const T input, const int valid_items, ReductionOp reduction_op)
  {
    // Dispatch to more efficient intrinsics when applicable
    [[maybe_unused]] const int last_lane = (AllValidLanes) ? LogicalWarpThreads - 1 : valid_items - 1;

    if constexpr (is_warp_vote_any_op<ReductionOp, T>)
    {
      return __any_sync(member_mask, input && (AllValidLanes || lane_id <= last_lane));
    }
    else if constexpr (is_warp_vote_all_op<ReductionOp, T>)
    {
      return __all_sync(member_mask, input || (!AllValidLanes && lane_id > last_lane));
    }
    else
    {
      constexpr bool has_identity = ::cuda::has_identity_element_v<ReductionOp, T>;
      T reduce_value              = input;
      if constexpr (is_full_warp && (AllValidLanes || has_identity) && is_warp_redux_op_supported<ReductionOp, T>)
      {
        if constexpr (!AllValidLanes)
        {
          reduce_value = lane_id <= last_lane ? input : ::cuda::identity_element<ReductionOp, T>();
        }
        if (const auto output = CUB_NS_QUALIFIER::detail::warp_redux(reduce_value, member_mask, reduction_op))
        {
          return *output;
        }
      }
      if constexpr (AllValidLanes && ::cuda::is_trivially_copyable_v<T>)
      {
        FullShuffleReduce(reduce_value, reduction_op);
      }
      else
      {
        // Keep the last_lane clamp: invalid lanes may not exist in an incomplete physical warp.
        ShuffleReduce(reduce_value, reduction_op, last_lane);
      }
      return reduce_value;
    }
  }

  // HeadSegmented: Whether flags indicate a segment-head or a segment-tail
  // flag:          Whether or not the current lane is a segment head/tail
  template <bool HeadSegmented, typename FlagT, typename ReductionOp>
  [[nodiscard]] _CCCL_DEVICE_API _CCCL_FORCEINLINE T
  SegmentedReduce(const T input, const FlagT flag, ReductionOp reduction_op)
  {
    // Get the start flags for each thread in the warp.
    unsigned warp_flags = __ballot_sync(member_mask, flag);
    // Convert to tail-segmented
    warp_flags >>= HeadSegmented;
    // Mask out the bits below the current thread
    warp_flags &= ::cuda::ptx::get_sreg_lanemask_ge();

    // Mask of physical lanes outside the logical warp and convert to logical lanemask
    if constexpr (!is_full_warp)
    {
      warp_flags = (warp_flags & member_mask) >> (warp_id * LogicalWarpThreads);
    }
    // Mask in the last lane of logical warp
    warp_flags |= 1u << (LogicalWarpThreads - 1);
    // Find the next set flag
    const int last_lane = ::cuda::std::countr_zero(warp_flags);
    T output            = input;
    ShuffleReduce(output, reduction_op, last_lane);
    return output;
  }
};
} // namespace detail
CUB_NAMESPACE_END
