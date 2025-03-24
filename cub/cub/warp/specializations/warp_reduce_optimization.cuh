/***********************************************************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2025, NVIDIA CORPORATION.  All rights reserved.
 *
 * Redistribution and use in source and binary forms, with or without modification, are permitted provided that the
 * following conditions are met:
 *     * Redistributions of source code must retain the above copyright
 *       notice, this list of conditions and the following disclaimer.
 *     * Redistributions in binary form must reproduce the above copyright
 *       notice, this list of conditions and the following disclaimer in the
 *       documentation and/or other materials provided with the distribution.
 *     * Neither the name of the NVIDIA CORPORATION nor the
 *       names of its contributors may be used to endorse or promote products
 *       derived from this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES,
 * INCLUDING, BUT NOT LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY,
 * OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN
 * CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **********************************************************************************************************************/

/**
 * @file
 * cub::WarpReduceSmem provides smem-based variants of parallel reduction of items partitioned
 * across a CUDA thread warp.
 */

#pragma once

#include <cub/config.cuh>

#include <cstdint>

#include "cuda/__functional/maximum.h"
#include "cuda/std/__internal/namespaces.h"
#include "cuda/std/__type_traits/integral_constant.h"
#include "cuda/std/__type_traits/make_unsigned.h"
#include <sys/types.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/thread/thread_load.cuh>
#include <cub/thread/thread_operators.cuh>
#include <cub/thread/thread_store.cuh>
#include <cub/util_type.cuh>

#include <cuda/ptx>

CUB_NAMESPACE_BEGIN
namespace detail
{

enum class WarpReduceMode
{
  SingleLogicalWarp,
  MultipleLogicalWarps
};

template <WarpReduceMode Mode>
using warp_reduce_mode_t = _CUDA_VSTD::integral_constant<WarpReduceMode, Mode>;

inline constexpr auto single_logical_warp    = warp_reduce_mode_t<WarpReduceMode::SingleLogicalWarp>{};
inline constexpr auto multiple_logical_warps = warp_reduce_mode_t<WarpReduceMode::MultipleLogicalWarps>{};

enum class WarpReduceResult
{
  AllLanes,
  SingleLane
};

template <WarpReduceResult Kind>
using warp_reduce_result_t = _CUDA_VSTD::integral_constant<WarpReduceResult, Kind>;

inline constexpr auto all_lanes_result   = warp_reduce_result_t<WarpReduceResult::AllLanes>{};
inline constexpr auto single_lane_result = warp_reduce_result_t<WarpReduceResult::SingleLane>{};

/**
 * @brief WarpReduceSmem provides smem-based variants of parallel reduction of items partitioned
 *        across a CUDA thread warp.
 *
 * @tparam T
 *   Data type being reduced
 *
 * @tparam LOGICAL_WARP_THREADS
 *   Number of threads per logical warp
 */
template <typename T, int LOGICAL_WARP_THREADS>
struct WarpReduceSmem
{
  /******************************************************************************
   * Constants and type definitions
   ******************************************************************************/

  /// Whether the logical warp size and the PTX warp size coincide
  static constexpr bool IS_ARCH_WARP = (LOGICAL_WARP_THREADS == warp_threads);

  /// Whether the logical warp size is a power-of-two
  static constexpr bool IS_POW_OF_TWO = PowerOfTwo<LOGICAL_WARP_THREADS>::VALUE;

  /// The number of warp reduction steps
  static constexpr int STEPS = Log2<LOGICAL_WARP_THREADS>::VALUE;

  /// The number of threads in half a warp
  static constexpr int HALF_WARP_THREADS = 1 << (STEPS - 1);

  /// The number of shared memory elements per warp
  static constexpr int WARP_SMEM_ELEMENTS = LOGICAL_WARP_THREADS + HALF_WARP_THREADS;

  /// FlagT status (when not using ballot)
  static constexpr auto UNSET = 0x0; // Is initially unset
  static constexpr auto SET   = 0x1; // Is initially set
  static constexpr auto SEEN  = 0x2; // Has seen another head flag from a successor peer

  /// Shared memory flag type
  using SmemFlag = unsigned char;

  /// Shared memory storage layout type (1.5 warps-worth of elements for each warp)
  struct _TempStorage
  {
    T reduce[WARP_SMEM_ELEMENTS];
    SmemFlag flags[WARP_SMEM_ELEMENTS];
  };

  // Alias wrapper allowing storage to be unioned
  struct TempStorage : Uninitialized<_TempStorage>
  {};

  /******************************************************************************
   * Thread fields
   ******************************************************************************/

  _TempStorage& temp_storage;
  unsigned int lane_id;
  unsigned int member_mask;

  /******************************************************************************
   * Construction
   ******************************************************************************/

  /// Constructor
  explicit _CCCL_DEVICE _CCCL_FORCEINLINE WarpReduceSmem(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
      , lane_id(IS_ARCH_WARP ? ::cuda::ptx::get_sreg_laneid() : ::cuda::ptx::get_sreg_laneid() % LOGICAL_WARP_THREADS)
      , member_mask(WarpMask<LOGICAL_WARP_THREADS>(::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS))
  {}

  /******************************************************************************
   * Interface
   ******************************************************************************/

  using sing

    /**
     * @brief Reduction
     *
     * @tparam ALL_LANES_VALID
     *   Whether all lanes in each warp are contributing a valid fold of items
     *
     * @param[in] input
     *   Calling thread's input
     *
     * @param[in] valid_items
     *   Total number of valid items across the logical warp
     *
     * @param[in] reduction_op
     *   Reduction operator
     */
    template <bool ALL_LANES_VALID, typename ReductionOp>
    _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, ReductionOp reduction_op, single_logical_warp)
  {
    constexpr bool is_cuda_std_plus =
      _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::plus<>> || _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::plus<T>>;
    constexpr bool is_cuda_maximum =
      _CUDA_VSTD::is_same_v<ReductionOp, ::cuda::maximum<>> || _CUDA_VSTD::is_same_v<ReductionOp, ::cuda::maximum<T>>;
    constexpr bool is_cuda_minimum =
      _CUDA_VSTD::is_same_v<ReductionOp, ::cuda::minimum<>> || _CUDA_VSTD::is_same_v<ReductionOp, ::cuda::minimum<T>>;
    constexpr bool is_cuda_std_bit_and = _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::bit_and<>>
                                      || _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::bit_and<T>>;
    constexpr bool is_cuda_std_bit_or = _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::bit_or<>>
                                     || _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::bit_or<T>>;
    constexpr bool is_cuda_std_bit_xor = _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::bit_xor<>>
                                      || _CUDA_VSTD::is_same_v<ReductionOp, _CUDA_VSTD::bit_xor<T>>;
    if constexpr (is_cuda_std_plus)
    {
      if constexpr (_CUDA_VSTD::is_same_v<T, bool>)
      {
        return _CUDA_VSTD::popcount(::__ballot_sync(member_mask, input));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t))
      {
        using unsigned_t = _CUDA_VSTD::_If<_CUDA_VSTD::is_signed_v<T>, int, uint32_t>;
        return static_cast<T>(::__reduce_add_sync(member_mask, static_cast<unsigned_t>(input)));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) == sizeof(uint64_t))
      {
        auto high           = static_cast<uint32_t>(static_cast<uint64_t>(input) >> 32u);
        auto low            = static_cast<uint32_t>(input);
        auto high_reduction = ::__reduce_add_sync(member_mask, high);
        auto mid_reduction  = ::__reduce_add_sync(member_mask, low >> (32u - 5u)); // carry out
        auto low_reduction  = ::__reduce_add_sync(member_mask, low);
        auto result_high    = high_reduction + mid_reduction;
        return static_cast<T>((static_cast<uint64_t>(result_high) << 32u) + low_reduction);
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) == sizeof(__uint128_t))
      {
        auto high           = static_cast<uint64_t>(static_cast<__uint128_t>(input) >> 64u);
        auto low            = static_cast<uint64_t>(input);
        auto high_reduction = this->Reduce(high, reduction_op);
        auto mid_reduction  = ::__reduce_add_sync(member_mask, static_cast<uint32_t>(low >> (64u - 5u))); // carry out
        auto low_reduction  = this->Reduce(low, reduction_op);
        auto result_high    = high_reduction + mid_reduction;
        return static_cast<T>((static_cast<__uint128_t>(result_high) << 64u) + low_reduction);
      }
    }
    else if constexpr (is_cuda_maximum)
    {
      using cast_t = _CUDA_VSTD::_If<_CUDA_VSTD::is_signed_v<T>, int, uint32_t>;
      if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t))
      {
        return static_cast<T>(::__reduce_max_sync(member_mask, static_cast<cast_t>(input)));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) == sizeof(uint64_t))
      {
        auto high     = static_cast<cast_t>(static_cast<uint64_t>(input) >> 32);
        auto high_max = ::__reduce_max_sync(member_mask, high);
        if (high_max == 0) // shortcut: max input is positive and < 2^32
        {
          return ::__reduce_max_sync(member_mask, static_cast<uint32_t>(input));
        }
        if (high_max > 0)
        {
          constexpr auto min_v = _CUDA_VSTD::numeric_limits<uint32_t>::min();
          auto low             = static_cast<uint32_t>(input);
          auto low_max         = ::__reduce_max_sync(member_mask, (high_max == high) ? low : min_v);
          return static_cast<T>(static_cast<uint64_t>(high_max) << 32 | low_max);
        }
        constexpr auto min_v = _CUDA_VSTD::numeric_limits<int32_t>::min();
        auto low             = static_cast<int32_t>(input);
        auto low_max         = ::__reduce_max_sync(member_mask, (high_max == high) ? low : min_v);
        return static_cast<T>(static_cast<uint64_t>(high_max) << 32 | low_max);
      }
    }
    else if constexpr (is_cuda_minimum)
    {
      using cast_t = _CUDA_VSTD::_If<_CUDA_VSTD::is_signed_v<T>, int, uint32_t>;
      if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t))
      {
        return static_cast<T>(::__reduce_min_sync(member_mask, static_cast<cast_t>(input)));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) == sizeof(uint64_t))
      {
        auto high     = static_cast<cast_t>(static_cast<uint64_t>(input) >> 32);
        auto high_min = ::__reduce_min_sync(member_mask, high);
        if (high_min == 0)
        {
          return ::__reduce_min_sync(member_mask, static_cast<cast_t>(input));
        }
        constexpr auto max_v = _CUDA_VSTD::numeric_limits<cast_t>::max();
        auto low             = static_cast<cast_t>(input);
        auto low_min         = ::__reduce_min_sync(member_mask, (high_min == high) ? low : max_v);
        return static_cast<T>(static_cast<uint64_t>(high_min) << 32 | low_min);
      }
    }
    else if constexpr (is_cuda_std_bit_and)
    {
      if constexpr (_CUDA_VSTD::is_same_v<T, bool>)
      {
        return ::__all_sync(member_mask, input);
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t))
      {
        return static_cast<T>(::__reduce_and_sync(member_mask, static_cast<uint32_t>(input)));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T>)
      {
        constexpr auto half_bits = _CUDA_VSTD::numeric_limits<T>::digits / 2u;
        using unsigned_t         = _CUDA_VSTD::make_unsigned_t<T>;
        using half_size_t        = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
        auto high                = static_cast<half_size_t>(static_cast<unsigned_t>(input) >> half_bits);
        auto low                 = static_cast<half_size_t>(input);
        auto high_reduction      = this->Reduce(high, reduction_op);
        auto low_reduction       = this->Reduce(low, reduction_op);
        return static_cast<T>((static_cast<unsigned_t>(high_reduction) << half_bits) & low_reduction);
      }
    }
    else if constexpr (is_cuda_std_bit_or)
    {
      if constexpr (_CUDA_VSTD::is_same_v<T, bool>)
      {
        return ::__any_sync(member_mask, input);
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t))
      {
        return static_cast<T>(::__reduce_or_sync(member_mask, static_cast<uint32_t>(input)));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T>)
      {
        constexpr auto half_bits = _CUDA_VSTD::numeric_limits<T>::digits / 2u;
        using unsigned_t         = _CUDA_VSTD::make_unsigned_t<T>;
        using half_size_t        = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
        auto high                = static_cast<half_size_t>(static_cast<unsigned_t>(input) >> half_bits);
        auto low                 = static_cast<half_size_t>(input);
        auto high_reduction      = this->Reduce(high, reduction_op);
        auto low_reduction       = this->Reduce(low, reduction_op);
        return static_cast<T>((static_cast<unsigned_t>(high_reduction) << half_bits) | low_reduction);
      }
    }
    else if constexpr (is_cuda_std_bit_xor)
    {
      if constexpr (_CUDA_VSTD::is_same_v<T, bool>)
      {
        return _CUDA_VSTD::popcount(::__ballot_sync(member_mask, input)) % 2u; // TODO: __reduce_xor_sync
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T> && sizeof(T) <= sizeof(uint32_t))
      {
        return static_cast<T>(::__reduce_xor_sync(member_mask, static_cast<uint32_t>(input)));
      }
      else if constexpr (_CUDA_VSTD::is_integral_v<T>)
      {
        constexpr auto half_bits = _CUDA_VSTD::numeric_limits<T>::digits / 2u;
        using unsigned_t         = _CUDA_VSTD::make_unsigned_t<T>;
        using half_size_t        = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
        auto high                = static_cast<half_size_t>(static_cast<unsigned_t>(input) >> half_bits);
        auto low                 = static_cast<half_size_t>(input);
        auto high_reduction      = this->Reduce(high, reduction_op);
        auto low_reduction       = this->Reduce(low, reduction_op);
        return static_cast<T>((static_cast<unsigned_t>(high_reduction) << half_bits) ^ low_reduction);
      }
    }
    else
    {
    }
  }

  template <typename T>
  auto split_integral(T input)
  {
    constexpr auto half_bits = _CUDA_VSTD::numeric_limits<T>::digits / 2u;
    using unsigned_t         = _CUDA_VSTD::make_unsigned_t<T>;
    using half_size_t        = _CUDA_VSTD::__make_nbit_uint_t<half_bits>;
    auto high                = static_cast<half_size_t>(static_cast<unsigned_t>(input) >> half_bits);
    auto low                 = static_cast<half_size_t>(input);
    return ::cuda::std::array<half_size_t, 2>{high, low};
  }

  template <bool ALL_LANES_VALID, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T Reduce(T input, int valid_items, ReductionOp reduction_op)
  {}

  /**
   * @brief Segmented reduction
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op)
  {
    return SegmentedReduce<HEAD_SEGMENTED>(input, flag, reduction_op, ::cuda::std::true_type());
  }

private:
  /******************************************************************************
   * Utility methods
   ******************************************************************************/

  //---------------------------------------------------------------------
  // Regular reduction
  //---------------------------------------------------------------------

  /**
   * @brief Reduction step
   *
   * @tparam ALL_LANES_VALID
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp, int STEP>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  ReduceStep(T input, int valid_items, ReductionOp reduction_op, constant_t<STEP> /*step*/)
  {
    constexpr int OFFSET = 1 << STEP;

    // Share input through buffer
    ThreadStore<STORE_VOLATILE>(&temp_storage.reduce[lane_id], input);

    __syncwarp(member_mask);

    // Update input if peer_addend is in range
    if ((ALL_LANES_VALID && IS_POW_OF_TWO) || ((lane_id + OFFSET) < valid_items))
    {
      T peer_addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage.reduce[lane_id + OFFSET]);
      input         = reduction_op(input, peer_addend);
    }

    __syncwarp(member_mask);

    return ReduceStep<ALL_LANES_VALID>(input, valid_items, reduction_op, constant_v<STEP + 1>);
  }

  /**
   * @brief Reduction step (terminate)
   *
   * @tparam ALL_LANES_VALID
   *   Whether all lanes in each warp are contributing a valid fold of items
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] valid_items
   *   Total number of valid items across the logical warp
   *
   * @param[in] reduction_op
   *   Reduction operator
   */
  template <bool ALL_LANES_VALID, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  ReduceStep(T input, int valid_items, ReductionOp /*reduction_op*/, constant_t<STEPS> /*step*/)
  {
    return input;
  }

  //---------------------------------------------------------------------
  // Segmented reduction
  //---------------------------------------------------------------------

  /**
   * @brief Ballot-based segmented reduce
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Reduction operator
   *
   * @param[in] has_ballot
   *   Marker type for whether the target arch has ballot functionality
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op, ::cuda::std::true_type /*has_ballot*/)
  {
    // Get the start flags for each thread in the warp.
    int warp_flags = __ballot_sync(member_mask, flag);

    if (!HEAD_SEGMENTED)
    {
      warp_flags <<= 1;
    }

    // Keep bits above the current thread.
    warp_flags &= ::cuda::ptx::get_sreg_lanemask_gt();

    // Accommodate packing of multiple logical warps in a single physical warp
    if (!IS_ARCH_WARP)
    {
      warp_flags >>= (::cuda::ptx::get_sreg_laneid() / LOGICAL_WARP_THREADS) * LOGICAL_WARP_THREADS;
    }

    // Find next flag
    int next_flag = __clz(__brev(warp_flags));

    // Clip the next segment at the warp boundary if necessary
    if (LOGICAL_WARP_THREADS != 32)
    {
      next_flag = _CUDA_VSTD::min(next_flag, LOGICAL_WARP_THREADS);
    }

    _CCCL_PRAGMA_UNROLL_FULL()
    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      const int OFFSET = 1 << STEP;

      // Share input into buffer
      ThreadStore<STORE_VOLATILE>(&temp_storage.reduce[lane_id], input);

      __syncwarp(member_mask);

      // Update input if peer_addend is in range
      if (OFFSET + lane_id < next_flag)
      {
        T peer_addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage.reduce[lane_id + OFFSET]);
        input         = reduction_op(input, peer_addend);
      }

      __syncwarp(member_mask);
    }

    return input;
  }

  /**
   * @brief Smem-based segmented reduce
   *
   * @tparam HEAD_SEGMENTED
   *   Whether flags indicate a segment-head or a segment-tail
   *
   * @param[in] input
   *   Calling thread's input
   *
   * @param[in] flag
   *   Whether or not the current lane is a segment head/tail
   *
   * @param[in] reduction_op
   *   Reduction operator
   *
   * @param[in] has_ballot
   *   Marker type for whether the target arch has ballot functionality
   */
  template <bool HEAD_SEGMENTED, typename FlagT, typename ReductionOp>
  _CCCL_DEVICE _CCCL_FORCEINLINE T
  SegmentedReduce(T input, FlagT flag, ReductionOp reduction_op, ::cuda::std::false_type /*has_ballot*/)
  {
    enum
    {
      UNSET = 0x0, // Is initially unset
      SET   = 0x1, // Is initially set
      SEEN  = 0x2, // Has seen another head flag from a successor peer
    };

    // Alias flags onto shared data storage
    volatile SmemFlag* flag_storage = temp_storage.flags;

    SmemFlag flag_status = (flag) ? SET : UNSET;

    for (int STEP = 0; STEP < STEPS; STEP++)
    {
      const int OFFSET = 1 << STEP;

      // Share input through buffer
      ThreadStore<STORE_VOLATILE>(&temp_storage.reduce[lane_id], input);

      __syncwarp(member_mask);

      // Get peer from buffer
      T peer_addend = ThreadLoad<LOAD_VOLATILE>(&temp_storage.reduce[lane_id + OFFSET]);

      __syncwarp(member_mask);

      // Share flag through buffer
      flag_storage[lane_id] = flag_status;

      // Get peer flag from buffer
      SmemFlag peer_flag_status = flag_storage[lane_id + OFFSET];

      // Update input if peer was in range
      if (lane_id < LOGICAL_WARP_THREADS - OFFSET)
      {
        if (HEAD_SEGMENTED)
        {
          // Head-segmented
          if ((flag_status & SEEN) == 0)
          {
            // Has not seen a more distant head flag
            if (peer_flag_status & SET)
            {
              // Has now seen a head flag
              flag_status |= SEEN;
            }
            else
            {
              // Peer is not a head flag: grab its count
              input = reduction_op(input, peer_addend);
            }

            // Update seen status to include that of peer
            flag_status |= (peer_flag_status & SEEN);
          }
        }
        else
        {
          // Tail-segmented.  Simply propagate flag status
          if (!flag_status)
          {
            input = reduction_op(input, peer_addend);
            flag_status |= peer_flag_status;
          }
        }
      }
    }

    return input;
  }
};

} // namespace detail

CUB_NAMESPACE_END
