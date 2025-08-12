/******************************************************************************
 * Copyright (c) 2016, NVIDIA CORPORATION.  All rights reserved.
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
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE
 * ARE DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CUDA_COMPILER()

#  include <thrust/system/cuda/config.h>

#  include <cub/device/device_reduce.cuh>
#  include <cub/iterator/cache_modified_input_iterator.cuh>
#  include <cub/util_math.cuh>

#  include <thrust/detail/alignment.h>
#  include <thrust/detail/raw_reference_cast.h>
#  include <thrust/detail/temporary_array.h>
#  include <thrust/distance.h>
#  include <thrust/functional.h>
#  include <thrust/system/cuda/detail/cdp_dispatch.h>
#  include <thrust/system/cuda/detail/core/agent_launcher.h>
#  include <thrust/system/cuda/detail/dispatch.h>
#  include <thrust/system/cuda/detail/get_value.h>
#  include <thrust/system/cuda/detail/make_unsigned_special.h>
#  include <thrust/system/cuda/detail/par_to_seq.h>
#  include <thrust/system/cuda/detail/util.h>

#  include <cuda/std/cstdint>

THRUST_NAMESPACE_BEGIN

// Forward declare generic reduce circumvent circular dependency.
template <typename DerivedPolicy, typename InputIterator, typename T, typename BinaryFunction>
T _CCCL_HOST_DEVICE
reduce(const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
       InputIterator first,
       InputIterator last,
       T init,
       BinaryFunction binary_op);

template <typename DerivedPolicy, typename InputIterator, typename OutputIterator, typename T, typename BinaryFunction>
void _CCCL_HOST_DEVICE reduce_into(
  const thrust::detail::execution_policy_base<DerivedPolicy>& exec,
  InputIterator first,
  InputIterator last,
  OutputIterator output,
  T init,
  BinaryFunction binary_op);

namespace cuda_cub
{

namespace __reduce
{

template <bool>
struct is_true : thrust::detail::false_type
{};
template <>
struct is_true<true> : thrust::detail::true_type
{};

template <int _BLOCK_THREADS,
          int _ITEMS_PER_THREAD                      = 1,
          int _VECTOR_LOAD_LENGTH                    = 1,
          cub::BlockReduceAlgorithm _BLOCK_ALGORITHM = cub::BLOCK_REDUCE_RAKING,
          cub::CacheLoadModifier _LOAD_MODIFIER      = cub::LOAD_DEFAULT,
          cub::GridMappingStrategy _GRID_MAPPING     = cub::GRID_MAPPING_DYNAMIC>
struct PtxPolicy
{
  enum
  {
    BLOCK_THREADS      = _BLOCK_THREADS,
    ITEMS_PER_THREAD   = _ITEMS_PER_THREAD,
    VECTOR_LOAD_LENGTH = _VECTOR_LOAD_LENGTH,
    ITEMS_PER_TILE     = _BLOCK_THREADS * _ITEMS_PER_THREAD
  };

  static const cub::BlockReduceAlgorithm BLOCK_ALGORITHM = _BLOCK_ALGORITHM;
  static const cub::CacheLoadModifier LOAD_MODIFIER      = _LOAD_MODIFIER;
  static const cub::GridMappingStrategy GRID_MAPPING     = _GRID_MAPPING;
}; // struct PtxPolicy

template <class, class>
struct Tuning;

template <class T>
struct Tuning<core::detail::sm52, T>
{
  enum
  {
    // Relative size of T type to a 4-byte word
    SCALE_FACTOR_4B = (sizeof(T) + 3) / 4,
    // Relative size of T type to a 1-byte word
    SCALE_FACTOR_1B = sizeof(T),
  };

  // ReducePolicy1B (GTX Titan: 228.7 GB/s @ 192M 1B items)
  using ReducePolicy1B =
    PtxPolicy<128,
              (((24 / Tuning::SCALE_FACTOR_1B) > (1)) ? (24 / Tuning::SCALE_FACTOR_1B) : (1)),
              4,
              cub::BLOCK_REDUCE_WARP_REDUCTIONS,
              cub::LOAD_LDG,
              cub::GRID_MAPPING_DYNAMIC>;

  // ReducePolicy4B types (GTX Titan: 255.1 GB/s @ 48M 4B items)
  using ReducePolicy4B =
    PtxPolicy<256,
              (((20 / Tuning::SCALE_FACTOR_4B) > (1)) ? (20 / Tuning::SCALE_FACTOR_4B) : (1)),
              4,
              cub::BLOCK_REDUCE_WARP_REDUCTIONS,
              cub::LOAD_LDG,
              cub::GRID_MAPPING_DYNAMIC>;

  using type = ::cuda::std::conditional_t<(sizeof(T) < 4), ReducePolicy1B, ReducePolicy4B>;
}; // Tuning sm52

template <class InputIt, class OutputIt, class T, class Size, class ReductionOp>
struct ReduceAgent
{
  using UnsignedSize = typename detail::make_unsigned_special<Size>::type;

  template <class Arch>
  struct PtxPlan : Tuning<Arch, T>::type
  {
    // we need this type definition to indicate "specialize_plan" metafunction
    // that this PtxPlan may have specializations for different Arch
    // via Tuning<Arch,T> type.
    //
    using tuning = Tuning<Arch, T>;

    using Vector      = cub::CubVector<T, PtxPlan::VECTOR_LOAD_LENGTH>;
    using LoadIt      = cub::detail::try_make_cache_modified_iterator_t<PtxPlan::LOAD_MODIFIER, InputIt>;
    using BlockReduce = cub::BlockReduce<T, PtxPlan::BLOCK_THREADS, PtxPlan::BLOCK_ALGORITHM, 1, 1>;

    using VectorLoadIt = cub::CacheModifiedInputIterator<PtxPlan::LOAD_MODIFIER, Vector, Size>;

    struct TempStorage
    {
      typename BlockReduce::TempStorage reduce;
      //
      Size dequeue_offset;
    }; // struct TempStorage

  }; // struct PtxPlan

  // Reduction need additional information which is not covered in
  // default core::AgentPlan. We thus inherit from core::AgentPlan
  // and add additional member fields that are needed.
  // Other algorithms, e.g. merge, may not need additional information,
  // and may use AgentPlan directly, instead of defining their own Plan type.
  //
  struct Plan : core::detail::AgentPlan
  {
    cub::GridMappingStrategy grid_mapping;

    THRUST_RUNTIME_FUNCTION Plan() {}

    template <class P>
    THRUST_RUNTIME_FUNCTION Plan(P)
        : core::detail::AgentPlan(P())
        , grid_mapping(P::GRID_MAPPING)
    {}
  };

  // this specialized PtxPlan for a device-compiled Arch
  // ptx_plan type *must* only be used from device code
  // Its use from host code will result in *undefined behaviour*
  //
  using ptx_plan = typename core::detail::specialize_plan_msvc10_war<PtxPlan>::type::type;

  using TempStorage  = typename ptx_plan::TempStorage;
  using Vector       = typename ptx_plan::Vector;
  using LoadIt       = typename ptx_plan::LoadIt;
  using BlockReduce  = typename ptx_plan::BlockReduce;
  using VectorLoadIt = typename ptx_plan::VectorLoadIt;

  enum
  {
    ITEMS_PER_THREAD   = ptx_plan::ITEMS_PER_THREAD,
    BLOCK_THREADS      = ptx_plan::BLOCK_THREADS,
    ITEMS_PER_TILE     = ptx_plan::ITEMS_PER_TILE,
    VECTOR_LOAD_LENGTH = ptx_plan::VECTOR_LOAD_LENGTH,

    ATTEMPT_VECTORIZATION = (VECTOR_LOAD_LENGTH > 1) && (ITEMS_PER_THREAD % VECTOR_LOAD_LENGTH == 0)
                         && ::cuda::std::is_pointer<InputIt>::value
                         && ::cuda::std::is_arithmetic<typename ::cuda::std::remove_cv<T>>::value
  };

  struct impl
  {
    //---------------------------------------------------------------------
    // Per thread data
    //---------------------------------------------------------------------

    TempStorage& storage;
    InputIt input_it;
    LoadIt load_it;
    ReductionOp reduction_op;

    //---------------------------------------------------------------------
    // Constructor
    //---------------------------------------------------------------------

    THRUST_DEVICE_FUNCTION impl(TempStorage& storage_, InputIt input_it_, ReductionOp reduction_op_)
        : storage(storage_)
        , input_it(input_it_)
        , load_it(cub::detail::try_make_cache_modified_iterator<ptx_plan::LOAD_MODIFIER>(input_it))
        , reduction_op(reduction_op_)
    {}

    //---------------------------------------------------------------------
    // Utility
    //---------------------------------------------------------------------

    // Whether or not the input is aligned with the vector type
    // (specialized for types we can vectorize)
    //
    template <class Iterator>
    static THRUST_DEVICE_FUNCTION bool is_aligned(Iterator d_in, thrust::detail::true_type /* can_vectorize */)
    {
      return (size_t(d_in) & (sizeof(Vector) - 1)) == 0;
    }

    // Whether or not the input is aligned with the vector type
    // (specialized for types we cannot vectorize)
    //
    template <class Iterator>
    static THRUST_DEVICE_FUNCTION bool is_aligned(Iterator, thrust::detail::false_type /* can_vectorize */)
    {
      return false;
    }

    //---------------------------------------------------------------------
    // Tile processing
    //---------------------------------------------------------------------

    // Consume a full tile of input (non-vectorized)
    //
    template <int IS_FIRST_TILE>
    THRUST_DEVICE_FUNCTION void consume_tile(
      T& thread_aggregate,
      Size block_offset,
      int /*valid_items*/,
      thrust::detail::true_type /* is_full_tile */,
      thrust::detail::false_type /* can_vectorize */)
    {
      T items[ITEMS_PER_THREAD];

      // Load items in striped fashion
      cub::LoadDirectStriped<BLOCK_THREADS>(threadIdx.x, load_it + block_offset, items);

      // Reduce items within each thread stripe
      thread_aggregate = (IS_FIRST_TILE) ? cub::ThreadReduce(items, reduction_op)
                                         : cub::ThreadReduce(items, reduction_op, thread_aggregate);
    }

    // Consume a full tile of input (vectorized)
    //
    template <int IS_FIRST_TILE>
    THRUST_DEVICE_FUNCTION void consume_tile(
      T& thread_aggregate,
      Size block_offset,
      int /*valid_items*/,
      thrust::detail::true_type /* is_full_tile */,
      thrust::detail::true_type /* can_vectorize */)
    {
      // Alias items as an array of VectorT and load it in striped fashion
      enum
      {
        WORDS = ITEMS_PER_THREAD / VECTOR_LOAD_LENGTH
      };

      T items[ITEMS_PER_THREAD];

      Vector* vec_items = reinterpret_cast<Vector*>(items);

      // Vector Input iterator wrapper type (for applying cache modifier)
      T* d_in_unqualified = const_cast<T*>(input_it) + block_offset + (threadIdx.x * VECTOR_LOAD_LENGTH);
      VectorLoadIt vec_load_it(reinterpret_cast<Vector*>(d_in_unqualified));

      _CCCL_PRAGMA_UNROLL_FULL()
      for (int i = 0; i < WORDS; ++i)
      {
        vec_items[i] = vec_load_it[BLOCK_THREADS * i];
      }

      // Reduce items within each thread stripe
      thread_aggregate = (IS_FIRST_TILE) ? cub::ThreadReduce(items, reduction_op)
                                         : cub::ThreadReduce(items, reduction_op, thread_aggregate);
    }

    // Consume a partial tile of input
    //
    template <int IS_FIRST_TILE, class CAN_VECTORIZE>
    THRUST_DEVICE_FUNCTION void consume_tile(
      T& thread_aggregate,
      Size block_offset,
      int valid_items,
      thrust::detail::false_type /* is_full_tile */,
      CAN_VECTORIZE)
    {
      // Partial tile
      int thread_offset = threadIdx.x;

      // Read first item
      if ((IS_FIRST_TILE) && (thread_offset < valid_items))
      {
        thread_aggregate = load_it[block_offset + thread_offset];
        thread_offset += BLOCK_THREADS;
      }

      // Continue reading items (block-striped)
      while (thread_offset < valid_items)
      {
        thread_aggregate =
          reduction_op(thread_aggregate, thrust::raw_reference_cast(load_it[block_offset + thread_offset]));
        thread_offset += BLOCK_THREADS;
      }
    }

    //---------------------------------------------------------------
    // Consume a contiguous segment of tiles
    //---------------------------------------------------------------------

    // Reduce a contiguous segment of input tiles
    //
    template <class CAN_VECTORIZE>
    THRUST_DEVICE_FUNCTION T consume_range_impl(Size block_offset, Size block_end, CAN_VECTORIZE can_vectorize)
    {
      T thread_aggregate;

      if (block_offset + ITEMS_PER_TILE > block_end)
      {
        // First tile isn't full (not all threads have valid items)
        int valid_items = block_end - block_offset;
        consume_tile<true>(thread_aggregate, block_offset, valid_items, thrust::detail::false_type(), can_vectorize);
        return BlockReduce(storage.reduce).Reduce(thread_aggregate, reduction_op, valid_items);
      }

      // At least one full block
      consume_tile<true>(thread_aggregate, block_offset, ITEMS_PER_TILE, thrust::detail::true_type(), can_vectorize);
      block_offset += ITEMS_PER_TILE;

      // Consume subsequent full tiles of input
      while (block_offset + ITEMS_PER_TILE <= block_end)
      {
        consume_tile<false>(thread_aggregate, block_offset, ITEMS_PER_TILE, thrust::detail::true_type(), can_vectorize);
        block_offset += ITEMS_PER_TILE;
      }

      // Consume a partially-full tile
      if (block_offset < block_end)
      {
        int valid_items = block_end - block_offset;
        consume_tile<false>(thread_aggregate, block_offset, valid_items, thrust::detail::false_type(), can_vectorize);
      }

      // Compute block-wide reduction (all threads have valid items)
      return BlockReduce(storage.reduce).Reduce(thread_aggregate, reduction_op);
    }

    // Reduce a contiguous segment of input tiles
    //
    THRUST_DEVICE_FUNCTION T consume_range(Size block_offset, Size block_end)
    {
      using attempt_vec = is_true<ATTEMPT_VECTORIZATION>;
      using path_a      = is_true<true && ATTEMPT_VECTORIZATION>;
      using path_b      = is_true<false && ATTEMPT_VECTORIZATION>;

      return is_aligned(input_it + block_offset, attempt_vec())
             ? consume_range_impl(block_offset, block_end, path_a())
             : consume_range_impl(block_offset, block_end, path_b());
    }

    // Reduce a contiguous segment of input tiles
    //
    THRUST_DEVICE_FUNCTION T consume_tiles(
      Size /*num_items*/,
      cub::GridEvenShare<Size>& even_share,
      cub::GridQueue<UnsignedSize>& /*queue*/,
      thrust::detail::integral_constant<cub::GridMappingStrategy, cub::GRID_MAPPING_RAKE> /*is_rake*/)
    {
      using attempt_vec = is_true<ATTEMPT_VECTORIZATION>;
      using path_a      = is_true<true && ATTEMPT_VECTORIZATION>;
      using path_b      = is_true<false && ATTEMPT_VECTORIZATION>;

      // Initialize even-share descriptor for this thread block
      even_share.template BlockInit<ITEMS_PER_TILE, cub::GRID_MAPPING_RAKE>();

      return is_aligned(input_it, attempt_vec())
             ? consume_range_impl(even_share.block_offset, even_share.block_end, path_a())
             : consume_range_impl(even_share.block_offset, even_share.block_end, path_b());
    }

    //---------------------------------------------------------------------
    // Dynamically consume tiles
    //---------------------------------------------------------------------

    // Dequeue and reduce tiles of items as part of a inter-block reduction
    //
    template <class CAN_VECTORIZE>
    THRUST_DEVICE_FUNCTION T
    consume_tiles_impl(Size num_items, cub::GridQueue<UnsignedSize> queue, CAN_VECTORIZE can_vectorize)
    {
      // We give each thread block at least one tile of input.
      T thread_aggregate;
      Size block_offset    = blockIdx.x * ITEMS_PER_TILE;
      Size even_share_base = gridDim.x * ITEMS_PER_TILE;

      if (block_offset + ITEMS_PER_TILE > num_items)
      {
        // First tile isn't full (not all threads have valid items)
        int valid_items = num_items - block_offset;
        consume_tile<true>(thread_aggregate, block_offset, valid_items, thrust::detail::false_type(), can_vectorize);
        return BlockReduce(storage.reduce).Reduce(thread_aggregate, reduction_op, valid_items);
      }

      // Consume first full tile of input
      consume_tile<true>(thread_aggregate, block_offset, ITEMS_PER_TILE, thrust::detail::true_type(), can_vectorize);

      if (num_items > even_share_base)
      {
        // Dequeue a tile of items
        if (threadIdx.x == 0)
        {
          storage.dequeue_offset = queue.Drain(ITEMS_PER_TILE) + even_share_base;
        }

        __syncthreads();

        // Grab tile offset and check if we're done with full tiles
        block_offset = storage.dequeue_offset;

        // Consume more full tiles
        while (block_offset + ITEMS_PER_TILE <= num_items)
        {
          consume_tile<false>(
            thread_aggregate, block_offset, ITEMS_PER_TILE, thrust::detail::true_type(), can_vectorize);

          __syncthreads();

          // Dequeue a tile of items
          if (threadIdx.x == 0)
          {
            storage.dequeue_offset = queue.Drain(ITEMS_PER_TILE) + even_share_base;
          }

          __syncthreads();

          // Grab tile offset and check if we're done with full tiles
          block_offset = storage.dequeue_offset;
        }

        // Consume partial tile
        if (block_offset < num_items)
        {
          int valid_items = num_items - block_offset;
          consume_tile<false>(thread_aggregate, block_offset, valid_items, thrust::detail::false_type(), can_vectorize);
        }
      }

      // Compute block-wide reduction (all threads have valid items)
      return BlockReduce(storage.reduce).Reduce(thread_aggregate, reduction_op);
    }

    // Dequeue and reduce tiles of items as part of a inter-block reduction
    //
    THRUST_DEVICE_FUNCTION T consume_tiles(
      Size num_items,
      cub::GridEvenShare<Size>& /*even_share*/,
      cub::GridQueue<UnsignedSize>& queue,
      thrust::detail::integral_constant<cub::GridMappingStrategy, cub::GRID_MAPPING_DYNAMIC>)
    {
      using attempt_vec = is_true<ATTEMPT_VECTORIZATION>;
      using path_a      = is_true<true && ATTEMPT_VECTORIZATION>;
      using path_b      = is_true<false && ATTEMPT_VECTORIZATION>;

      return is_aligned(input_it, attempt_vec())
             ? consume_tiles_impl(num_items, queue, path_a())
             : consume_tiles_impl(num_items, queue, path_b());
    }
  }; // struct impl

  //---------------------------------------------------------------------
  // Agent entry points
  //---------------------------------------------------------------------

  // single tile reduce entry point
  //
  THRUST_AGENT_ENTRY(InputIt input_it, OutputIt output_it, Size num_items, ReductionOp reduction_op, char* shmem)
  {
    TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

    if (num_items == 0)
    {
      return;
    }

    T block_aggregate = impl(storage, input_it, reduction_op).consume_range((Size) 0, num_items);

    if (threadIdx.x == 0)
    {
      *output_it = block_aggregate;
    }
  }

  // single tile reduce entry point
  //
  THRUST_AGENT_ENTRY(InputIt input_it, OutputIt output_it, Size num_items, ReductionOp reduction_op, T init, char* shmem)
  {
    TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

    if (num_items == 0)
    {
      if (threadIdx.x == 0)
      {
        *output_it = init;
      }
      return;
    }

    T block_aggregate = impl(storage, input_it, reduction_op).consume_range((Size) 0, num_items);

    if (threadIdx.x == 0)
    {
      *output_it = reduction_op(init, block_aggregate);
    }
  }

  THRUST_AGENT_ENTRY(
    InputIt input_it,
    OutputIt output_it,
    Size num_items,
    cub::GridEvenShare<Size> even_share,
    cub::GridQueue<UnsignedSize> queue,
    ReductionOp reduction_op,
    char* shmem)
  {
    TempStorage& storage = *reinterpret_cast<TempStorage*>(shmem);

    using grid_mapping = thrust::detail::integral_constant<cub::GridMappingStrategy, ptx_plan::GRID_MAPPING>;

    T block_aggregate =
      impl(storage, input_it, reduction_op).consume_tiles(num_items, even_share, queue, grid_mapping());

    if (threadIdx.x == 0)
    {
      output_it[blockIdx.x] = block_aggregate;
    }
  }
}; // struct ReduceAgent

template <class Size>
struct DrainAgent
{
  using UnsignedSize = typename detail::make_unsigned_special<Size>::type;

  template <class Arch>
  struct PtxPlan : PtxPolicy<1>
  {};
  using ptx_plan = core::detail::specialize_plan<PtxPlan>;

  //---------------------------------------------------------------------
  // Agent entry point
  //---------------------------------------------------------------------

  THRUST_AGENT_ENTRY(cub::GridQueue<UnsignedSize> grid_queue, Size num_items, char* /*shmem*/)
  {
    grid_queue.FillAndResetDrain(num_items);
  }
}; // struct DrainAgent;
} // namespace __reduce

namespace detail
{

template <typename Derived, typename InputIt, typename Size, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION size_t get_reduce_n_temporary_storage_size(
  execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  size_t tmp_size = 0;

  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::Reduce,
    num_items,
    (nullptr, tmp_size, first, static_cast<T*>(nullptr), num_items_fixed, binary_op, init, stream));
  cuda_cub::throw_on_error(status, "after determining reduce temporary storage size");

  return tmp_size;
}

template <typename Derived, typename InputIt, typename Size, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION T
reduce_n_impl(execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  // Determine temporary device storage requirements.

  size_t tmp_size = get_reduce_n_temporary_storage_size(policy, first, num_items, init, binary_op);

  // Allocate temporary storage.

  // We allocate both the temporary storage needed for the algorithm, and a `T`
  // to store the result.
  //
  // The array was dynamically allocated, so we assume that it's suitably
  // aligned for any type of data. `malloc`/`cudaMalloc`/`new`/`std::allocator`
  // make this guarantee.

  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, tmp_size + sizeof(T));

  // Run reduction.

  T* ret_ptr    = thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get());
  void* tmp_ptr = static_cast<void*>((tmp.data() + sizeof(T)).get());
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::Reduce,
    num_items,
    (tmp_ptr, tmp_size, first, ret_ptr, num_items_fixed, binary_op, init, stream));
  cuda_cub::throw_on_error(status, "after reduce invocation");

  // Synchronize the stream and get the value.

  status = cuda_cub::synchronize(policy);
  cuda_cub::throw_on_error(status, "reduce failed to synchronize");
  return thrust::cuda_cub::get_value(policy, thrust::detail::aligned_reinterpret_cast<T*>(tmp.data().get()));
}

template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename BinaryOp>
THRUST_RUNTIME_FUNCTION void reduce_n_into_impl(
  execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt output, T init, BinaryOp binary_op)
{
  cudaStream_t stream = cuda_cub::stream(policy);
  cudaError_t status;

  // Determine temporary device storage requirements.

  size_t tmp_size = get_reduce_n_temporary_storage_size(policy, first, num_items, init, binary_op);

  // Allocate temporary storage.

  thrust::detail::temporary_array<std::uint8_t, Derived> tmp(policy, tmp_size);

  // Run reduction.

  void* tmp_ptr = thrust::raw_pointer_cast(tmp.data());
  THRUST_INDEX_TYPE_DISPATCH(
    status,
    cub::DeviceReduce::Reduce,
    num_items,
    (tmp_ptr, tmp_size, first, output, num_items_fixed, binary_op, init, stream));
  cuda_cub::throw_on_error(status, "after reduce invocation");

  // Synchronize the stream and get the value.

  status = cuda_cub::synchronize_optional(policy);
  cuda_cub::throw_on_error(status, "reduce failed to synchronize");
}

} // namespace detail

//-------------------------
// Thrust API entry points
//-------------------------

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename T, typename BinaryOp>
_CCCL_HOST_DEVICE T
reduce_n(execution_policy<Derived>& policy, InputIt first, Size num_items, T init, BinaryOp binary_op)
{
  THRUST_CDP_DISPATCH(
    (init = thrust::cuda_cub::detail::reduce_n_impl(policy, first, num_items, init, binary_op);),
    (init = thrust::reduce(cvt_to_seq(derived_cast(policy)), first, first + num_items, init, binary_op);));
  return init;
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Derived, typename InputIt, typename Size, typename OutputIt, typename T, typename BinaryOp>
_CCCL_HOST_DEVICE void reduce_n_into(
  execution_policy<Derived>& policy, InputIt first, Size num_items, OutputIt output, T init, BinaryOp binary_op)
{
  THRUST_CDP_DISPATCH(
    (thrust::cuda_cub::detail::reduce_n_into_impl(policy, first, num_items, output, init, binary_op);),
    (thrust::reduce_into(cvt_to_seq(derived_cast(policy)), first, first + num_items, output, init, binary_op);));
}

template <class Derived, class InputIt, class T, class BinaryOp>
_CCCL_HOST_DEVICE T reduce(execution_policy<Derived>& policy, InputIt first, InputIt last, T init, BinaryOp binary_op)
{
  using size_type = thrust::detail::it_difference_t<InputIt>;
  // FIXME: Check for RA iterator.
  size_type num_items = static_cast<size_type>(::cuda::std::distance(first, last));
  return cuda_cub::reduce_n(policy, first, num_items, init, binary_op);
}

template <class Derived, class InputIt, class T>
_CCCL_HOST_DEVICE T reduce(execution_policy<Derived>& policy, InputIt first, InputIt last, T init)
{
  return cuda_cub::reduce(policy, first, last, init, ::cuda::std::plus<T>());
}

template <class Derived, class InputIt>
_CCCL_HOST_DEVICE thrust::detail::it_value_t<InputIt>
reduce(execution_policy<Derived>& policy, InputIt first, InputIt last)
{
  using value_type = thrust::detail::it_value_t<InputIt>;
  return cuda_cub::reduce(policy, first, last, value_type(0));
}

template <class Derived, class InputIt, class OutputIt, class T, class BinaryOp>
_CCCL_HOST_DEVICE void
reduce_into(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, T init, BinaryOp binary_op)
{
  using size_type = thrust::detail::it_difference_t<InputIt>;
  // FIXME: Check for RA iterator.
  size_type num_items = static_cast<size_type>(::cuda::std::distance(first, last));
  cuda_cub::reduce_n_into(policy, first, num_items, output, init, binary_op);
}

template <class Derived, class InputIt, class OutputIt, class T>
_CCCL_HOST_DEVICE void
reduce_into(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output, T init)
{
  cuda_cub::reduce_into(policy, first, last, output, init, ::cuda::std::plus<T>());
}

template <class Derived, class InputIt, class OutputIt>
_CCCL_HOST_DEVICE void reduce_into(execution_policy<Derived>& policy, InputIt first, InputIt last, OutputIt output)
{
  using value_type = thrust::detail::it_value_t<InputIt>;
  return cuda_cub::reduce_into(policy, first, last, output, value_type(0));
}

} // namespace cuda_cub

THRUST_NAMESPACE_END

#  include <thrust/memory.h>
#  include <thrust/reduce.h>

#endif
