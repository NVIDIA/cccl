// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::BlockLoadToShared class provides a :ref:`collective <collective-primitives>` method for asynchronously
//! loading data from global to shared memory.

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <thrust/type_traits/is_trivially_relocatable.h>

#include <cuda/cmath>
#include <cuda/memory>
#include <cuda/ptx>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__bit/has_single_bit.h>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <nv/target>

CUB_NAMESPACE_BEGIN

namespace detail
{
//! @rst
//! The @c BlockLoadToShared class provides a :ref:`collective <collective-primitives>` method for asynchronously
//! loading data from global to shared memory.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - Given one or more spans of input elements in global memory and buffers in shared memory, this primitive
//!   asynchronously copies the elements to shared memory and takes care of synchronization.
//! - @rowmajor
//! - Shared memory buffers are assumed to be aligned according to `SharedBufferAlignBytes<T>()`.
//! - Global memory spans are by default assumed to be aligned according to the value type. Higher alignment guarantees
//!   can optionally be specified.
//! - After one or more calls to `CopyAsync`, `Commit` needs to be called before optionally doing other work and then
//!   calling `Wait` which guarantees the data to be available in shared memory and resets the state and allows for the
//!   next wave of `CopyAsync`.
//!
//! Performance Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - Uses special instructions/hardware acceleration when available (cp.async.bulk on Hopper+, copy.async on Ampere).
//! - By guaranteeing 16 byte alignment and size multiple for the global span, a faster path is taken.
template <int BlockDimX, int BlockDimY = 1, int BlockDimZ = 1>
struct BlockLoadToShared
{
private:
  /// Constants
  static constexpr int block_threads = BlockDimX * BlockDimY * BlockDimZ;
  // The alignment needed for cp.async.bulk and L1-skipping cp.async
  static constexpr int minimum_align = 16;

  // Helper for fallback to gmem->reg->smem
  struct alignas(minimum_align) vec_load_t
  {
    char c_array[minimum_align];
  };

  struct _TempStorage
  {
    ::cuda::std::uint64_t mbarrier_handle;
  };

#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
  enum struct State
  {
    ready_to_copy,
    ready_to_copy_or_commit,
    committed,
    invalidated,
  };
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

  /// Shared storage reference
  _TempStorage& temp_storage;

  const int linear_tid{cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ)};

  // Thread selection for uniform operations
  const bool elected{__elect_thread()};
  // Keep track of current mbarrier phase for waiting.
  uint32_t phase_parity{};
  // Keep track of the amount of bytes from multiple transactions for Commit() (only needed for TMA).
  // Also used to check for proper ordering of member function calls in debug mode.
  uint32_t num_bytes_bulk_total{};

#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
  State state{State::ready_to_copy};
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

  _CCCL_DEVICE _CCCL_FORCEINLINE bool __elect_thread() const
  {
    // Otherwise elect.sync in the last warp with a full mask is UB.
    static_assert(block_threads % cub::detail::warp_threads == 0, "The block size must be a multiple of the warp size");
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      ( // Use last warp to try to avoid having the elected thread also working on the peeling in the first warp.
        return (linear_tid >= block_threads - cub::detail::warp_threads) && ::cuda::ptx::elect_sync(~0u);),
      NV_IS_DEVICE,
      (return linear_tid == 0;));
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void __init_mbarrier()
  {
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90,
                   (if (elected) { ::cuda::ptx::mbarrier_init(&temp_storage.mbarrier_handle, 1); }
                    // TODO The following sync was added to avoid a racecheck posititive. Is it really needed?
                    __syncthreads();));
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void __copy_aligned_async_bulk(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    if (elected)
    {
#if __cccl_ptx_isa >= 860
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (::cuda::ptx::cp_async_bulk(
           ::cuda::ptx::space_shared,
           ::cuda::ptx::space_global,
           smem_dst,
           gmem_src,
           num_bytes,
           &temp_storage.mbarrier_handle);));
#else
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (::cuda::ptx::cp_async_bulk(
           ::cuda::ptx::space_cluster,
           ::cuda::ptx::space_global,
           smem_dst,
           gmem_src,
           num_bytes,
           &temp_storage.mbarrier_handle);));
#endif // __cccl_ptx_isa >= 800
      // Needed for arrival on mbarrier in Commit()
      num_bytes_bulk_total += num_bytes;
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void __copy_aligned_async(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    for (int offset = linear_tid * minimum_align; offset < num_bytes; offset += block_threads * minimum_align)
    {
      [[maybe_unused]] const auto thread_src = gmem_src + offset;
      [[maybe_unused]] const auto thread_dst = smem_dst + offset;
      // LDGSTS borrowed from cuda::memcpy_async, assumes 16 byte alignment to avoid L1 (.cg)
      NV_IF_TARGET(NV_PROVIDES_SM_80,
                   (asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %2;" : : "r"(
                                   static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(thread_dst))),
                                 "l"(thread_src),
                                 "n"(16) : "memory");));
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void __copy_aligned_fallback(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    for (int offset = linear_tid * minimum_align; offset < num_bytes; offset += block_threads * minimum_align)
    {
      const auto thread_src                       = gmem_src + offset;
      const auto thread_dst                       = smem_dst + offset;
      *::cuda::ptr_rebind<vec_load_t>(thread_dst) = *::cuda::ptr_rebind<vec_load_t>(thread_src);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void __copy_aligned(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (__copy_aligned_async_bulk(smem_dst, gmem_src, num_bytes);),
      NV_PROVIDES_SM_80,
      (__copy_aligned_async(smem_dst, gmem_src, num_bytes);),
      NV_IS_DEVICE,
      (__copy_aligned_fallback(smem_dst, gmem_src, num_bytes);));
  }

  // Dispatch to fallback for waiting pre TMA/SM_90
  _CCCL_DEVICE _CCCL_FORCEINLINE bool __try_wait()
  {
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return ::cuda::ptx::mbarrier_try_wait_parity(&temp_storage.mbarrier_handle, phase_parity);),
      NV_PROVIDES_SM_80,
      (asm volatile("cp.async.wait_group 0;" :: : "memory"); //
       __syncthreads();
       return true;),
      NV_IS_DEVICE,
      (__syncthreads(); //
       return true;));
  }

  // token is only constructible by BlockLoadToShared
  class token_impl
  {
    friend class BlockLoadToShared;
    _CCCL_DEVICE _CCCL_FORCEINLINE token_impl() {} // ctor must have a body to avoid token_impl{} to compile

    token_impl(const token_impl&)            = delete;
    token_impl& operator=(const token_impl&) = delete;
  };

public:
  /// @smemstorage{BlockLoadToShared}
  using TempStorage = cub::Uninitialized<_TempStorage>;

  //! Token type used to enforce correct call order between Commit() and Wait()
  //! member functions. Returned by Commit() and required by Wait() as a usage
  //! guard.
  using CommitToken = token_impl;

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using the specified memory allocation as temporary storage.
  //!
  //! @param[in] temp_storage
  //!   Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoadToShared(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
  {
    _CCCL_ASSERT(::cuda::device::is_object_from(temp_storage, ::cuda::device::address_space::shared),
                 "temp_storage has to be in shared memory");
    __init_mbarrier();
  }

  _CCCL_DEVICE BlockLoadToShared(const BlockLoadToShared<BlockDimX, BlockDimY, BlockDimZ>&) = delete;

  //! @}  end member group

  _CCCL_DEVICE BlockLoadToShared& operator=(const BlockLoadToShared<BlockDimX, BlockDimY, BlockDimZ>&) = delete;

  //! @brief Invalidates underlying @c mbarrier enabling reuse of its temporary storage.
  //! @note
  //! Block-synchronization is needed after calling `Invalidate()` to reuse the shared memory from the temporary
  //! storage.
  // This is not the destructor to avoid overhead when shared memory reuse is not needed.
  _CCCL_DEVICE _CCCL_FORCEINLINE void Invalidate()
  {
#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
    _CCCL_ASSERT(state == State::ready_to_copy, "Wait() must be called before Invalidate()");
    state = State::invalidated;
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS
    // Make sure all threads are done interacting with the mbarrier
    __syncthreads();
    if (elected)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_90, ::cuda::ptx::mbarrier_inval(&temp_storage.mbarrier_handle););
    }
    // Make sure the elected thread is done invalidating the mbarrier
    __syncthreads();
  }

  //! @brief Copy elements from global to shared memory
  //! @tparam T
  //!   **[inferred]** Value type for this transaction
  //! @tparam GmemAlign
  //!   Guaranteed alignment in bytes of the source range (both begin and end) in global memory
  //! @param[in] smem_dst
  //!   Destination buffer in shared memory that is aligned to `SharedBufferAlignBytes<T>()` and at least
  //!   `SharedBufferSizeBytes<T, GmemAlign>(size(gmem_src))` big.
  //! @param[in] gmem_src
  //!   Source range in global memory, determines the size of the transaction
  //! @return
  //!   The range in shared memory (same size as `gmem_src`) which should be used to access the data after `Commit` and
  //!   `Wait`.
  // TODO Allow spans with static sizes?
  template <typename T, int GmemAlign = alignof(T)>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<T>
  CopyAsync(::cuda::std::span<char> smem_dst, ::cuda::std::span<const T> gmem_src)
  {
    // TODO Should this be weakened to thrust::is_trivially_relocatable?
    static_assert(THRUST_NS_QUALIFIER::is_trivially_relocatable_v<T>);
    static_assert(::cuda::std::has_single_bit(unsigned{GmemAlign}));
    static_assert(GmemAlign >= int{alignof(T)});
    constexpr bool bulk_aligned = GmemAlign >= minimum_align;
    // Avoid 64b multiplication in span::size_bytes()
    const int num_bytes = static_cast<int>(sizeof(T)) * static_cast<int>(size(gmem_src));
    const auto dst_ptr  = data(smem_dst);
    const auto src_ptr  = ::cuda::ptr_rebind<char>(data(gmem_src));
    _CCCL_ASSERT(dst_ptr == nullptr || ::cuda::device::is_address_from(dst_ptr, ::cuda::device::address_space::shared),
                 "Destination address needs to point to shared memory");
    _CCCL_ASSERT(src_ptr == nullptr || ::cuda::device::is_address_from(src_ptr, ::cuda::device::address_space::global),
                 "Source address needs to point to global memory");
    _CCCL_ASSERT((src_ptr != nullptr && dst_ptr != nullptr) || num_bytes == 0,
                 "Only when the source range is empty are nullptrs allowed");
    _CCCL_ASSERT(::cuda::is_aligned(src_ptr, GmemAlign),
                 "Begin of global memory range needs to be aligned according to GmemAlign.");
    _CCCL_ASSERT(::cuda::is_aligned(src_ptr + num_bytes, GmemAlign),
                 "End of global memory range needs to be aligned according to GmemAlign.");
    _CCCL_ASSERT(::cuda::is_aligned(dst_ptr, SharedBufferAlignBytes<T>()),
                 "Shared memory needs to be 16 byte aligned.");
    _CCCL_ASSERT((static_cast<int>(size(smem_dst)) >= SharedBufferSizeBytes<T, GmemAlign>(size(gmem_src))),
                 "Shared memory destination buffer must have enough space");
#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
    _CCCL_ASSERT(state == State::ready_to_copy || state == State::ready_to_copy_or_commit,
                 "Wait() must be called before another CopyAsync()");
    state = State::ready_to_copy_or_commit;
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS
    if constexpr (bulk_aligned)
    {
      __copy_aligned(dst_ptr, src_ptr, num_bytes);
      return {::cuda::ptr_rebind<T>(data(smem_dst)), size(gmem_src)};
    }
    else
    {
      const auto src_ptr_aligned   = ::cuda::align_up(src_ptr, minimum_align);
      const int align_diff         = static_cast<int>(src_ptr_aligned - src_ptr);
      const int head_padding_bytes = (minimum_align - align_diff) % minimum_align;
      const auto actual_dst_ptr    = dst_ptr + head_padding_bytes;
      const int head_peeling_bytes = ::cuda::std::min(align_diff, num_bytes);
      const int num_bytes_bulk     = ::cuda::round_down(num_bytes - head_peeling_bytes, minimum_align);
      __copy_aligned(actual_dst_ptr + head_peeling_bytes, src_ptr_aligned, num_bytes_bulk);

      // Peel head and tail
      // Make sure we have enough threads for the worst case of minimum_align bytes on each side.
      static_assert(block_threads >= 2 * (minimum_align - 1));
      // |-------------head--------------|--------------------------tail--------------------------|
      // 0, 1, ... head_peeling_bytes - 1, head_peeling_bytes + num_bytes_bulk, ..., num_bytes - 1
      const int begin_offset = linear_tid < head_peeling_bytes ? 0 : num_bytes_bulk;
      if (const int idx = begin_offset + linear_tid; idx < num_bytes)
      {
        actual_dst_ptr[idx] = src_ptr[idx];
      }
      return {::cuda::ptr_rebind<T>(actual_dst_ptr), size(gmem_src)};
    }
  }

  // Avoid need to explicitly specify `T` for non-const src.
  //! @brief Convenience overload, see `CopyAsync(span<char>, span<const T>)`.
  template <typename T, int GmemAlign = alignof(T)>
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE ::cuda::std::span<T>
  CopyAsync(::cuda::std::span<char> smem_dst, ::cuda::std::span<T> gmem_src)
  {
    return CopyAsync<T, GmemAlign>(smem_dst, ::cuda::std::span<const T>{gmem_src});
  }

  //! @brief Commit one or more @c CopyAsync() calls.
  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE CommitToken Commit()
  {
#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
    _CCCL_ASSERT(state == State::ready_to_copy_or_commit, "CopyAsync() must be called before Commit()");
    state = State::committed;
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (if (elected) {
        ::cuda::ptx::mbarrier_arrive_expect_tx(
          ::cuda::ptx::sem_release,
          ::cuda::ptx::scope_cta,
          ::cuda::ptx::space_shared,
          &temp_storage.mbarrier_handle,
          num_bytes_bulk_total);
        num_bytes_bulk_total = 0u;
      } //
       __syncthreads();),
      NV_PROVIDES_SM_80,
      (asm volatile("cp.async.commit_group ;" :: : "memory");));

    // Token's mere purpose currently is to prevent calling Wait() without a
    // prior Commit()
    return CommitToken{};
  }

  //! @brief Wait for previously committed copies to arrive. Prepare for next
  //! calls to @c CopyAsync() .
  _CCCL_DEVICE _CCCL_FORCEINLINE void Wait(CommitToken&&)
  {
#ifdef CCCL_ENABLE_DEVICE_ASSERTIONS
    _CCCL_ASSERT(state == State::committed, "Commit() must be called before Wait()");
    state = State::ready_to_copy;
#endif // CCCL_ENABLE_DEVICE_ASSERTIONS

    while (!__try_wait())
      ;
    phase_parity ^= 1u;
  }

  //! @brief Convenience overload calling `Commit()` and `Wait()`.
  _CCCL_DEVICE _CCCL_FORCEINLINE void CommitAndWait()
  {
    Wait(Commit());
  }

  // Having these as static members does require using "template" in user code which is kludgy.

  //! @brief Returns the alignment needed for the shared memory destination buffer.
  //! @tparam T
  //!   Value type to be loaded.
  template <typename T>
  _CCCL_HOST_DEVICE static constexpr int SharedBufferAlignBytes()
  {
    return (::cuda::std::max) (int{alignof(T)}, minimum_align);
  }

  //! @brief Returns the size needed for the shared memory destination buffer.
  //! @tparam T
  //!   Value type to be loaded.
  //! @tparam GmemAlign
  //!   Guaranteed alignment in bytes of the source range (both begin and end) in global memory
  //! @param[in] num_items
  //!   Size of the source range in global memory
  template <typename T, int GmemAlign = alignof(T)>
  _CCCL_HOST_DEVICE static constexpr int SharedBufferSizeBytes(::cuda::std::size_t num_items)
  {
    static_assert(::cuda::std::has_single_bit(unsigned{GmemAlign}));
    static_assert(GmemAlign >= int{alignof(T)});
    _CCCL_ASSERT(num_items <= ::cuda::std::size_t{::cuda::std::numeric_limits<int>::max()},
                 "num_items must fit into an int");
    constexpr bool bulk_aligned = GmemAlign >= minimum_align;
    const int num_bytes         = static_cast<int>(num_items) * int{sizeof(T)};
    const int extra_space       = (bulk_aligned || num_bytes == 0) ? 0 : minimum_align;
    return bulk_aligned ? num_bytes : (::cuda::round_up(num_bytes, minimum_align) + extra_space);
  }
};
} // namespace detail

CUB_NAMESPACE_END
