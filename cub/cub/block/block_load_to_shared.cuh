// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

//! @file
//! The @c cub::BlockLoadToShared class provides a :ref:`collective <collective-primitives>` method for asynchronously
//! loading data from global to shared memory on Ampere and newer architectures.

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
//! loading data from global to shared memory on Ampere and newer architectures.
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
//! - Uses special instructions/hardware acceleration when available (TMA for Hopper+, otherwise LDGSTS).
//! - By guaranteeing 16 byte alignment for the global span (both start and end/size must be a multiple), a faster path
//!   is taken.
//! - Depending on architecture even higher alignments can be beneficial.
template <int BlockDimX, int BlockDimY = 1, int BlockDimZ = 1>
struct BlockLoadToShared
{
private:
  /// Constants
  static constexpr int block_threads = BlockDimX * BlockDimY * BlockDimZ;
  // The alignment needed for TMA/efficient LDGSTS
  static constexpr int minimum_align = 16;

  struct _TempStorage
  {
    // mbarrier
    ::cuda::std::uint64_t bar;
  };

  enum struct State
  {
    ready_to_copy,
    ready_to_copy_or_commit,
    committed,
  };

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  const int linear_tid{cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ)};

  // Thread selection for non-collective operations
  const bool elected{Elect()};
  // Keep track of current mbarrier phase for waiting.
  uint32_t phase_parity{};
  // Keep track of the amount of bytes from multiple transactions for Commit() (only needed for TMA).
  // Also used to check for proper ordering of member function calls in debug mode.
  uint32_t num_bytes_bulk_total{};
  // Only for debugging/asserts, should be optimized away in release builds
  State state{State::ready_to_copy};

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE bool Elect()
  {
    return NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (::cuda::ptx::elect_sync(~0) && linear_tid < 32), //
      NV_IS_DEVICE,
      (linear_tid == 0));
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void InitBarrier()
  {
    {
      NV_DISPATCH_TARGET(
        NV_PROVIDES_SM_90,
        (if (elected) { ::cuda::ptx::mbarrier_init(&temp_storage.bar, 1); }
         // TODO The following sync was added to avoid a racecheck posititive. Is it really needed?
         __syncthreads();),
        NV_PROVIDES_SM_80,
        (if (elected) { ::cuda::ptx::mbarrier_init(&temp_storage.bar, block_threads); } //
         __syncthreads();));
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void CopyAlignedBulk(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    if (elected)
    {
#if __cccl_ptx_isa >= 860
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (::cuda::ptx::cp_async_bulk(
           ::cuda::ptx::space_shared, ::cuda::ptx::space_global, smem_dst, gmem_src, num_bytes, &temp_storage.bar);));
#elif __cccl_ptx_isa >= 800
      NV_IF_TARGET(
        NV_PROVIDES_SM_90,
        (::cuda::ptx::cp_async_bulk(
           ::cuda::ptx::space_cluster, ::cuda::ptx::space_global, smem_dst, gmem_src, num_bytes, &temp_storage.bar);));
#endif // __cccl_ptx_isa >= 800
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void CopyAlignedLDGSTS(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    for (int offset = linear_tid * minimum_align; offset < num_bytes; offset += block_threads * minimum_align)
    {
      const auto thread_src = gmem_src + offset;
      const auto thread_dst = smem_dst + offset;
// LDGSTS borrowed from cuda::memcpy_async, assumes 16 byte alignment to avoid L1 (.cg)
#if _CCCL_CUDA_COMPILER(NVCC, <, 12, 1) // WAR for compiler state space issues
      NV_IF_TARGET(NV_PROVIDES_SM_80,
                   (asm volatile(R"XYZ(
        {
          .reg .u64 tmp;
          .reg .u32 dst;

          cvta.to.shared.u64 tmp, %0;
          cvt.u32.u64 dst, tmp;
          cvta.to.global.u64 tmp, %1;
          cp.async.cg.shared.global [dst], [tmp], 16, 16;
        }
        )XYZ" : : "l"(thread_dst),
                                 "l"(thread_src) : "memory");));
#else // ^^^^ NVCC 12.0 / !NVCC 12.0 vvvvv WAR for compiler state space issues
      NV_IF_TARGET(NV_PROVIDES_SM_80,
                   (asm volatile("cp.async.cg.shared.global [%0], [%1], %2, %2;" : : "r"(
                                   static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(thread_dst))),
                                 "l"(static_cast<::cuda::std::uint64_t>(::__cvta_generic_to_global(thread_src))),
                                 "n"(16) : "memory");));
#endif // _CCCL_CUDA_COMPILER(NVCC, >=, 12, 1)
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void CopyAlignedFallback(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    for (int offset = linear_tid * minimum_align; offset < num_bytes; offset += block_threads * minimum_align)
    {
      const auto thread_src                  = gmem_src + offset;
      const auto thread_dst                  = smem_dst + offset;
      *::cuda::ptr_rebind<uint4>(thread_dst) = *::cuda::ptr_rebind<uint4>(thread_src);
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void CopyAligned(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    num_bytes_bulk_total += num_bytes;
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (CopyAlignedBulk(smem_dst, gmem_src, num_bytes);),
      NV_PROVIDES_SM_80,
      (CopyAlignedLDGSTS(smem_dst, gmem_src, num_bytes);),
      NV_IS_DEVICE,
      (CopyAlignedFallback(smem_dst, gmem_src, num_bytes);));
  }

  // WAR for waiting pre TMA/SM_90
  _CCCL_DEVICE _CCCL_FORCEINLINE bool TryWait()
  {
    // TODO Add backoff at least for SM_80?
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (return ::cuda::ptx::mbarrier_try_wait_parity(&temp_storage.bar, phase_parity);),
      NV_PROVIDES_SM_80,
      (const bool done = ::cuda::ptx::mbarrier_test_wait_parity(&temp_storage.bar, phase_parity); //
       if (!done) { __nanosleep(0); } //
       return done;),
      NV_IS_DEVICE,
      (__syncthreads(); //
       return true;));
  }

public:
  /// @smemstorage{BlockLoadToShared}
  struct TempStorage : cub::Uninitialized<_TempStorage>
  {};

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoadToShared()
      : temp_storage(PrivateStorage())
  {
    InitBarrier();
  }

  //! @brief Collective constructor using the specified memory allocation as temporary storage.
  //!
  //! @param[in] temp_storage
  //!   Reference to memory allocation having layout type TempStorage
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoadToShared(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
  {
    _CCCL_ASSERT(::cuda::device::is_object_from(temp_storage, ::cuda::device::address_space::shared),
                 "temp_storage has to be in shared memory");
    InitBarrier();
  }

  //! @}  end member group

  //! @brief Collective destructor invalidates underlying @c mbarrier enabling reuse of its temporary storage.
  //! @note
  //! Block-synchronization is needed afterwards to reuse the shared memory from the temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE ~BlockLoadToShared()
  {
    if (elected)
    {
      NV_IF_TARGET(NV_PROVIDES_SM_80,
                   (
                     // Stolen from cuda::barrier
                     asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
                       ::__cvta_generic_to_shared(&temp_storage.bar))) : "memory");));
    }
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
    static_assert(::cuda::std::is_trivially_copyable_v<T>);
    static_assert(::cuda::std::has_single_bit(unsigned{GmemAlign}));
    static_assert(GmemAlign >= int{alignof(T)});
    constexpr bool bulk_aligned = GmemAlign >= minimum_align;
    // Avoid 64b multiplication in span::size_bytes()
    const int num_bytes = static_cast<int>(sizeof(T)) * static_cast<int>(size(gmem_src));
    const auto dst_ptr  = data(smem_dst);
    const auto src_ptr  = ::cuda::ptr_rebind<char>(data(gmem_src));
    _CCCL_ASSERT(::cuda::device::is_address_from(dst_ptr, ::cuda::device::address_space::shared),
                 "Destination address needs to point to shared memory");
    _CCCL_ASSERT(src_ptr == nullptr || ::cuda::device::is_address_from(src_ptr, ::cuda::device::address_space::global),
                 "Source address needs to point to global memory");
    _CCCL_ASSERT(src_ptr != nullptr || num_bytes == 0, "Only an empty source range can be nullptr");
    _CCCL_ASSERT(::cuda::is_aligned(src_ptr, GmemAlign),
                 "Begin of global memory range needs to be aligned according to GmemAlign.");
    _CCCL_ASSERT(::cuda::is_aligned(src_ptr + num_bytes, GmemAlign),
                 "End of global memory range needs to be aligned according to GmemAlign.");
    _CCCL_ASSERT(::cuda::is_aligned(dst_ptr, SharedBufferAlignBytes<T>()),
                 "Shared memory needs to be 16 byte aligned.");
    _CCCL_ASSERT((static_cast<int>(size(smem_dst)) >= SharedBufferSizeBytes<T, GmemAlign>(size(gmem_src))),
                 "Shared memory destination buffer must have enough space");
    _CCCL_ASSERT(state == State::ready_to_copy || state == State::ready_to_copy_or_commit,
                 "Wait() must be called before another CopyAsync()");
    state = State::ready_to_copy_or_commit;
    if constexpr (bulk_aligned)
    {
      CopyAligned(dst_ptr, src_ptr, num_bytes);
      return {::cuda::ptr_rebind<T>(data(smem_dst)), size(gmem_src)};
    }
    else
    {
      const auto src_ptr_aligned      = ::cuda::align_down(src_ptr, minimum_align);
      const int head_padding_bytes    = src_ptr - src_ptr_aligned;
      const int padded_num_bytes      = num_bytes + head_padding_bytes;
      const int padded_num_bytes_bulk = ::cuda::round_down(padded_num_bytes, minimum_align);
      CopyAligned(dst_ptr, src_ptr_aligned, padded_num_bytes_bulk);
      // Peeling
      static_assert(block_threads >= 16);

      if (const int idx = padded_num_bytes_bulk + linear_tid; idx < padded_num_bytes)
      {
        dst_ptr[idx] = src_ptr_aligned[idx];
      }
      return {::cuda::ptr_rebind<T>(dst_ptr + head_padding_bytes), size(gmem_src)};
    }
  }

  //! @brief Commit one or more @c CopyAsync() calls.
  _CCCL_DEVICE _CCCL_FORCEINLINE void Commit()
  {
    _CCCL_ASSERT(state == State::ready_to_copy_or_commit, "CopyAsync() must be called before Commit()");
    NV_DISPATCH_TARGET(
      NV_PROVIDES_SM_90,
      (if (elected) {
        ::cuda::ptx::mbarrier_arrive_expect_tx(
          ::cuda::ptx::sem_release,
          ::cuda::ptx::scope_cta,
          ::cuda::ptx::space_shared,
          &temp_storage.bar,
          num_bytes_bulk_total);
      } //
       __syncthreads();),
      NV_PROVIDES_SM_80,
      (asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(
        static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&temp_storage.bar))) : "memory");));
    state = State::committed;
  }

  //! @brief Wait for previously committed copies to arrive. Prepare for next calls to @c CopyAsync() .
  _CCCL_DEVICE _CCCL_FORCEINLINE void Wait()
  {
    _CCCL_ASSERT(state == State::committed, "Commit() must be called before Wait()");
    while (!TryWait())
      ;
    phase_parity ^= 1u;
    num_bytes_bulk_total = 0u;
    state                = State::ready_to_copy;
  }

  // Having these as static members does require using "template" in user code which is kludgy.

  //! @brief Returns the alignment needed for the shared memory destination buffer.
  //! @tparam T
  //!   Value type to be loaded.
  template <typename T>
  _CCCL_HOST_DEVICE static constexpr int SharedBufferAlignBytes()
  {
    static_assert(::cuda::std::is_trivially_copyable_v<T>);
    return (::cuda::std::max)(int{alignof(T)}, minimum_align);
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
    static_assert(::cuda::std::is_trivially_copyable_v<T>);
    static_assert(::cuda::std::has_single_bit(unsigned{GmemAlign}));
    static_assert(GmemAlign >= int{alignof(T)});
    _CCCL_ASSERT(num_items <= ::cuda::std::size_t{::cuda::std::numeric_limits<int>::max()},
                 "num_items must fit into an int");
    constexpr bool bulk_aligned = GmemAlign >= minimum_align;
    const int extra_space       = bulk_aligned ? 0 : minimum_align;
    const int num_bytes         = static_cast<int>(num_items) * int{sizeof(T)};
    return bulk_aligned ? num_bytes : (::cuda::round_up(num_bytes, minimum_align) + extra_space);
  }
};

} // namespace detail

CUB_NAMESPACE_END
