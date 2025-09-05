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

//! @rst
//! The @c BlockLoadToShared class provides a :ref:`collective <collective-primitives>` method for asynchronously
//! loading data from global to shared memory on Ampere and newer architectures.
//!
//! Overview
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - Given one or more spans of input elements in global memory and buffers in shared memory, this primitve
//!   asynchronously copies the elements to shared memory and takes care of synchronization.
//! - @rowmajor
//! - Shared memory buffers are assumed to be 16 byte aligned. When copying types requiring an even
//!   higher alignment, the shared memory buffer is assumed to be aligned to that.
//! - Global memory spans are assumed to be aligned according to the value type. Higher alignment guarantees can
//!   optionally be specified.
//!
//! Performance Considerations
//! +++++++++++++++++++++++++++++++++++++++++++++
//!
//! - Uses special instructions/hardware acceleration (TMA for Hopper+, otherwise LDGSTS).
//! - By guaranteeing 16 byte alignment for the global span (both start and end/size must be a multiple), a faster path
//! is taken.
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
    // mbarrier state
    ::cuda::std::uint64_t bar;
  };

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  const unsigned int linear_tid{cub::RowMajorTid(BlockDimX, BlockDimY, BlockDimZ)};

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  // Thread selection for non-collective operations
  const bool elected{::cuda::ptx::elect_sync(~0) && linear_tid < 32u};
  // Keep track of current mbarrier phase for waiting.
  uint32_t phase_parity{};
  // Keep track of the amount of bytes from multiple transactions for Commit() (only needed for TMA).
  // Also used to check for proper ordering of member function calls in debug mode.
  uint32_t num_bytes_bulk_total{};

  _CCCL_DEVICE _CCCL_FORCEINLINE void Init()
  {
    {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90,
                        (if (elected) { ::cuda::ptx::mbarrier_init(&temp_storage.bar, 1); }), //
                        (if (elected) { ::cuda::ptx::mbarrier_init(&temp_storage.bar, block_threads); } //
                         __syncthreads();));
    }
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void CopyAligned(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (num_bytes_bulk_total += num_bytes; //
       if (elected) {
         ::cuda::ptx::cp_async_bulk(
           ::cuda::ptx::space_shared, ::cuda::ptx::space_global, smem_dst, gmem_src, num_bytes, &temp_storage.bar);
       }),
      (for (int offset = linear_tid * minimum_align; offset < num_bytes; offset += block_threads * minimum_align) {
        const auto thread_src = gmem_src + offset;
        const auto thread_dst = smem_dst + offset;
        // LDGSTS stolen from cuda::memcpy_async, assumes 16 byte alignment to avoid L1 (.cg)
        asm volatile(R"XYZ(
        {
          .reg .u64 tmp;
          .reg .u32 dst;

          cvta.to.shared.u64 tmp, %0;
          cvt.u32.u64 dst, tmp;
          cvta.to.global.u64 tmp, %1;
          cp.async.cg.shared.global [dst], [tmp], 16, 16;
        }
        )XYZ"
                     :
                     : "l"(thread_dst), "l"(thread_src)
                     : "memory");
      }));
  }

  // WAR for waiting on SM_80
  _CCCL_DEVICE _CCCL_FORCEINLINE bool try_wait()
  {
    // TODO Add backoff at least for SM_80?
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (return ::cuda::ptx::mbarrier_try_wait_parity(&temp_storage.bar, phase_parity);),
      (const bool done = ::cuda::ptx::mbarrier_test_wait_parity(&temp_storage.bar, phase_parity); //
       if (!done) { __nanosleep(0); } //
       return done;));
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
    Init();
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
    Init();
  }

  //! @}  end member group

  //! @brief Collective destructor invalidates underlying @c mbarrier enabling reuse of its temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE ~BlockLoadToShared()
  {
    if (elected)
    {
      // Stolen from cuda::barrier
      asm volatile("mbarrier.inval.shared.b64 [%0];" ::"r"(static_cast<::cuda::std::uint32_t>(
        ::__cvta_generic_to_shared(&temp_storage.bar)))
                   : "memory");
    }
    // Make sure the memory is usable
    // TODO Should this sync be left to the user like after execution of other block-wide primitives?
    __syncthreads();
  }

  //! @brief Copy elements from global to shared memory
  //! @tparam T
  //!   **[inferred]** Value type for this transaction
  //! @tparam GmemAlign
  //!   Guaranteed alignment in bytes of the source range (both begin and end) in global memory
  //! @param[in] smem_dst
  //!   Destination buffer in shared memory that is aligned to @c max(alignof(T),16) .
  //!   If @c GmemAlign is smaller than 16, this buffer must be big enough to hold TODO
  //! @param[in] gmem_src
  //!   Source range in global memory, determines the size of the transaction
  //! @return
  //!   The copied range in shared memory (same size as @c gmem_src )
  // TODO Order of arguments?
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
    const int num_bytes = int{sizeof(T)} * int{size(gmem_src)};
    const auto dst_ptr  = ::cuda::ptr_rebind<char>(data(smem_dst));
    const auto src_ptr  = ::cuda::ptr_rebind<char>(data(gmem_src));
    _CCCL_ASSERT(::cuda::device::is_address_from(dst_ptr, ::cuda::device::address_space::shared),
                 "Destination address needs to point to shared memory");
    _CCCL_ASSERT(::cuda::device::is_address_from(src_ptr, ::cuda::device::address_space::global),
                 "Source address needs to point to global memory");
    _CCCL_ASSERT(::cuda::is_aligned(src_ptr, GmemAlign),
                 "Begin of global memory range needs to be aligned according to GmemAlign.");
    _CCCL_ASSERT(::cuda::is_aligned(src_ptr + num_bytes, GmemAlign),
                 "End of global memory range needs to be aligned according to GmemAlign.");
    _CCCL_ASSERT(::cuda::is_aligned(dst_ptr, ::cuda::std::max(int{alignof(T)}, minimum_align)),
                 "Shared memory needs to be 16 byte aligned.");
    _CCCL_ASSERT(
      int{size(smem_dst)} >= (bulk_aligned ? num_bytes : ::cuda::round_up(num_bytes, minimum_align) + minimum_align),
      "Shared memory destination buffer must have enough space");
    _CCCL_ASSERT(num_bytes_bulk_total >= 0, "Wait() must be called before another CopyAsync()");
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
      // TODO If we would static_assert(block_threads >= 16) we could avoid the loop in favor of a conditional.
      for (int idx = padded_num_bytes_bulk + linear_tid; idx < padded_num_bytes; idx += block_threads)
      {
        dst_ptr[idx] = src_ptr_aligned[idx];
      }
      return {::cuda::ptr_rebind<T>(dst_ptr + head_padding_bytes), size(gmem_src)};
    }
  }

  //! @brief Commit one or more @c CopyAsync() calls.
  _CCCL_DEVICE _CCCL_FORCEINLINE void Commit()
  {
    NV_IF_ELSE_TARGET(
      NV_PROVIDES_SM_90,
      (
        // TODO Should empty copies be "allowed"?
        _CCCL_ASSERT(num_bytes_bulk_total > 0, "CopyAsync() must be called before Commit()"); //
        if (elected) {
          ::cuda::ptx::mbarrier_arrive_expect_tx(
            ::cuda::ptx::sem_release,
            ::cuda::ptx::scope_cta,
            ::cuda::ptx::space_shared,
            &temp_storage.bar,
            num_bytes_bulk_total);
        } //
        __syncthreads();),
      (asm volatile("cp.async.mbarrier.arrive.noinc.shared.b64 [%0];" ::"r"(
        static_cast<::cuda::std::uint32_t>(::__cvta_generic_to_shared(&temp_storage.bar))) : "memory");));
    // Set to "committed state" for debugging/assert in Wait(), should be optimized away in release builds
    num_bytes_bulk_total = ~0u;
  }

  //! @brief Wait for previously committed copies to arrive. Prepare for next calls to @c CopyAsync() .
  _CCCL_DEVICE _CCCL_FORCEINLINE void Wait()
  {
    _CCCL_ASSERT(num_bytes_bulk_total == ~0u, "Commit() must be called before Wait()");
    while (!try_wait())
      ;
    phase_parity ^= 1u;
    num_bytes_bulk_total = 0u;
  }
};

CUB_NAMESPACE_END
