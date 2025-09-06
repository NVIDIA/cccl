// SPDX-FileCopyrightText: Copyright (c) 2025, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
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

#include <cub/util_ptx.cuh>
#include <cub/util_type.cuh>

#include <cuda/cmath>
#include <cuda/memory>
#include <cuda/ptx>
#include <cuda/std/__algorithm/max.h>
#include <cuda/std/bit>
#include <cuda/std/cstdint>
#include <cuda/std/span>

#include <nv/target>

CUB_NAMESPACE_BEGIN

template <int BLOCK_DIM_X, int BLOCK_DIM_Y = 1, int BLOCK_DIM_Z = 1>
struct BlockLoadToShared
{
private:
  /// Constants
  static constexpr int BLOCK_THREADS = BLOCK_DIM_X * BLOCK_DIM_Y * BLOCK_DIM_Z;
  // TODO Should this be public?
  static constexpr unsigned BLK_MINIMUM_ALIGN = 16u;

  struct _TempStorage
  {
    ::cuda::std::uint64_t bar;
  };

  /// Shared storage reference
  _TempStorage& temp_storage;

  /// Linear thread-id
  const unsigned int linear_tid{cub::RowMajorTid(BLOCK_DIM_X, BLOCK_DIM_Y, BLOCK_DIM_Z)};

  /// Internal storage allocator
  _CCCL_DEVICE _CCCL_FORCEINLINE _TempStorage& PrivateStorage()
  {
    __shared__ _TempStorage private_storage;
    return private_storage;
  }

  const bool elected{::cuda::ptx::elect_sync(~0) && linear_tid < 32u};
  uint32_t phase_parity{};
  uint32_t num_bytes_bulk_total{};

  _CCCL_DEVICE _CCCL_FORCEINLINE void Init()
  {
    if (elected)
    {
      ::cuda::ptx::mbarrier_init(&temp_storage.bar, 1);
    }
  }

  _CCCL_DEVICE_API void CopyBulkAligned(char* smem_dst, const char* gmem_src, int num_bytes)
  {
    if (elected)
    {
      ::cuda::ptx::cp_async_bulk(
        ::cuda::ptx::space_shared, ::cuda::ptx::space_global, smem_dst, gmem_src, num_bytes, &temp_storage.bar);
      num_bytes_bulk_total += num_bytes;
    }
  }

public:
  /// @smemstorage{BlockLoadToShared}
  struct TempStorage : cub::Uninitialized<_TempStorage>
  {};

  using commit_token = bool;

  //! @name Collective constructors
  //! @{

  //! @brief Collective constructor using a private static allocation of shared memory as temporary storage.
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoadToShared()
      : temp_storage(PrivateStorage())
  {
    Init();
  }

  /**
   * @brief Collective constructor using the specified memory allocation as temporary storage.
   *
   * @param[in] temp_storage
   *   Reference to memory allocation having layout type TempStorage
   */
  _CCCL_DEVICE _CCCL_FORCEINLINE BlockLoadToShared(TempStorage& temp_storage)
      : temp_storage(temp_storage.Alias())
  {
    Init();
  }

  //! @}  end member group

  _CCCL_DEVICE _CCCL_FORCEINLINE ~BlockLoadToShared()
  {
    if (elected)
    {
      // TODO: inline PTX mbarrier.inval
    }
  }

  //! @brief Copy elements from global to shared memory
  //! @tparam T
  //!   **[inferred]** Value type for this transaction
  //! @tparam GmemAlign
  //!   Guaranteed alignment in bytes of the source range (both begin and end) in global memory
  //! @param[in] smem_dst
  //!   Destination buffer in shared memory that is aligned to @c max(alignof(T),16) .
  //!   If @c GmemAlign is smaller than 16, this buffer must be big enough to hold @c
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
    static_assert(::cuda::std::has_single_bit(unsigned{GmemAlign}));
    static_assert(GmemAlign >= int{alignof(T)});
    constexpr bool bulk_aligned = GmemAlign >= BLK_MINIMUM_ALIGN;
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
    _CCCL_ASSERT(::cuda::is_aligned(dst_ptr, ::cuda::std::max(int{alignof(T)}, BLK_MINIMUM_ALIGN)),
                 "Shared memory needs to be 16 byte aligned.");
    _CCCL_ASSERT(int{size(smem_dst)}
                   >= (bulk_aligned ? num_bytes : ::cuda::round_up(num_bytes, BLK_MINIMUM_ALIGN) + BLK_MINIMUM_ALIGN),
                 "Shared memory destination buffer must have enough space");
    if constexpr (bulk_aligned)
    {
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (CopyBulkAligned(data(smem_dst), data(gmem_src), num_bytes);), ());
      return {::cuda::ptr_rebind<T>(data(smem_dst)), size(gmem_src)};
    }
    else
    {
      const auto src_ptr_aligned      = ::cuda::align_down(src_ptr, BLK_MINIMUM_ALIGN);
      const int head_padding_bytes    = src_ptr - src_ptr_aligned;
      const int padded_num_bytes      = num_bytes + head_padding_bytes;
      const int padded_num_bytes_bulk = ::cuda::round_down(padded_num_bytes, BLK_MINIMUM_ALIGN);
      NV_IF_ELSE_TARGET(NV_PROVIDES_SM_90, (CopyBulkAligned(smem_dst, gmem_src_aligned, padded_num_bytes_bulk);), ());
      // Peeling
      for (int idx = padded_num_bytes_bulk + linear_tid; idx < padded_num_bytes; idx += BLOCK_THREADS)
      {
        dst_ptr[idx] = src_ptr_aligned[idx];
      }
      return ::cuda::ptr_rebind<T>(dst_ptr + head_padding_bytes);
    }
  }

  [[nodiscard]] _CCCL_DEVICE _CCCL_FORCEINLINE commit_token Commit()
  {
    if (elected)
    {
      // TODO Return value of arrive (mbarrier state) could be used as commit token in which case phase_parity is not
      // needed. Is parity-based waiting cheaper than state based?
      ::cuda::ptx::mbarrier_arrive_expect_tx(
        ::cuda::ptx::sem_release,
        ::cuda::ptx::scope_cta,
        ::cuda::ptx::space_shared,
        &temp_storage.bar,
        num_bytes_bulk_total);
      num_bytes_bulk_total = 0;
    }
    __syncthreads();
    return phase_parity == 1u;
  }

  _CCCL_DEVICE _CCCL_FORCEINLINE void Wait(commit_token&& commit)
  {
    // TODO Should we actually count phases to allow for a better debugging experience?
    _CCCL_ASSERT(commit == (phase_parity == 1u), "Commit token needs to be from directly preceeding Commit()");
    while (!::cuda::ptx::mbarrier_try_wait_parity(&temp_storage.bar, phase_parity))
      ;
    phase_parity ^= 1u;
  }
};

CUB_NAMESPACE_END
