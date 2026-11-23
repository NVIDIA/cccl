// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#ifdef _CCCL_IMPLICIT_SYSTEM_HEADER_GCC
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/iket_support.cuh>
#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/squad/squad.cuh>

#include <cuda/__memory/align_down.h>
#include <cuda/__memory/is_aligned.h>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/instructions/cp_async_bulk_commit_group.h>
#include <cuda/__ptx/instructions/cp_async_bulk_wait_group.h>
#include <cuda/__ptx/instructions/elect_sync.h>
#include <cuda/__ptx/instructions/fence.h>
#include <cuda/async_copy>
#include <cuda/std/__cstddef/byte.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/cstdint>

#include <nv/target>

#if __cccl_ptx_isa < 920
#  include <cuda/std/__type_traits/make_nbit_int.h>
#endif // __cccl_ptx_isa < 920

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
_CCCL_IKET_CREATE_PUSH_POP_RANGE(LoadSmem);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(StoreSmem);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(LoadBulk);
_CCCL_IKET_CREATE_PUSH_POP_RANGE(StoreBulk);

#if __cccl_ptx_isa >= 860

template <typename ResourceTp, typename Tp>
_CCCL_DEVICE_API void squadLoadBulk(
  Squad squad, SmemRef<ResourceTp>& refDestSmem, ::cuda::device::aligned_ptr<Tp> ptrGmem, ::cuda::std::uint32_t bytes)
{
  _CCCL_IKET_RANGE_PUSH(LoadBulk);
  ::cuda::std::byte* ptrSmem = refDestSmem.data().inout;
  _CCCL_ASSERT(::cuda::is_aligned(ptrSmem, 16), "");
  ::cuda::std::uint64_t* ptrBar = refDestSmem.ptrCurBarrierRelease();

  if constexpr (alignof(Tp) >= 16)
  {
    // for alignments larger than 16, we can just bulk copy, even just a single element
    _CCCL_ASSERT(ptrGmem.head_padding == 0, "");
    if (squad.isLeaderThread())
    {
      ::cuda::device::bulk_load(ptrSmem, ptrGmem.data(), bytes, ptrBar);
    }
    refDestSmem.squadIncreaseTxCount(squad, bytes);
  }
  else
  {
    // for alignments smaller than 16, we can overcopy but need to declare the ignored bytes left and right
#  if __cccl_ptx_isa >= 920
    ::cuda::std::uint32_t copied = 0;
    if (squad.isLeaderThread())
    {
      copied =
        ::cuda::device::bulk_load_unaligned_ignore_oob(ptrSmem, ptrGmem.ptr, bytes, ptrGmem.head_padding, ptrBar);
    }
    refDestSmem.squadIncreaseTxCount(squad, copied);
#  else // __cccl_ptx_isa >= 920
    // if we don't have cp_async_bulk_ignore_oob, we have to undercopy and copy head and tail elements manually
    ::cuda::std::uint32_t copied = 0;
    ::cuda::device::bulk_load_unaligned_precise<16, alignof(Tp)>(
      ptrSmem + ptrGmem.head_padding,
      ptrGmem.data(),
      bytes,
      ptrGmem.head_padding,
      ptrBar,
      squad.threadRank(),
      squad.isLeaderThread(),
      copied);
    refDestSmem.squadIncreaseTxCount(squad, copied);
#  endif // __cccl_ptx_isa >= 920
  }
  _CCCL_IKET_RANGE_POP();
}

template <typename OutputT>
_CCCL_DEVICE_API void squadStoreBulkSync(
  Squad squad,
  ::cuda::device::aligned_ptr<OutputT> ptrGmem,
  ::cuda::std::uint32_t bytes,
  const ::cuda::std::byte* srcSmem)
{
  if (squad.isLeaderWarp())
  {
    _CCCL_IKET_RANGE_PUSH(StoreBulk);
    ::cuda::device::bulk_store_unaligned(
      const_cast<OutputT*>(ptrGmem.data()), srcSmem, bytes, squad.threadRank(), ::cuda::ptx::elect_sync(~0));
    // wait for the stores to have completed reading from shared memory
    ::cuda::ptx::cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<0>{});
    _CCCL_IKET_RANGE_POP();
  }
}

#endif // __cccl_ptx_isa >= 860

template <typename InputT, typename AccumT, int ElemPerThread>
_CCCL_DEVICE_API void squadLoadSmem(Squad squad, AccumT (&outReg)[ElemPerThread], const InputT* smemBuf)
{
  _CCCL_IKET_RANGE_PUSH(LoadSmem);
  for (int i = 0; i < ElemPerThread; ++i)
  {
    const int elem_idx = (squad.threadRank() * ElemPerThread) + i;
    outReg[i]          = smemBuf[elem_idx];
  }
  _CCCL_IKET_RANGE_POP();
}

template <typename OutputT, typename AccumT, int ElemPerThread>
_CCCL_DEVICE_API void squadStoreSmem(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[ElemPerThread])
{
  _CCCL_IKET_RANGE_PUSH(StoreSmem);
  for (int i = 0; i < ElemPerThread; ++i)
  {
    const int elem_idx = (squad.threadRank() * ElemPerThread) + i;
    smemBuf[elem_idx]  = inReg[i];
  }
  _CCCL_IKET_RANGE_POP();
}

template <typename OutputT, typename AccumT, int ElemPerThread>
_CCCL_DEVICE_API void
squadStoreSmemPartial(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[ElemPerThread], int beginIndex, int endIndex)
{
  _CCCL_IKET_RANGE_PUSH(StoreSmem);
  for (int i = 0; i < ElemPerThread; ++i)
  {
    const int elem_idx = (squad.threadRank() * ElemPerThread) + i;
    if (beginIndex <= elem_idx && elem_idx < endIndex)
    {
      smemBuf[elem_idx - beginIndex] = inReg[i];
    }
  }
  _CCCL_IKET_RANGE_POP();
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
