// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025-2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/resource/smem_ref.cuh>
#include <cub/detail/warpspeed/squad/squad.cuh>

#include <cuda/__memory/align_down.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/instructions/cp_async_bulk_commit_group.h>
#include <cuda/__ptx/instructions/cp_async_bulk_wait_group.h>
#include <cuda/__ptx/instructions/elect_sync.h>
#include <cuda/__ptx/instructions/fence.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail::warpspeed
{
#if __cccl_ptx_isa >= 860

template <typename Tp>
struct CpAsyncOobInfo
{
  // The aligned up and down pointers below must be ::cuda::std::byte*, since the nearest aligned up/down ptr may not
  // point to a multiple of sizeof(Tp). E.g. a uchar3* pointing to address 0x...5 will be aligned down to 0x...0, but
  // that's not a valid start for an uchar3 in that array. So we must express all aligned pointers in bytes here.

  ::cuda::std::byte* ptrGmem;
  ::cuda::std::byte* ptrGmemStartAlignDown;
  ::cuda::std::byte* ptrGmemStartAlignUp;
  ::cuda::std::byte* ptrGmemEnd;
  ::cuda::std::byte* ptrGmemEndAlignDown;
  ::cuda::std::byte* ptrGmemEndAlignUp;
  ::cuda::std::uint32_t overCopySizeBytes;
  ::cuda::std::uint32_t underCopySizeBytes;
  ::cuda::std::uint32_t origCopySizeBytes;
  ::cuda::std::uint32_t smemStartSkipBytes; // ptrSmem + smemStartSkipBytes will point to the first valid element copied
                                            // from ptrGmem
  ::cuda::std::uint32_t smemEndBytesAfter16BBoundary; // number of bytes after the last 16B boundary in GMEM/SMEM that
                                                      // still contains valid (partial) elements
};

template <typename Tp>
_CCCL_DEVICE_API _CCCL_FORCEINLINE CpAsyncOobInfo<Tp> prepareCpAsyncOob(Tp* ptrGmem, ::cuda::std::uint32_t sizeElem)
{
  auto ptrGmemBytes = reinterpret_cast<::cuda::std::byte*>(ptrGmem);
  auto ptrGmemEnd   = reinterpret_cast<::cuda::std::byte*>(ptrGmem + sizeElem);

  // We will copy from [ptrGmemBase, ptrGmemEnd). Both pointers have to be 16B aligned.
  ::cuda::std::byte* ptrGmemStartAlignDown = ::cuda::align_down(ptrGmemBytes, 16);
  ::cuda::std::byte* ptrGmemStartAlignUp   = ::cuda::align_up(ptrGmemBytes, 16);
  ::cuda::std::byte* ptrGmemEndAlignUp     = ::cuda::align_up(ptrGmemEnd, 16);
  ::cuda::std::byte* ptrGmemEndAlignDown   = ::cuda::align_down(ptrGmemEnd, 16);

  // Compute the final copy size in bytes. It can be either sizeElem or sizeElem + 16 / sizeof(T).
  const auto origCopySizeBytes = static_cast<::cuda::std::uint32_t>(sizeof(Tp) * sizeElem);
  const auto overCopySizeBytes = static_cast<::cuda::std::uint32_t>(ptrGmemEndAlignUp - ptrGmemStartAlignDown);
  auto underCopySizeBytes      = static_cast<::cuda::std::uint32_t>(ptrGmemEndAlignDown - ptrGmemStartAlignUp);
  if (origCopySizeBytes < underCopySizeBytes)
  {
    // If ptrGmemStart and ptrGmemEnd are aligned to [1, .., 15] bytes, then
    // when we align the one up and the other down we get overflow. We check for
    // that here. In that case, the undercopy size is zero.
    underCopySizeBytes = 0;
  }

  _CCCL_ASSERT(overCopySizeBytes % 16 == 0, "");
  _CCCL_ASSERT(underCopySizeBytes % 16 == 0, "");

  return {
    ptrGmemBytes,
    ptrGmemStartAlignDown,
    ptrGmemStartAlignUp,
    ptrGmemEnd,
    ptrGmemEndAlignDown,
    ptrGmemEndAlignUp,
    overCopySizeBytes,
    underCopySizeBytes,
    origCopySizeBytes,
    static_cast<::cuda::std::uint32_t>(ptrGmemBytes - ptrGmemStartAlignDown),
    static_cast<::cuda::std::uint32_t>(ptrGmemEnd - ptrGmemEndAlignDown),
  };
}

template <typename ResourceTp, typename Tp>
_CCCL_DEVICE_API void squadLoadBulk(Squad squad, SmemRef<ResourceTp>& refDestSmem, CpAsyncOobInfo<Tp> cpAsyncOobInfo)
{
  ::cuda::std::byte* ptrSmem = refDestSmem.data().inout;
  _CCCL_ASSERT(::cuda::is_aligned(ptrSmem, 16), "");
  ::cuda::std::uint64_t* ptrBar = refDestSmem.ptrCurBarrierRelease();

  if constexpr (alignof(Tp) >= 16)
  {
    // for alignments larger than 16, we can just bulk copy, even just a single element
    if (squad.isLeaderThread())
    {
      ::cuda::ptx::cp_async_bulk(
        ::cuda::std::conditional_t<__cccl_ptx_isa >= 860, ::cuda::ptx::space_shared_t, ::cuda::ptx::space_cluster_t>{},
        ::cuda::ptx::space_global,
        ptrSmem,
        cpAsyncOobInfo.ptrGmem,
        cpAsyncOobInfo.origCopySizeBytes,
        ptrBar);
    }
    refDestSmem.squadIncreaseTxCount(squad, cpAsyncOobInfo.underCopySizeBytes);
  }
  else
  {
    // for alignments smaller than 16, we can overcopy but need to declare the ignored bytes left and right
#  if __cccl_ptx_isa >= 920
    if (squad.isLeaderThread())
    {
      ::cuda::ptx::cp_async_bulk_ignore_oob(
        ::cuda::ptx::space_shared,
        ::cuda::ptx::space_global,
        ptrSmem,
        cpAsyncOobInfo.ptrGmemStartAlignDown,
        cpAsyncOobInfo.overCopySizeBytes,
        /* ignore left */ cpAsyncOobInfo.smemStartSkipBytes,
        /* ignore right */ cpAsyncOobInfo.ptrGmemEndAlignUp - cpAsyncOobInfo.ptrGmemEnd,
        ptrBar);
    }
    refDestSmem.squadIncreaseTxCount(squad, cpAsyncOobInfo.overCopySizeBytes);
#  else // __cccl_ptx_isa >= 920
    // if we don't have cp_async_bulk_ignore_oob, we have to undercopy and copy head and tail elements manually

    // handle small copies first. If we have less than 16 bytes we may not straddle a 16B boundary
    if (cpAsyncOobInfo.origCopySizeBytes < 16)
    {
      const auto elemCount = cpAsyncOobInfo.origCopySizeBytes / sizeof(Tp);
      _CCCL_ASSERT(elemCount <= squad.threadCount(), "");
      if (squad.threadRank() < elemCount)
      {
        reinterpret_cast<Tp*>(ptrSmem + cpAsyncOobInfo.smemStartSkipBytes)[squad.threadRank()] =
          reinterpret_cast<const Tp*>(cpAsyncOobInfo.ptrGmem)[squad.threadRank()];
      }
      return; // no bulk copy has been performed so we don't need to update the tx count of any barrier
    }

    // copies larger than 16 byte which straddle at least one 16B boundary, so we have dedicated start and end copies

    const bool doStartCopy = cpAsyncOobInfo.smemStartSkipBytes > 0;

    ::cuda::std::byte* ptrSmemMiddle = ptrSmem;
    if (doStartCopy)
    {
      ptrSmemMiddle += 16;
    }

    // TODO(bgruber): we could skip the middle if underCopySizeBytes is zero
    if (squad.isLeaderThread())
    {
      ::cuda::ptx::cp_async_bulk(
        ::cuda::std::conditional_t<__cccl_ptx_isa >= 860, ::cuda::ptx::space_shared_t, ::cuda::ptx::space_cluster_t>{},
        ::cuda::ptx::space_global,
        ptrSmemMiddle,
        cpAsyncOobInfo.ptrGmemStartAlignUp,
        cpAsyncOobInfo.underCopySizeBytes,
        ptrBar);
    }
    refDestSmem.squadIncreaseTxCount(squad, cpAsyncOobInfo.underCopySizeBytes);

    // we cannot use Tp to load the head and tail elements, because sizeof(Tp) may be larger than alignof(Tp)
    using load_word_t = ::cuda::__make_nbit_uint_t<alignof(Tp) * CHAR_BIT>;

    const int head_elements = (cpAsyncOobInfo.ptrGmemStartAlignUp - cpAsyncOobInfo.ptrGmem) / sizeof(load_word_t);
    const int tail_elements = (cpAsyncOobInfo.ptrGmemEnd - cpAsyncOobInfo.ptrGmemEndAlignDown) / sizeof(load_word_t);
    _CCCL_ASSERT(head_elements <= squad.threadCount(), "");
    _CCCL_ASSERT(tail_elements <= squad.threadCount(), "");
    load_word_t head_value, tail_value;
    if (squad.threadRank() < head_elements)
    {
      head_value = reinterpret_cast<const load_word_t*>(cpAsyncOobInfo.ptrGmem)[squad.threadRank()];
    }
    if (squad.threadRank() < tail_elements)
    {
      tail_value = reinterpret_cast<const load_word_t*>(cpAsyncOobInfo.ptrGmemEndAlignDown)[squad.threadRank()];
    }

    if (squad.threadRank() < head_elements)
    {
      reinterpret_cast<load_word_t*>(ptrSmem + cpAsyncOobInfo.smemStartSkipBytes)[squad.threadRank()] = head_value;
    }
    if (squad.threadRank() < tail_elements)
    {
      reinterpret_cast<load_word_t*>(ptrSmemMiddle + cpAsyncOobInfo.underCopySizeBytes)[squad.threadRank()] =
        tail_value;
    }
#  endif // __cccl_ptx_isa >= 920
  }
}

template <typename OutputT>
_CCCL_DEVICE_API void
squadStoreBulkSync(Squad squad, CpAsyncOobInfo<OutputT> cpAsyncOobInfo, const ::cuda::std::byte* srcSmem)
{
  // This function performs either 1 copy, or three copies, depending on the
  // size and alignment of the output tile in global memory.
  //
  // If the output tile is contained in a single 16-byte aligned and sized region, then we
  // only perform a single masked copy.
  //
  // If the output tile is larger than 16 bytes or straddles two 16-byte aligned and sized regions, then
  // we perform up to three copies:
  // - One copy for the first up to 15 bytes at the start of the region.
  // - One copy that starts at a 16-byte aligned address and ends at the latest 16-byte aligned address.
  // - One copy that cleans up the last up to 15 bytes.
  if (squad.isLeaderWarp())
  {
    // Acquire shared memory in async proxy
    // Perform fence.proxy.async with full warp to avoid BSSY+BSYNC
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);

    // FIXME(bgruber): for some reason the optimizer propagates some information from the computation of
    // overCopySizeBytes to the masked bulk copy below and generates an unaligned access error.
    // The artificial read modification of overCopySizeBytes prevents the propagation here works around this.
    // It also solves the issue described in nvbug 5848313 by accident on nvcc 13.2+
    asm volatile("" : "+r"(cpAsyncOobInfo.overCopySizeBytes));

    const bool doStartCopy  = cpAsyncOobInfo.smemStartSkipBytes > 0;
    const bool doEndCopy    = cpAsyncOobInfo.smemEndBytesAfter16BBoundary > 0;
    const bool doMiddleCopy = cpAsyncOobInfo.ptrGmemStartAlignUp != cpAsyncOobInfo.ptrGmemEndAlignUp;

    constexpr ::cuda::std::uint16_t byteMask  = 0xFFFF;
    const ::cuda::std::uint16_t byteMaskStart = byteMask << cpAsyncOobInfo.smemStartSkipBytes;
    const ::cuda::std::uint16_t byteMaskEnd   = byteMask >> (16 - cpAsyncOobInfo.smemEndBytesAfter16BBoundary);
    // byteMaskStart contains zeroes at the left
#  if _CCCL_CUDA_COMPILER(NVCC, >=, 13, 2)
    const ::cuda::std::uint16_t byteMaskSmall = byteMaskStart & byteMaskEnd;
#  else // _CCCL_CUDA_COMPILER(NVCC, >=, 13, 2)
    // `ptxas fatal   : (C7907) Internal compiler error`, see nvbug 5848313
    const ::cuda::std::uint16_t byteMaskSmall =
      byteMaskStart & (byteMask >> (16 - (cpAsyncOobInfo.ptrGmemEnd - cpAsyncOobInfo.ptrGmemStartAlignDown)));
#  endif // _CCCL_CUDA_COMPILER(NVCC, >=, 13, 2)

    const ::cuda::std::byte* ptrSmemMiddle = srcSmem;
    if (doStartCopy)
    {
      ptrSmemMiddle += 16;
    }

    if (doMiddleCopy)
    {
      // Copy the middle part. Starting at byte 0 or 16 in shared memory. This
      // is the large copy. We perform this one first, so that the compiler can
      // (hopefully) hide all the arithmetic behind this instruction.
      if (::cuda::ptx::elect_sync(~0))
      {
        ::cuda::ptx::cp_async_bulk(
          ::cuda::ptx::space_global,
          ::cuda::ptx::space_shared,
          cpAsyncOobInfo.ptrGmemStartAlignUp,
          ptrSmemMiddle,
          cpAsyncOobInfo.underCopySizeBytes);
      }
      if (doStartCopy)
      {
        // Copy a subset of the first 16 bytes
        if (::cuda::ptx::elect_sync(~0))
        {
          ::cuda::ptx::cp_async_bulk_cp_mask(
            ::cuda::ptx::space_global,
            ::cuda::ptx::space_shared,
            cpAsyncOobInfo.ptrGmemStartAlignDown,
            srcSmem,
            /*size*/ 16,
            byteMaskStart);
        }
      }
      if (doEndCopy)
      {
        // Copy a subset of the last 16 bytes
        if (::cuda::ptx::elect_sync(~0))
        {
          ::cuda::ptx::cp_async_bulk_cp_mask(
            ::cuda::ptx::space_global,
            ::cuda::ptx::space_shared,
            cpAsyncOobInfo.ptrGmemEndAlignDown,
            ptrSmemMiddle + cpAsyncOobInfo.underCopySizeBytes,
            /*size*/ 16,
            byteMaskEnd);
        }
      }
    }
    else
    {
      // Copy a subset of the first 16 bytes
      if (::cuda::ptx::elect_sync(~0))
      {
        ::cuda::ptx::cp_async_bulk_cp_mask(
          ::cuda::ptx::space_global,
          ::cuda::ptx::space_shared,
          cpAsyncOobInfo.ptrGmemStartAlignDown,
          srcSmem,
          /*size*/ 16,
          byteMaskSmall);
      }
    }
    // Commit and wait for store to have completed reading from shared memory
    ::cuda::ptx::cp_async_bulk_commit_group();
    ::cuda::ptx::cp_async_bulk_wait_group_read(::cuda::ptx::n32_t<0>{});
  }
}

#endif // __cccl_ptx_isa >= 860

template <typename InputT, typename AccumT, int ElemPerThread>
_CCCL_DEVICE_API void squadLoadSmem(Squad squad, AccumT (&outReg)[ElemPerThread], const InputT* smemBuf)
{
  for (int i = 0; i < ElemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * ElemPerThread + i;
    outReg[i]          = smemBuf[elem_idx];
  }
}

template <typename OutputT, typename AccumT, int ElemPerThread>
_CCCL_DEVICE_API void squadStoreSmem(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[ElemPerThread])
{
  for (int i = 0; i < ElemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * ElemPerThread + i;
    smemBuf[elem_idx]  = inReg[i];
  }
}

template <typename OutputT, typename AccumT, int ElemPerThread>
_CCCL_DEVICE_API void
squadStoreSmemPartial(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[ElemPerThread], int beginIndex, int endIndex)
{
  for (int i = 0; i < ElemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * ElemPerThread + i;
    if (beginIndex <= elem_idx && elem_idx < endIndex)
    {
      smemBuf[elem_idx - beginIndex] = inReg[i];
    }
  }
}
} // namespace detail::warpspeed

CUB_NAMESPACE_END
