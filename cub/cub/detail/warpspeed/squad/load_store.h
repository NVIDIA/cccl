// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/warpspeed/squad/squad.h>

#include <cuda/__memory/align_down.h>
#include <cuda/__memory/align_up.h>
#include <cuda/__ptx/instructions/cp_async_bulk.h>
#include <cuda/__ptx/instructions/cp_async_bulk_commit_group.h>
#include <cuda/__ptx/instructions/cp_async_bulk_wait_group.h>
#include <cuda/__ptx/instructions/elect_sync.h>
#include <cuda/__ptx/instructions/fence.h>
#include <cuda/std/cstdint>

CUB_NAMESPACE_BEGIN

namespace detail
{
#if __cccl_ptx_isa >= 860

struct CpAsyncOobInfo
{
  char* ptrGmem;
  char* ptrGmemStartAlignDown;
  char* ptrGmemStartAlignUp;
  char* ptrGmemEnd;
  char* ptrGmemEndAlignDown;
  char* ptrGmemEndAlignUp;
  uint32_t overCopySizeBytes;
  uint32_t underCopySizeBytes;
  uint32_t origCopySizeBytes;
  uint32_t smemStartOffsetElem;
  uint32_t smemStartOffsetBytes;
  uint32_t smemEndOffsetElem;
  uint32_t smemEndOffsetBytes;
};

template <typename Tp>
_CCCL_DEVICE_API inline CpAsyncOobInfo prepareCpAsyncOob(const Tp* ptrGmem, uint32_t sizeElem)
{
  // We will copy from [ptrGmemBase, ptrGmemEnd). Both pointers have to be 16B
  // aligned.
  const Tp* ptrGmemStartAlignDown = cuda::align_down(ptrGmem, ::cuda::std::size_t(16));
  const Tp* ptrGmemStartAlignUp   = cuda::align_up(ptrGmem, ::cuda::std::size_t(16));
  const Tp* ptrGmemEnd            = ptrGmem + sizeElem;
  const Tp* ptrGmemEndAlignUp     = cuda::align_up(ptrGmemEnd, ::cuda::std::size_t(16));
  const Tp* ptrGmemEndAlignDown   = cuda::align_down(ptrGmemEnd, ::cuda::std::size_t(16));

  // Compute the final copy size in bytes. It can be either sizeElem or sizeElem + 16 / sizeof(T).
  uint32_t origCopySizeBytes  = static_cast<uint32_t>(sizeof(Tp) * sizeElem);
  uint32_t overCopySizeBytes  = static_cast<uint32_t>(sizeof(Tp) * (ptrGmemEndAlignUp - ptrGmemStartAlignDown));
  uint32_t underCopySizeBytes = static_cast<uint32_t>(sizeof(Tp) * (ptrGmemEndAlignDown - ptrGmemStartAlignUp));
  if (origCopySizeBytes < underCopySizeBytes)
  {
    // If ptrGmemStart and ptrGmemEnd are aligned to [1, .., 15] bytes, then
    // when we align the one up and the other down we get overflow. We check for
    // that here. In that case, the undercopy size is zero.
    underCopySizeBytes = 0;
  }
  // The offset in elements to the first valid element in shared memory.
  // ptrSmem + smemOffsetElem will point to the element copied from ptrGmem.
  uint32_t smemStartOffsetElem = static_cast<uint32_t>(ptrGmem - ptrGmemStartAlignDown);
  // The offset in elements between ptrGmemEnd and ptrGmemEndAlignDown
  uint32_t smemEndOffsetElem = static_cast<uint32_t>(ptrGmemEnd - ptrGmemEndAlignDown);

  return CpAsyncOobInfo{
    .ptrGmem               = (char*) ptrGmem,
    .ptrGmemStartAlignDown = (char*) ptrGmemStartAlignDown,
    .ptrGmemStartAlignUp   = (char*) ptrGmemStartAlignUp,
    .ptrGmemEnd            = (char*) ptrGmemEnd,
    .ptrGmemEndAlignDown   = (char*) ptrGmemEndAlignDown,
    .ptrGmemEndAlignUp     = (char*) ptrGmemEndAlignUp,
    .overCopySizeBytes     = overCopySizeBytes,
    .underCopySizeBytes    = underCopySizeBytes,
    .origCopySizeBytes     = static_cast<uint32_t>(sizeof(Tp) * sizeElem),
    .smemStartOffsetElem   = smemStartOffsetElem,
    .smemStartOffsetBytes  = static_cast<uint32_t>(sizeof(Tp) * smemStartOffsetElem),
    .smemEndOffsetElem     = smemEndOffsetElem,
    .smemEndOffsetBytes    = static_cast<uint32_t>(sizeof(Tp) * smemEndOffsetElem),
  };
}

template <typename Tp>
_CCCL_DEVICE_API inline void squadLoadBulk(const Squad& squad, SmemRef<Tp>& refDestSmem, CpAsyncOobInfo cpAsyncOobInfo)
{
  void* ptrSmem    = refDestSmem.data().in;
  uint64_t* ptrBar = refDestSmem.ptrCurBarrierRelease();

  if (squad.isLeaderThread())
  {
    ::cuda::ptx::cp_async_bulk(
      ::cuda::ptx::space_cluster,
      ::cuda::ptx::space_global,
      ptrSmem,
      cpAsyncOobInfo.ptrGmemStartAlignDown,
      cpAsyncOobInfo.overCopySizeBytes,
      ptrBar);
  }
  refDestSmem.squadIncreaseTxCount(squad, cpAsyncOobInfo.overCopySizeBytes);
}

template <typename OutputT>
_CCCL_DEVICE_API inline void squadStoreBulkSync(const Squad& squad, CpAsyncOobInfo cpAsyncOobInfo, OutputT* srcSmem)
{
  // This function performs either 1 copy, or three copies, depending on the
  // size and alignment of the output tile in global memory.
  //
  // If the output tile is contained in a single 16-byte aligned region, then we
  // only perform a single masked copy.
  //
  // If the output tile is larger or straddles two 16-byte aligned regions, then
  // we perform up to three copies:
  // - One copy for the first up to 15 bytes at the start of the region.
  // - One copy that starts at a 16-byte aligned address and ends at the latest 16-byte aligned address.
  // - One copy that cleans up the last up to 15 bytes.
  if (squad.isLeaderWarp())
  {
    // Acquire shared memory in async proxy
    // Perform fence.proxy.async with full warp to avoid BSSY+BSYNC
    ::cuda::ptx::fence_proxy_async(::cuda::ptx::space_shared);

    bool doStartCopy  = cpAsyncOobInfo.smemStartOffsetBytes > 0;
    bool doEndCopy    = cpAsyncOobInfo.smemEndOffsetBytes > 0;
    bool doMiddleCopy = cpAsyncOobInfo.ptrGmemStartAlignUp != cpAsyncOobInfo.ptrGmemEndAlignUp;

    uint16_t byteMask      = 0xFFFF;
    uint16_t byteMaskStart = byteMask << cpAsyncOobInfo.smemStartOffsetBytes;
    uint16_t byteMaskEnd   = byteMask >> (16 - cpAsyncOobInfo.smemEndOffsetBytes);
    // byteMaskStart contains zeroes at the left.
    uint16_t byteMaskSmall =
      byteMaskStart & (byteMask >> (16 - (cpAsyncOobInfo.ptrGmemEnd - cpAsyncOobInfo.ptrGmemStartAlignDown)));

    char* ptrSmemMiddle = (char*) srcSmem;
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
            ((char*) cpAsyncOobInfo.ptrGmemStartAlignUp) + cpAsyncOobInfo.underCopySizeBytes,
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

template <typename InputT, typename AccumT, int elemPerThread>
_CCCL_DEVICE_API inline void squadLoadSmem(Squad squad, AccumT (&outReg)[elemPerThread], const InputT* smemBuf)
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * elemPerThread + i;
    outReg[i]          = smemBuf[elem_idx];
  }
}

template <typename OutputT, typename AccumT, int elemPerThread>
_CCCL_DEVICE_API inline void squadStoreSmem(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[elemPerThread])
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * elemPerThread + i;
    smemBuf[elem_idx]  = inReg[i];
  }
}

template <typename OutputT, typename AccumT, int elemPerThread>
_CCCL_DEVICE_API inline void
squadStoreSmemPartial(Squad squad, OutputT* smemBuf, const AccumT (&inReg)[elemPerThread], int beginIndex, int endIndex)
{
  for (int i = 0; i < elemPerThread; ++i)
  {
    const int elem_idx = squad.threadRank() * elemPerThread + i;
    if (beginIndex <= elem_idx && elem_idx < endIndex)
    {
      smemBuf[elem_idx - beginIndex] = inReg[i];
    }
  }
}
} // namespace detail

CUB_NAMESPACE_END
