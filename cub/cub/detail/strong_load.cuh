/******************************************************************************
 * Copyright (c) 2011, Duane Merrill.  All rights reserved.
 * Copyright (c) 2011-2022, NVIDIA CORPORATION.  All rights reserved.
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
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
 * ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
 * WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL NVIDIA CORPORATION BE LIABLE FOR ANY
 * DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
 * (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
 * LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
 * ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
 * (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
 * SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 ******************************************************************************/

/**
 * @file Utilities for strong memory operations.
 */

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

CUB_NAMESPACE_BEGIN

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{

static _CCCL_DEVICE _CCCL_FORCEINLINE uint4 load_relaxed(uint4 const* ptr)
{
  uint4 retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("ld.relaxed.gpu.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(retval.x),
                  "=r"(retval.y),
                  "=r"(retval.z),
                  "=r"(retval.w) : "l"(ptr) : "memory");),
    (asm volatile("ld.cg.v4.u32 {%0, %1, %2, %3}, [%4];" : "=r"(retval.x),
                  "=r"(retval.y),
                  "=r"(retval.z),
                  "=r"(retval.w) : "l"(ptr) : "memory");));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE ulonglong2 load_relaxed(ulonglong2 const* ptr)
{
  ulonglong2 retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("ld.relaxed.gpu.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr) : "memory");),
    (asm volatile("ld.cg.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr) : "memory");));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE ushort4 load_relaxed(ushort4 const* ptr)
{
  ushort4 retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("ld.relaxed.gpu.v4.u16 {%0, %1, %2, %3}, [%4];" : "=h"(retval.x),
                  "=h"(retval.y),
                  "=h"(retval.z),
                  "=h"(retval.w) : "l"(ptr) : "memory");),
    (asm volatile("ld.cg.v4.u16 {%0, %1, %2, %3}, [%4];" : "=h"(retval.x),
                  "=h"(retval.y),
                  "=h"(retval.z),
                  "=h"(retval.w) : "l"(ptr) : "memory");));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE uint2 load_relaxed(uint2 const* ptr)
{
  uint2 retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("ld.relaxed.gpu.v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr) : "memory");),
    (asm volatile("ld.cg.v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr) : "memory");));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE unsigned long long load_relaxed(unsigned long long const* ptr)
{
  unsigned long long retval;
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("ld.relaxed.gpu.u64 %0, [%1];" : "=l"(retval) : "l"(ptr) : "memory");),
               (asm volatile("ld.cg.u64 %0, [%1];" : "=l"(retval) : "l"(ptr) : "memory");));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int load_relaxed(unsigned int const* ptr)
{
  unsigned int retval;
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("ld.relaxed.gpu.u32 %0, [%1];" : "=r"(retval) : "l"(ptr) : "memory");),
               (asm volatile("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(ptr) : "memory");));

  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE unsigned short load_relaxed(unsigned short const* ptr)
{
  unsigned short retval;
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("ld.relaxed.gpu.u16 %0, [%1];" : "=h"(retval) : "l"(ptr) : "memory");),
               (asm volatile("ld.cg.u16 %0, [%1];" : "=h"(retval) : "l"(ptr) : "memory");));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE unsigned char load_relaxed(unsigned char const* ptr)
{
  unsigned short retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  ld.relaxed.gpu.u8 datum, [%1];"
                  "  cvt.u16.u8 %0, datum;"
                  "}" : "=h"(retval) : "l"(ptr) : "memory");),
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  ld.cg.u8 datum, [%1];"
                  "  cvt.u16.u8 %0, datum;"
                  "}" : "=h"(retval) : "l"(ptr) : "memory");));
  return (unsigned char) retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE ulonglong2 load_acquire(ulonglong2 const* ptr)
{
  ulonglong2 retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("ld.acquire.gpu.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr) : "memory");),
    (asm volatile("ld.cg.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr) : "memory");
     __threadfence();));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE uint2 load_acquire(uint2 const* ptr)
{
  uint2 retval;
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("ld.acquire.gpu.v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr) : "memory");),
    (asm volatile("ld.cg.v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr) : "memory");
     __threadfence();));
  return retval;
}

static _CCCL_DEVICE _CCCL_FORCEINLINE unsigned int load_acquire(unsigned int const* ptr)
{
  unsigned int retval;
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("ld.acquire.gpu.u32 %0, [%1];" : "=r"(retval) : "l"(ptr) : "memory");),
               (asm volatile("ld.cg.u32 %0, [%1];" : "=r"(retval) : "l"(ptr) : "memory"); __threadfence();));

  return retval;
}

} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
