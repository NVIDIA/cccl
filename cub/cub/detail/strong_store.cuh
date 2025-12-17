// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2022, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

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
static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(uint4* ptr, uint4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.relaxed.gpu.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "r"(val.x),
                  "r"(val.y),
                  "r"(val.z),
                  "r"(val.w) : "memory");),
    (asm volatile(
       "st.cg.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(ulonglong2* ptr, ulonglong2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");),
               (asm volatile("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(ushort4* ptr, ushort4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.relaxed.gpu.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "h"(val.x),
                  "h"(val.y),
                  "h"(val.z),
                  "h"(val.w) : "memory");),
    (asm volatile(
       "st.cg.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(uint2* ptr, uint2 val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");),
               (asm volatile("st.cg.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned long long* ptr, unsigned long long val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");),
               (asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned int* ptr, unsigned int val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");),
               (asm volatile("st.cg.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned short* ptr, unsigned short val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.relaxed.gpu.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");),
               (asm volatile("st.cg.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");));
}

static _CCCL_DEVICE _CCCL_FORCEINLINE void store_relaxed(unsigned char* ptr, unsigned char val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  cvt.u8.u16 datum, %1;"
                  "  st.relaxed.gpu.u8 [%0], datum;"
                  "}" : : "l"(ptr),
                  "h"((unsigned short) val) : "memory");),
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  cvt.u8.u16 datum, %1;"
                  "  st.cg.u8 [%0], datum;"
                  "}" : : "l"(ptr),
                  "h"((unsigned short) val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(uint4* ptr, uint4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "r"(val.x),
                  "r"(val.y),
                  "r"(val.z),
                  "r"(val.w) : "memory");),
    (__threadfence(); asm volatile(
       "st.cg.v4.u32 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "r"(val.x), "r"(val.y), "r"(val.z), "r"(val.w) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(ulonglong2* ptr, ulonglong2 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");),
    (__threadfence(); asm volatile("st.cg.v2.u64 [%0], {%1, %2};" : : "l"(ptr), "l"(val.x), "l"(val.y) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(ushort4* ptr, ushort4 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr),
                  "h"(val.x),
                  "h"(val.y),
                  "h"(val.z),
                  "h"(val.w) : "memory");),
    (__threadfence(); asm volatile(
       "st.cg.v4.u16 [%0], {%1, %2, %3, %4};" : : "l"(ptr), "h"(val.x), "h"(val.y), "h"(val.z), "h"(val.w) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(uint2* ptr, uint2 val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("st.release.gpu.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");),
    (__threadfence(); asm volatile("st.cg.v2.u32 [%0], {%1, %2};" : : "l"(ptr), "r"(val.x), "r"(val.y) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned long long* ptr, unsigned long long val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");),
               (__threadfence(); asm volatile("st.cg.u64 [%0], %1;" : : "l"(ptr), "l"(val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned int* ptr, unsigned int val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");),
               (__threadfence(); asm volatile("st.cg.u32 [%0], %1;" : : "l"(ptr), "r"(val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned short* ptr, unsigned short val)
{
  NV_IF_TARGET(NV_PROVIDES_SM_70,
               (asm volatile("st.release.gpu.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");),
               (__threadfence(); asm volatile("st.cg.u16 [%0], %1;" : : "l"(ptr), "h"(val) : "memory");));
}

_CCCL_DEVICE _CCCL_FORCEINLINE void store_release(unsigned char* ptr, unsigned char val)
{
  NV_IF_TARGET(
    NV_PROVIDES_SM_70,
    (asm volatile("{"
                  "  .reg .u8 datum;"
                  "  cvt.u8.u16 datum, %1;"
                  "  st.release.gpu.u8 [%0], datum;"
                  "}" : : "l"(ptr),
                  "h"((unsigned short) val) : "memory");),
    (__threadfence(); asm volatile(
       "{"
       "  .reg .u8 datum;"
       "  cvt.u8.u16 datum, %1;"
       "  st.cg.u8 [%0], datum;"
       "}" : : "l"(ptr),
       "h"((unsigned short) val) : "memory");));
}
} // namespace detail

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
