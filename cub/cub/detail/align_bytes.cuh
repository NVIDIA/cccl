// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

CUB_NAMESPACE_BEGIN
#ifndef _CCCL_DOXYGEN_INVOKED // Do not document
/******************************************************************************
 * Size and alignment
 ******************************************************************************/

namespace detail
{
/// Structure alignment
template <typename T>
struct AlignBytes
{
  /// The "true CUDA" alignment of T in bytes
  static constexpr unsigned ALIGN_BYTES = alignof(T);

  /// The "truly aligned" type
  using Type = T;
};

// Specializations where host C++ compilers (e.g., 32-bit Windows) may disagree
// with device C++ compilers (EDG) on types passed as template parameters through
// kernel functions

#  define __CUB_ALIGN_BYTES(t, b)                                                                  \
    template <>                                                                                    \
    struct AlignBytes<t>                                                                           \
    {                                                                                              \
      static constexpr unsigned ALIGN_BYTES = b;                                                   \
                                                                                                   \
      typedef __align__(b) t Type;                                                                 \
      /* TODO(bgruber): rewriting the above to using Type __align__(b) = t; does not compile :S */ \
    };

__CUB_ALIGN_BYTES(short4, 8)
__CUB_ALIGN_BYTES(ushort4, 8)
__CUB_ALIGN_BYTES(int2, 8)
__CUB_ALIGN_BYTES(uint2, 8)
__CUB_ALIGN_BYTES(long long, 8)
__CUB_ALIGN_BYTES(unsigned long long, 8)
__CUB_ALIGN_BYTES(float2, 8)
__CUB_ALIGN_BYTES(double, 8)
#  ifdef _WIN32
__CUB_ALIGN_BYTES(long2, 8)
__CUB_ALIGN_BYTES(ulong2, 8)
#  else
__CUB_ALIGN_BYTES(long2, 16)
__CUB_ALIGN_BYTES(ulong2, 16)
#  endif
__CUB_ALIGN_BYTES(int4, 16)
__CUB_ALIGN_BYTES(uint4, 16)
__CUB_ALIGN_BYTES(float4, 16)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
__CUB_ALIGN_BYTES(long4, 16)
__CUB_ALIGN_BYTES(ulong4, 16)
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
__CUB_ALIGN_BYTES(long4_16a, 16)
__CUB_ALIGN_BYTES(long4_32a, 32)
__CUB_ALIGN_BYTES(ulong4_16a, 16)
__CUB_ALIGN_BYTES(ulong4_32a, 32)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)
__CUB_ALIGN_BYTES(longlong2, 16)
__CUB_ALIGN_BYTES(ulonglong2, 16)
__CUB_ALIGN_BYTES(double2, 16)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
__CUB_ALIGN_BYTES(longlong4, 16)
__CUB_ALIGN_BYTES(ulonglong4, 16)
__CUB_ALIGN_BYTES(double4, 16)
_CCCL_SUPPRESS_DEPRECATED_POP
#  if _CCCL_CTK_AT_LEAST(13, 0)
__CUB_ALIGN_BYTES(longlong4_16a, 16)
__CUB_ALIGN_BYTES(longlong4_32a, 32)
__CUB_ALIGN_BYTES(ulonglong4_16a, 16)
__CUB_ALIGN_BYTES(ulonglong4_32a, 32)
__CUB_ALIGN_BYTES(double4_16a, 16)
__CUB_ALIGN_BYTES(double4_32a, 32)
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

// clang-format off
template <typename T> struct AlignBytes<volatile T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const T> : AlignBytes<T> {};
template <typename T> struct AlignBytes<const volatile T> : AlignBytes<T> {};
// clang-format on
} // namespace detail

template <typename T>
using AlignBytes CCCL_DEPRECATED_BECAUSE("Use alignof(T) directly") = detail::AlignBytes<T>;
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
