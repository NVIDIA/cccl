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
 * Vector type inference utilities.
 ******************************************************************************/

/**
 * \brief Exposes a member alias \p Type that names the corresponding CUDA vector type if one exists.  Otherwise \p
 * Type refers to the CubVector structure itself, which will wrap the corresponding \p x, \p y, etc. vector fields.
 */
template <typename T, int VecElements>
struct CubVector
{
  static_assert(!sizeof(T), "CubVector can only have 1-4 elements");
};

/// The maximum number of elements in CUDA vector types
inline constexpr int MAX_VEC_ELEMENTS = 4;

/**
 * Generic vector-1 type
 */
template <typename T>
struct CubVector<T, 1>
{
  T x;

  using BaseType = T;
  using Type     = CubVector;
};

/**
 * Generic vector-2 type
 */
template <typename T>
struct CubVector<T, 2>
{
  T x;
  T y;

  using BaseType = T;
  using Type     = CubVector;
};

/**
 * Generic vector-3 type
 */
template <typename T>
struct CubVector<T, 3>
{
  T x;
  T y;
  T z;

  using BaseType = T;
  using Type     = CubVector;
};

/**
 * Generic vector-4 type
 */
template <typename T>
struct CubVector<T, 4>
{
  T x;
  T y;
  T z;
  T w;

  using BaseType = T;
  using Type     = CubVector;
};

// TODO(bgruber): should CubVectorType support (and how?) the new type4_16a and type4_32a vector types from CTK 13?
/**
 * Macro for expanding partially-specialized built-in vector types
 */
#  define CUB_DEFINE_VECTOR_TYPE(base_type, short_type)                                     \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 1> : short_type##1                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##1;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };                                                                                      \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 2> : short_type##2                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##2;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        retval.y = y + other.y;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        retval.y = y - other.y;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };                                                                                      \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 3> : short_type##3                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##3;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        retval.y = y + other.y;                                                             \
        retval.z = z + other.z;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        retval.y = y - other.y;                                                             \
        retval.z = z - other.z;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };                                                                                      \
                                                                                            \
    template <>                                                                             \
    struct CubVector<base_type, 4> : short_type##4                                          \
    {                                                                                       \
      using BaseType = base_type;                                                           \
      using Type     = short_type##4;                                                       \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator+(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x + other.x;                                                             \
        retval.y = y + other.y;                                                             \
        retval.z = z + other.z;                                                             \
        retval.w = w + other.w;                                                             \
        return retval;                                                                      \
      }                                                                                     \
      _CCCL_HOST_DEVICE _CCCL_FORCEINLINE CubVector operator-(const CubVector& other) const \
      {                                                                                     \
        CubVector retval;                                                                   \
        retval.x = x - other.x;                                                             \
        retval.y = y - other.y;                                                             \
        retval.z = z - other.z;                                                             \
        retval.w = w - other.w;                                                             \
        return retval;                                                                      \
      }                                                                                     \
    };

// Expand CUDA vector types for built-in primitives
// clang-format off
CUB_DEFINE_VECTOR_TYPE(char,               char)
CUB_DEFINE_VECTOR_TYPE(signed char,        char)
CUB_DEFINE_VECTOR_TYPE(short,              short)
CUB_DEFINE_VECTOR_TYPE(int,                int)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
CUB_DEFINE_VECTOR_TYPE(long,               long)
CUB_DEFINE_VECTOR_TYPE(long long,          longlong)
_CCCL_SUPPRESS_DEPRECATED_POP
CUB_DEFINE_VECTOR_TYPE(unsigned char,      uchar)
CUB_DEFINE_VECTOR_TYPE(unsigned short,     ushort)
CUB_DEFINE_VECTOR_TYPE(unsigned int,       uint)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
CUB_DEFINE_VECTOR_TYPE(unsigned long,      ulong)
CUB_DEFINE_VECTOR_TYPE(unsigned long long, ulonglong)
_CCCL_SUPPRESS_DEPRECATED_POP
CUB_DEFINE_VECTOR_TYPE(float,              float)
_CCCL_SUPPRESS_DEPRECATED_PUSH
_CCCL_SUPPRESS_DEPRECATED_NVRTC_DIAG
CUB_DEFINE_VECTOR_TYPE(double,             double)
_CCCL_SUPPRESS_DEPRECATED_POP
CUB_DEFINE_VECTOR_TYPE(bool,               uchar)
// clang-format on

#  undef CUB_DEFINE_VECTOR_TYPE
#endif // _CCCL_DOXYGEN_INVOKED
CUB_NAMESPACE_END
