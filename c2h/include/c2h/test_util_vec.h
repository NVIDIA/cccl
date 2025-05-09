/******************************************************************************
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

#pragma once

#include <thrust/detail/config/device_system.h>

#include <cuda/std/limits>

#include <iostream>

/******************************************************************************
 * Console printing utilities
 ******************************************************************************/

/**
 * Helper for casting character types to integers for cout printing
 */
template <typename T>
T CoutCast(T val)
{
  return val;
}

inline int CoutCast(char val)
{
  return val;
}

inline int CoutCast(unsigned char val)
{
  return val;
}

inline int CoutCast(signed char val)
{
  return val;
}

/******************************************************************************
 * Comparison and ostream operators for CUDA vector types
 ******************************************************************************/

#if THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA

/**
 * Vector1 overloads
 */
#  define C2H_VEC_OVERLOAD_1(T, BaseT)                                        \
    /* Ostream output */                                                      \
    inline std::ostream& operator<<(std::ostream& os, const T& val)           \
    {                                                                         \
      os << '(' << CoutCast(val.x) << ')';                                    \
      return os;                                                              \
    }                                                                         \
    /* Inequality */                                                          \
    inline __host__ __device__ bool operator!=(const T& a, const T& b)        \
    {                                                                         \
      return (a.x != b.x);                                                    \
    }                                                                         \
    /* Equality */                                                            \
    inline __host__ __device__ bool operator==(const T& a, const T& b)        \
    {                                                                         \
      return (a.x == b.x);                                                    \
    }                                                                         \
    /* Max */                                                                 \
    inline __host__ __device__ bool operator>(const T& a, const T& b)         \
    {                                                                         \
      return (a.x > b.x);                                                     \
    }                                                                         \
    /* Min */                                                                 \
    inline __host__ __device__ bool operator<(const T& a, const T& b)         \
    {                                                                         \
      return (a.x < b.x);                                                     \
    }                                                                         \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */ \
    inline __host__ __device__ T operator+(T a, T b)                          \
    {                                                                         \
      T retval = make_##T(a.x + b.x);                                         \
      return retval;                                                          \
    }

/**
 * Vector2 overloads
 */
#  define C2H_VEC_OVERLOAD_2(T, BaseT)                                        \
    /* Ostream output */                                                      \
    inline std::ostream& operator<<(std::ostream& os, const T& val)           \
    {                                                                         \
      os << '(' << CoutCast(val.x) << ',' << CoutCast(val.y) << ')';          \
      return os;                                                              \
    }                                                                         \
    /* Inequality */                                                          \
    inline __host__ __device__ bool operator!=(const T& a, const T& b)        \
    {                                                                         \
      return (a.x != b.x) || (a.y != b.y);                                    \
    }                                                                         \
    /* Equality */                                                            \
    inline __host__ __device__ bool operator==(const T& a, const T& b)        \
    {                                                                         \
      return (a.x == b.x) && (a.y == b.y);                                    \
    }                                                                         \
    /* Max */                                                                 \
    inline __host__ __device__ bool operator>(const T& a, const T& b)         \
    {                                                                         \
      if (a.x > b.x)                                                          \
        return true;                                                          \
      else if (b.x > a.x)                                                     \
        return false;                                                         \
      return a.y > b.y;                                                       \
    }                                                                         \
    /* Min */                                                                 \
    inline __host__ __device__ bool operator<(const T& a, const T& b)         \
    {                                                                         \
      if (a.x < b.x)                                                          \
        return true;                                                          \
      else if (b.x < a.x)                                                     \
        return false;                                                         \
      return a.y < b.y;                                                       \
    }                                                                         \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */ \
    inline __host__ __device__ T operator+(T a, T b)                          \
    {                                                                         \
      T retval = make_##T(a.x + b.x, a.y + b.y);                              \
      return retval;                                                          \
    }

/**
 * Vector3 overloads
 */
#  define C2H_VEC_OVERLOAD_3(T, BaseT)                                                         \
    /* Ostream output */                                                                       \
    inline std::ostream& operator<<(std::ostream& os, const T& val)                            \
    {                                                                                          \
      os << '(' << CoutCast(val.x) << ',' << CoutCast(val.y) << ',' << CoutCast(val.z) << ')'; \
      return os;                                                                               \
    }                                                                                          \
    /* Inequality */                                                                           \
    inline __host__ __device__ bool operator!=(const T& a, const T& b)                         \
    {                                                                                          \
      return (a.x != b.x) || (a.y != b.y) || (a.z != b.z);                                     \
    }                                                                                          \
    /* Equality */                                                                             \
    inline __host__ __device__ bool operator==(const T& a, const T& b)                         \
    {                                                                                          \
      return (a.x == b.x) && (a.y == b.y) && (a.z == b.z);                                     \
    }                                                                                          \
    /* Max */                                                                                  \
    inline __host__ __device__ bool operator>(const T& a, const T& b)                          \
    {                                                                                          \
      if (a.x > b.x)                                                                           \
        return true;                                                                           \
      else if (b.x > a.x)                                                                      \
        return false;                                                                          \
      if (a.y > b.y)                                                                           \
        return true;                                                                           \
      else if (b.y > a.y)                                                                      \
        return false;                                                                          \
      return a.z > b.z;                                                                        \
    }                                                                                          \
    /* Min */                                                                                  \
    inline __host__ __device__ bool operator<(const T& a, const T& b)                          \
    {                                                                                          \
      if (a.x < b.x)                                                                           \
        return true;                                                                           \
      else if (b.x < a.x)                                                                      \
        return false;                                                                          \
      if (a.y < b.y)                                                                           \
        return true;                                                                           \
      else if (b.y < a.y)                                                                      \
        return false;                                                                          \
      return a.z < b.z;                                                                        \
    }                                                                                          \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                  \
    inline __host__ __device__ T operator+(T a, T b)                                           \
    {                                                                                          \
      T retval = make_##T(a.x + b.x, a.y + b.y, a.z + b.z);                                    \
      return retval;                                                                           \
    }

/**
 * Vector4 overloads
 */
#  define C2H_VEC_OVERLOAD_4(T, BaseT)                                                                           \
    /* Ostream output */                                                                                         \
    inline std::ostream& operator<<(std::ostream& os, const T& val)                                              \
    {                                                                                                            \
      os << '(' << CoutCast(val.x) << ',' << CoutCast(val.y) << ',' << CoutCast(val.z) << ',' << CoutCast(val.w) \
         << ')';                                                                                                 \
      return os;                                                                                                 \
    }                                                                                                            \
    /* Inequality */                                                                                             \
    inline __host__ __device__ bool operator!=(const T& a, const T& b)                                           \
    {                                                                                                            \
      return (a.x != b.x) || (a.y != b.y) || (a.z != b.z) || (a.w != b.w);                                       \
    }                                                                                                            \
    /* Equality */                                                                                               \
    inline __host__ __device__ bool operator==(const T& a, const T& b)                                           \
    {                                                                                                            \
      return (a.x == b.x) && (a.y == b.y) && (a.z == b.z) && (a.w == b.w);                                       \
    }                                                                                                            \
    /* Max */                                                                                                    \
    inline __host__ __device__ bool operator>(const T& a, const T& b)                                            \
    {                                                                                                            \
      if (a.x > b.x)                                                                                             \
        return true;                                                                                             \
      else if (b.x > a.x)                                                                                        \
        return false;                                                                                            \
      if (a.y > b.y)                                                                                             \
        return true;                                                                                             \
      else if (b.y > a.y)                                                                                        \
        return false;                                                                                            \
      if (a.z > b.z)                                                                                             \
        return true;                                                                                             \
      else if (b.z > a.z)                                                                                        \
        return false;                                                                                            \
      return a.w > b.w;                                                                                          \
    }                                                                                                            \
    /* Min */                                                                                                    \
    inline __host__ __device__ bool operator<(const T& a, const T& b)                                            \
    {                                                                                                            \
      if (a.x < b.x)                                                                                             \
        return true;                                                                                             \
      else if (b.x < a.x)                                                                                        \
        return false;                                                                                            \
      if (a.y < b.y)                                                                                             \
        return true;                                                                                             \
      else if (b.y < a.y)                                                                                        \
        return false;                                                                                            \
      if (a.z < b.z)                                                                                             \
        return true;                                                                                             \
      else if (b.z < a.z)                                                                                        \
        return false;                                                                                            \
      return a.w < b.w;                                                                                          \
    }                                                                                                            \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                    \
    inline __host__ __device__ T operator+(T a, T b)                                                             \
    {                                                                                                            \
      T retval = make_##T(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);                                           \
      return retval;                                                                                             \
    }

/**
 * All vector overloads
 */
#  define C2H_VEC_OVERLOAD(COMPONENT_T, BaseT) \
    C2H_VEC_OVERLOAD_1(COMPONENT_T##1, BaseT)  \
    C2H_VEC_OVERLOAD_2(COMPONENT_T##2, BaseT)  \
    C2H_VEC_OVERLOAD_3(COMPONENT_T##3, BaseT)  \
    C2H_VEC_OVERLOAD_4(COMPONENT_T##4, BaseT)

/**
 * Define for types
 */
C2H_VEC_OVERLOAD(char, char)
C2H_VEC_OVERLOAD(short, short)
C2H_VEC_OVERLOAD(int, int)
C2H_VEC_OVERLOAD(long, long)
C2H_VEC_OVERLOAD(longlong, long long)
C2H_VEC_OVERLOAD(uchar, unsigned char)
C2H_VEC_OVERLOAD(ushort, unsigned short)
C2H_VEC_OVERLOAD(uint, unsigned int)
C2H_VEC_OVERLOAD(ulong, unsigned long)
C2H_VEC_OVERLOAD(ulonglong, unsigned long long)
C2H_VEC_OVERLOAD(float, float)
C2H_VEC_OVERLOAD(double, double)

// Specialize cuda::std::numeric_limits for vector types.

#  define REPEAT_TO_LIST_1(a)  a
#  define REPEAT_TO_LIST_2(a)  a, a
#  define REPEAT_TO_LIST_3(a)  a, a, a
#  define REPEAT_TO_LIST_4(a)  a, a, a, a
#  define REPEAT_TO_LIST(N, a) _CCCL_PP_CAT(REPEAT_TO_LIST_, N)(a)

#  define C2H_VEC_TRAITS_OVERLOAD_IMPL(T, BaseT, N)                               \
    _LIBCUDACXX_BEGIN_NAMESPACE_STD                                               \
    template <>                                                                   \
    class numeric_limits<T>                                                       \
    {                                                                             \
    public:                                                                       \
      static constexpr bool is_specialized = true;                                \
      static __host__ __device__ T max()                                          \
      {                                                                           \
        return {REPEAT_TO_LIST(N, ::cuda::std::numeric_limits<BaseT>::max())};    \
      }                                                                           \
      static __host__ __device__ T min()                                          \
      {                                                                           \
        return {REPEAT_TO_LIST(N, ::cuda::std::numeric_limits<BaseT>::min())};    \
      }                                                                           \
      static __host__ __device__ T lowest()                                       \
      {                                                                           \
        return {REPEAT_TO_LIST(N, ::cuda::std::numeric_limits<BaseT>::lowest())}; \
      }                                                                           \
    };                                                                            \
    _LIBCUDACXX_END_NAMESPACE_STD

#  define C2H_VEC_TRAITS_OVERLOAD(COMPONENT_T, BaseT)      \
    C2H_VEC_TRAITS_OVERLOAD_IMPL(COMPONENT_T##1, BaseT, 1) \
    C2H_VEC_TRAITS_OVERLOAD_IMPL(COMPONENT_T##2, BaseT, 2) \
    C2H_VEC_TRAITS_OVERLOAD_IMPL(COMPONENT_T##3, BaseT, 3) \
    C2H_VEC_TRAITS_OVERLOAD_IMPL(COMPONENT_T##4, BaseT, 4)

C2H_VEC_TRAITS_OVERLOAD(char, signed char)
C2H_VEC_TRAITS_OVERLOAD(short, short)
C2H_VEC_TRAITS_OVERLOAD(int, int)
C2H_VEC_TRAITS_OVERLOAD(long, long)
C2H_VEC_TRAITS_OVERLOAD(longlong, long long)
C2H_VEC_TRAITS_OVERLOAD(uchar, unsigned char)
C2H_VEC_TRAITS_OVERLOAD(ushort, unsigned short)
C2H_VEC_TRAITS_OVERLOAD(uint, unsigned int)
C2H_VEC_TRAITS_OVERLOAD(ulong, unsigned long)
C2H_VEC_TRAITS_OVERLOAD(ulonglong, unsigned long long)
C2H_VEC_TRAITS_OVERLOAD(float, float)
C2H_VEC_TRAITS_OVERLOAD(double, double)

#  undef C2H_VEC_TRAITS_OVERLOAD
#  undef C2H_VEC_TRAITS_OVERLOAD_IMPL
#  undef REPEAT_TO_LIST_1
#  undef REPEAT_TO_LIST_2
#  undef REPEAT_TO_LIST_3
#  undef REPEAT_TO_LIST_4
#  undef REPEAT_TO_LIST

//----------------------------------------------------------------------------------------------------------------------
// vector2 type traits

template <typename T>
inline constexpr bool is_vector2_type_impl_v = false;

template <>
inline constexpr bool is_vector2_type_impl_v<char2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<short2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<int2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<long2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<longlong2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<uchar2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<ushort2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<uint2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<ulong2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<ulonglong2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<float2> = true;

template <>
inline constexpr bool is_vector2_type_impl_v<double2> = true;

#  if TEST_HALF_T()

template <>
inline constexpr bool is_vector2_type_impl_v<__half2> = true;

#  endif // TEST_HALF_T()

#  if TEST_BF_T()

template <>
inline constexpr bool is_vector2_type_impl_v<__nv_bfloat162> = true;

#  endif // TEST_BF_T()

template <typename T>
inline constexpr bool is_vector2_type_v = is_vector2_type_impl_v<cuda::std::remove_cv_t<T>>;

//----------------------------------------------------------------------------------------------------------------------
// vector2 floating point type traits

template <typename T>
inline constexpr bool is_vector2_fp_type_impl_v = false;

template <>
inline constexpr bool is_vector2_fp_type_impl_v<float2> = true;

template <>
inline constexpr bool is_vector2_fp_type_impl_v<double2> = true;

#  if TEST_HALF_T()

template <>
inline constexpr bool is_vector2_fp_type_impl_v<__half2> = true;

#  endif // TEST_HALF_T()

#  if TEST_BF_T()

template <>
inline constexpr bool is_vector2_fp_type_impl_v<__nv_bfloat162> = true;

#  endif // TEST_BF_T()

template <typename T>
inline constexpr bool is_vector2_fp_type_v = is_vector2_fp_type_impl_v<cuda::std::remove_cv_t<T>>;

#endif // THRUST_DEVICE_SYSTEM == THRUST_DEVICE_SYSTEM_CUDA
