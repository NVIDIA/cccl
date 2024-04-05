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

#include <cub/config.cuh>
#include <cub/util_type.cuh>

#include <iostream>

/******************************************************************************
 * Console printing utilities
 ******************************************************************************/

/**
 * Helper for casting character types to integers for cout printing
 */
template <typename T>
T CoutCast(T val) { return val; }

inline int CoutCast(char val) { return val; }

inline int CoutCast(unsigned char val) { return val; }

inline int CoutCast(signed char val) { return val; }

/******************************************************************************
 * Comparison and ostream operators for CUDA vector types
 ******************************************************************************/

/**
 * Vector1 overloads
 */
#define CUB_VEC_OVERLOAD_1(T, BaseT)                        \
    /* Ostream output */                                    \
    inline std::ostream& operator<<(                        \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '(' << CoutCast(val.x) << ')';                \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    inline __host__ __device__ bool operator!=(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x);                                \
    }                                                       \
    /* Equality */                                          \
    inline __host__ __device__ bool operator==(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x);                                \
    }                                                       \
    /* Max */                                               \
    inline __host__ __device__ bool operator>(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x > b.x);                                 \
    }                                                       \
    /* Min */                                               \
    inline __host__ __device__ bool operator<(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x < b.x);                                 \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                       \
    inline __host__ __device__ T operator+(                 \
        T a,                                                \
        T b)                                                \
    {                                                       \
        T retval = make_##T(a.x + b.x);                     \
        return retval;                                      \
    }



/**
 * Vector2 overloads
 */
#define CUB_VEC_OVERLOAD_2(T, BaseT)                        \
    /* Ostream output */                                    \
    inline std::ostream& operator<<(                        \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    inline __host__ __device__ bool operator!=(  \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) ||                              \
            (a.y != b.y);                                   \
    }                                                       \
    /* Equality */                                          \
    inline __host__ __device__ bool operator==(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x) &&                              \
            (a.y == b.y);                                   \
    }                                                       \
    /* Max */                                               \
    inline __host__ __device__ bool operator>(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x > b.x) return true; else if (b.x > a.x) return false;   \
        return a.y > b.y;                                               \
    }                                                       \
    /* Min */                                               \
    inline __host__ __device__ bool operator<(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x < b.x) return true; else if (b.x < a.x) return false;   \
        return a.y < b.y;                                               \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                         \
    inline __host__ __device__ T operator+(                 \
               T a,                                         \
               T b)                                         \
    {                                                       \
        T retval = make_##T(                                \
            a.x + b.x,                                      \
            a.y + b.y);                                     \
        return retval;                                      \
    }                                                       \



/**
 * Vector3 overloads
 */
#define CUB_VEC_OVERLOAD_3(T, BaseT)                        \
    /* Ostream output */                                    \
    inline std::ostream& operator<<(                        \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ','                       \
            << CoutCast(val.z) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    inline __host__ __device__ bool operator!=(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) ||                              \
            (a.y != b.y) ||                                 \
            (a.z != b.z);                                   \
    }                                                       \
    /* Equality */                                          \
    inline __host__ __device__ bool operator==(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x) &&                              \
            (a.y == b.y) &&                                 \
            (a.z == b.z);                                   \
    }                                                       \
    /* Max */                                               \
    inline __host__ __device__ bool operator>(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x > b.x) return true; else if (b.x > a.x) return false;   \
        if (a.y > b.y) return true; else if (b.y > a.y) return false;   \
        return a.z > b.z;                                               \
    }                                                       \
    /* Min */                                               \
    inline __host__ __device__ bool operator<(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x < b.x) return true; else if (b.x < a.x) return false;   \
        if (a.y < b.y) return true; else if (b.y < a.y) return false;   \
        return a.z < b.z;                                               \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                         \
    inline __host__ __device__ T operator+(                 \
        T a,                                                \
        T b)                                                \
    {                                                       \
        T retval = make_##T(                                \
            a.x + b.x,                                      \
            a.y + b.y,                                      \
            a.z + b.z);                                     \
        return retval;                                      \
    }


/**
 * Vector4 overloads
 */
#define CUB_VEC_OVERLOAD_4(T, BaseT)                        \
    /* Ostream output */                                    \
    inline std::ostream& operator<<(                        \
        std::ostream& os,                                   \
        const T& val)                                       \
    {                                                       \
        os << '('                                           \
            << CoutCast(val.x) << ','                       \
            << CoutCast(val.y) << ','                       \
            << CoutCast(val.z) << ','                       \
            << CoutCast(val.w) << ')';                      \
        return os;                                          \
    }                                                       \
    /* Inequality */                                        \
    inline __host__ __device__ bool operator!=(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x != b.x) ||                              \
            (a.y != b.y) ||                                 \
            (a.z != b.z) ||                                 \
            (a.w != b.w);                                   \
    }                                                       \
    /* Equality */                                          \
    inline __host__ __device__ bool operator==(             \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        return (a.x == b.x) &&                              \
            (a.y == b.y) &&                                 \
            (a.z == b.z) &&                                 \
            (a.w == b.w);                                   \
    }                                                       \
    /* Max */                                               \
    inline __host__ __device__ bool operator>(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x > b.x) return true; else if (b.x > a.x) return false;   \
        if (a.y > b.y) return true; else if (b.y > a.y) return false;   \
        if (a.z > b.z) return true; else if (b.z > a.z) return false;   \
        return a.w > b.w;                                               \
    }                                                       \
    /* Min */                                               \
    inline __host__ __device__ bool operator<(              \
        const T &a,                                         \
        const T &b)                                         \
    {                                                       \
        if (a.x < b.x) return true; else if (b.x < a.x) return false;   \
        if (a.y < b.y) return true; else if (b.y < a.y) return false;   \
        if (a.z < b.z) return true; else if (b.z < a.z) return false;   \
        return a.w < b.w;                                               \
    }                                                       \
    /* Summation (non-reference addends for VS2003 -O3 warpscan workaround */                                         \
    inline __host__ __device__ T operator+(                 \
        T a,                                                \
        T b)                                                \
    {                                                       \
        T retval = make_##T(                                        \
            a.x + b.x,                                      \
            a.y + b.y,                                      \
            a.z + b.z,                                      \
            a.w + b.w);                                     \
        return retval;                                      \
    }

/**
 * All vector overloads
 */
#define CUB_VEC_OVERLOAD(COMPONENT_T, BaseT)                    \
    CUB_VEC_OVERLOAD_1(COMPONENT_T##1, BaseT)                   \
    CUB_VEC_OVERLOAD_2(COMPONENT_T##2, BaseT)                   \
    CUB_VEC_OVERLOAD_3(COMPONENT_T##3, BaseT)                   \
    CUB_VEC_OVERLOAD_4(COMPONENT_T##4, BaseT)

/**
 * Define for types
 */
CUB_VEC_OVERLOAD(char, char)
CUB_VEC_OVERLOAD(short, short)
CUB_VEC_OVERLOAD(int, int)
CUB_VEC_OVERLOAD(long, long)
CUB_VEC_OVERLOAD(longlong, long long)
CUB_VEC_OVERLOAD(uchar, unsigned char)
CUB_VEC_OVERLOAD(ushort, unsigned short)
CUB_VEC_OVERLOAD(uint, unsigned int)
CUB_VEC_OVERLOAD(ulong, unsigned long)
CUB_VEC_OVERLOAD(ulonglong, unsigned long long)
CUB_VEC_OVERLOAD(float, float)
CUB_VEC_OVERLOAD(double, double)

/*
 * The following section defines macros to overload cub::NumericTraits<T>::{Max,Lowest}() for vector
 * types.
 */

/**
 * Vector1 overloads
 */
#define CUB_VEC_1_TRAITS_OVERLOAD(T, BaseT)                                                        \
  CUB_NAMESPACE_BEGIN                                                                              \
  template <>                                                                                      \
  struct NumericTraits<T>                                                                          \
  {                                                                                                \
    static constexpr Category CATEGORY = NOT_A_NUMBER;                                             \
    enum                                                                                           \
    {                                                                                              \
      PRIMITIVE = false,                                                                           \
      NULL_TYPE = false,                                                                           \
    };                                                                                             \
    static __host__ __device__ T Max()                                                             \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Max()};                                                    \
      return retval;                                                                               \
    }                                                                                              \
    static __host__ __device__ T Lowest()                                                          \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Lowest()};                                                 \
      return retval;                                                                               \
    }                                                                                              \
  };                                                                                               \
  CUB_NAMESPACE_END

/**
 * Vector2 overloads
 */
#define CUB_VEC_2_TRAITS_OVERLOAD(T, BaseT)                                                        \
  CUB_NAMESPACE_BEGIN                                                                              \
  template <>                                                                                      \
  struct NumericTraits<T>                                                                          \
  {                                                                                                \
    static constexpr Category CATEGORY = NOT_A_NUMBER;                                             \
    enum                                                                                           \
    {                                                                                              \
      PRIMITIVE = false,                                                                           \
      NULL_TYPE = false,                                                                           \
    };                                                                                             \
    static __host__ __device__ T Max()                                                             \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Max(), NumericTraits<BaseT>::Max()};                       \
      return retval;                                                                               \
    }                                                                                              \
    static __host__ __device__ T Lowest()                                                          \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Lowest(), NumericTraits<BaseT>::Lowest()};                 \
      return retval;                                                                               \
    }                                                                                              \
  };                                                                                               \
  CUB_NAMESPACE_END

/**
 * Vector3 overloads
 */
#define CUB_VEC_3_TRAITS_OVERLOAD(T, BaseT)                                                        \
  CUB_NAMESPACE_BEGIN                                                                              \
  template <>                                                                                      \
  struct NumericTraits<T>                                                                          \
  {                                                                                                \
    static constexpr Category CATEGORY = NOT_A_NUMBER;                                             \
    enum                                                                                           \
    {                                                                                              \
      PRIMITIVE = false,                                                                           \
      NULL_TYPE = false,                                                                           \
    };                                                                                             \
    static __host__ __device__ T Max()                                                             \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Max(),                                                     \
                  NumericTraits<BaseT>::Max(),                                                     \
                  NumericTraits<BaseT>::Max()};                                                    \
      return retval;                                                                               \
    }                                                                                              \
    static __host__ __device__ T Lowest()                                                          \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Lowest(),                                                  \
                  NumericTraits<BaseT>::Lowest(),                                                  \
                  NumericTraits<BaseT>::Lowest()};                                                 \
      return retval;                                                                               \
    }                                                                                              \
  };                                                                                               \
  CUB_NAMESPACE_END

/**
 * Vector4 overloads
 */
#define CUB_VEC_4_TRAITS_OVERLOAD(T, BaseT)                                                        \
  CUB_NAMESPACE_BEGIN                                                                              \
  template <>                                                                                      \
  struct NumericTraits<T>                                                                          \
  {                                                                                                \
    static constexpr Category CATEGORY = NOT_A_NUMBER;                                             \
    enum                                                                                           \
    {                                                                                              \
      PRIMITIVE = false,                                                                           \
      NULL_TYPE = false,                                                                           \
    };                                                                                             \
    static __host__ __device__ T Max()                                                             \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Max(),                                                     \
                  NumericTraits<BaseT>::Max(),                                                     \
                  NumericTraits<BaseT>::Max(),                                                     \
                  NumericTraits<BaseT>::Max()};                                                    \
      return retval;                                                                               \
    }                                                                                              \
    static __host__ __device__ T Lowest()                                                          \
    {                                                                                              \
      T retval = {NumericTraits<BaseT>::Lowest(),                                                  \
                  NumericTraits<BaseT>::Lowest(),                                                  \
                  NumericTraits<BaseT>::Lowest(),                                                  \
                  NumericTraits<BaseT>::Lowest()};                                                 \
      return retval;                                                                               \
    }                                                                                              \
  };                                                                                               \
  CUB_NAMESPACE_END

/**
 * All vector overloads
 */
#define CUB_VEC_TRAITS_OVERLOAD(COMPONENT_T, BaseT)                                                \
  CUB_VEC_1_TRAITS_OVERLOAD(COMPONENT_T##1, BaseT)                                                 \
  CUB_VEC_2_TRAITS_OVERLOAD(COMPONENT_T##2, BaseT)                                                 \
  CUB_VEC_3_TRAITS_OVERLOAD(COMPONENT_T##3, BaseT)                                                 \
  CUB_VEC_4_TRAITS_OVERLOAD(COMPONENT_T##4, BaseT)

/**
 * Define for types
 */
CUB_VEC_TRAITS_OVERLOAD(char, signed char)
CUB_VEC_TRAITS_OVERLOAD(short, short)
CUB_VEC_TRAITS_OVERLOAD(int, int)
CUB_VEC_TRAITS_OVERLOAD(long, long)
CUB_VEC_TRAITS_OVERLOAD(longlong, long long)
CUB_VEC_TRAITS_OVERLOAD(uchar, unsigned char)
CUB_VEC_TRAITS_OVERLOAD(ushort, unsigned short)
CUB_VEC_TRAITS_OVERLOAD(uint, unsigned int)
CUB_VEC_TRAITS_OVERLOAD(ulong, unsigned long)
CUB_VEC_TRAITS_OVERLOAD(ulonglong, unsigned long long)
CUB_VEC_TRAITS_OVERLOAD(float, float)
CUB_VEC_TRAITS_OVERLOAD(double, double)
