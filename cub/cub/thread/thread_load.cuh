// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * @file
 * Thread utilities for reading memory using PTX cache modifiers.
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

#include <cub/detail/constant.cuh>
#include <cub/detail/it_traits.cuh>
#include <cub/detail/numeric_traits.cuh>
#include <cub/detail/unit_word.cuh>

#include <thrust/type_traits/is_contiguous_iterator.h>

#include <cuda/std/__host_stdlib/ostream>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/integral_constant.h>
#include <cuda/std/__utility/integer_sequence.h>

#include <nv/target>

#if __cpp_lib_format >= 201907L
#  include <cuda/std/__concepts/same_as.h>
#  include <cuda/std/__fwd/format.h> // IWYU pragma: keep
#endif // __cpp_lib_format >= 201907L

CUB_NAMESPACE_BEGIN

//-----------------------------------------------------------------------------
// Tags and constants
//-----------------------------------------------------------------------------

//! @brief Enumeration of cache modifiers for memory load operations.
enum CacheLoadModifier // NOLINT(cppcoreguidelines-use-enum-class)
{
  LOAD_DEFAULT, ///< Default (no modifier)
  LOAD_CA, ///< Cache at all levels
  LOAD_CG, ///< Cache at global level
  LOAD_CS, ///< Cache streaming (likely to be accessed once)
  LOAD_CV, ///< Cache as volatile (including cached system lines)
  LOAD_LDG, ///< Cache as texture
  LOAD_VOLATILE, ///< Volatile (any memory space)
};

#if _CCCL_HOSTED()
namespace detail
{
[[nodiscard]] _CCCL_HOST_DEVICE_API constexpr const char* to_string(CacheLoadModifier modifier) noexcept
{
  switch (modifier)
  {
    case LOAD_DEFAULT:
      return "LOAD_DEFAULT";
    case LOAD_CA:
      return "LOAD_CA";
    case LOAD_CG:
      return "LOAD_CG";
    case LOAD_CS:
      return "LOAD_CS";
    case LOAD_CV:
      return "LOAD_CV";
    case LOAD_LDG:
      return "LOAD_LDG";
    case LOAD_VOLATILE:
      return "LOAD_VOLATILE";
  }
  return "<unknown CacheLoadModifier>";
}
} // namespace detail
#endif // _CCCL_HOSTED()

#if _CCCL_HOSTED() && !defined(_CCCL_DOXYGEN_INVOKED)
_CCCL_HOST_API inline ::std::ostream& operator<<(::std::ostream& os, CacheLoadModifier modifier)
{
  return os << _CUB::detail::to_string(modifier);
}
#endif // _CCCL_HOSTED() && !_CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END

#if __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)
template <::cuda::std::same_as<char> CharT>
struct std::formatter<_CUB::CacheLoadModifier, CharT> : formatter<const CharT*, CharT>
{
  template <class FmtCtx>
  _CCCL_HOST_API auto format(const _CUB::CacheLoadModifier& modifier, FmtCtx& ctx) const
  {
    return formatter<const CharT*, CharT>::format(_CUB::detail::to_string(modifier), ctx);
  }
};
#endif // __cpp_lib_format >= 201907L && !defined(_CCCL_DOXYGEN_INVOKED)

CUB_NAMESPACE_BEGIN

//! @name Thread I/O (cache modified)
//! @{

//! @brief Thread utility for reading memory using cub::CacheLoadModifier cache modifiers.
//!        Can be used to load any data type.
//!
//! With cub::LOAD_LDG, types that are aligned to 32 bytes are loaded with 32 byte accesses on sm_100 and newer.
//!
//! @par Example
//! @code
//! #include <cub/cub.cuh>   // or equivalently <cub/thread/thread_load.cuh>
//!
//! // 32-bit load using cache-global modifier:
//! int *d_in;
//! int val = cub::ThreadLoad<cub::LOAD_CA>(d_in + threadIdx.x);
//!
//! // 16-bit load using default modifier
//! short *d_in;
//! short val = cub::ThreadLoad<cub::LOAD_DEFAULT>(d_in + threadIdx.x);
//!
//! // 128-bit load using cache-volatile modifier
//! float4 *d_in;
//! float4 val = cub::ThreadLoad<cub::LOAD_CV>(d_in + threadIdx.x);
//!
//! // 96-bit load using cache-streaming modifier
//! struct TestFoo { bool a; short b; };
//! TestFoo *d_struct;
//! TestFoo val = cub::ThreadLoad<cub::LOAD_CS>(d_in + threadIdx.x);
//! \endcode
//!
//! @tparam MODIFIER
//!   <b>[inferred]</b> CacheLoadModifier enumeration
//!
//! @tparam RandomAccessIterator
//!   <b>[inferred]</b> The input's iterator type \iterator
template <CacheLoadModifier MODIFIER, typename RandomAccessIterator>
_CCCL_DEVICE_API _CCCL_FORCEINLINE cub::detail::it_value_t<RandomAccessIterator> ThreadLoad(RandomAccessIterator itr);

//@}

#ifndef _CCCL_DOXYGEN_INVOKED // Do not document

namespace detail
{
template <CacheLoadModifier MODIFIER, typename T, int... Is>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
UnrolledThreadLoadImpl(T const* src, T* dst, ::cuda::std::integer_sequence<int, Is...>)
{
  ((dst[Is] = _CUB::ThreadLoad<MODIFIER>(src + Is)), ...);
}

template <typename RandomAccessIterator, typename T, int... Is>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void
UnrolledCopyImpl(RandomAccessIterator src, T* dst, ::cuda::std::integer_sequence<int, Is...>)
{
  ((dst[Is] = src[Is]), ...);
}
} // namespace detail

template <int Count, CacheLoadModifier MODIFIER, typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void UnrolledThreadLoad(T const* src, T* dst)
{
  _CUB::detail::UnrolledThreadLoadImpl<MODIFIER>(src, dst, ::cuda::std::make_integer_sequence<int, Count>{});
}

template <int Count, typename RandomAccessIterator, typename T>
_CCCL_DEVICE_API _CCCL_FORCEINLINE void UnrolledCopy(RandomAccessIterator src, T* dst)
{
  _CUB::detail::UnrolledCopyImpl(src, dst, ::cuda::std::make_integer_sequence<int, Count>{});
}

//! Define a uint4 (16B) ThreadLoad specialization for the given Cache load modifier
#  define _CUB_LOAD_16(cub_modifier, ptx_modifier)                                                                   \
    template <>                                                                                                      \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE uint4 ThreadLoad<cub_modifier, uint4 const*>(uint4 const* ptr)                \
    {                                                                                                                \
      uint4 retval;                                                                                                  \
      asm volatile("ld." #ptx_modifier ".v4.u32 {%0, %1, %2, %3}, [%4];"                                             \
                   : "=r"(retval.x), "=r"(retval.y), "=r"(retval.z), "=r"(retval.w)                                  \
                   : "l"(ptr));                                                                                      \
      return retval;                                                                                                 \
    }                                                                                                                \
    template <>                                                                                                      \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE ulonglong2 ThreadLoad<cub_modifier, ulonglong2 const*>(ulonglong2 const* ptr) \
    {                                                                                                                \
      ulonglong2 retval;                                                                                             \
      asm volatile("ld." #ptx_modifier ".v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));       \
      return retval;                                                                                                 \
    }

//! Define a uint2 (8B) ThreadLoad specialization for the given Cache load modifier
#  define _CUB_LOAD_8(cub_modifier, ptx_modifier)                                                              \
    template <>                                                                                                \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE ushort4 ThreadLoad<cub_modifier, ushort4 const*>(ushort4 const* ptr)    \
    {                                                                                                          \
      ushort4 retval;                                                                                          \
      asm volatile("ld." #ptx_modifier ".v4.u16 {%0, %1, %2, %3}, [%4];"                                       \
                   : "=h"(retval.x), "=h"(retval.y), "=h"(retval.z), "=h"(retval.w)                            \
                   : "l"(ptr));                                                                                \
      return retval;                                                                                           \
    }                                                                                                          \
    template <>                                                                                                \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE uint2 ThreadLoad<cub_modifier, uint2 const*>(uint2 const* ptr)          \
    {                                                                                                          \
      uint2 retval;                                                                                            \
      asm volatile("ld." #ptx_modifier ".v2.u32 {%0, %1}, [%2];" : "=r"(retval.x), "=r"(retval.y) : "l"(ptr)); \
      return retval;                                                                                           \
    }                                                                                                          \
    template <>                                                                                                \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned long long ThreadLoad<cub_modifier, unsigned long long const*>( \
      unsigned long long const* ptr)                                                                           \
    {                                                                                                          \
      unsigned long long retval;                                                                               \
      asm volatile("ld." #ptx_modifier ".u64 %0, [%1];" : "=l"(retval) : "l"(ptr));                            \
      return retval;                                                                                           \
    }

//! Define a uint (4B) ThreadLoad specialization for the given Cache load modifier
#  define _CUB_LOAD_4(cub_modifier, ptx_modifier)                                                  \
    template <>                                                                                    \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned int ThreadLoad<cub_modifier, unsigned int const*>( \
      unsigned int const* ptr)                                                                     \
    {                                                                                              \
      unsigned int retval;                                                                         \
      asm volatile("ld." #ptx_modifier ".u32 %0, [%1];" : "=r"(retval) : "l"(ptr));                \
      return retval;                                                                               \
    }

//! Define a unsigned short (2B) ThreadLoad specialization for the given Cache load modifier
#  define _CUB_LOAD_2(cub_modifier, ptx_modifier)                                                      \
    template <>                                                                                        \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned short ThreadLoad<cub_modifier, unsigned short const*>( \
      unsigned short const* ptr)                                                                       \
    {                                                                                                  \
      unsigned short retval;                                                                           \
      asm volatile("ld." #ptx_modifier ".u16 %0, [%1];" : "=h"(retval) : "l"(ptr));                    \
      return retval;                                                                                   \
    }

//! Define an unsigned char (1B) ThreadLoad specialization for the given Cache load modifier
#  define _CUB_LOAD_1(cub_modifier, ptx_modifier)                                                    \
    template <>                                                                                      \
    _CCCL_DEVICE_API _CCCL_FORCEINLINE unsigned char ThreadLoad<cub_modifier, unsigned char const*>( \
      unsigned char const* ptr)                                                                      \
    {                                                                                                \
      unsigned short retval;                                                                         \
      asm volatile(                                                                                  \
        "{"                                                                                          \
        "   .reg .u8 datum;"                                                                         \
        "    ld." #ptx_modifier ".u8 datum, [%1];"                                                   \
        "    cvt.u16.u8 %0, datum;"                                                                  \
        "}"                                                                                          \
        : "=h"(retval)                                                                               \
        : "l"(ptr));                                                                                 \
      return (unsigned char) retval;                                                                 \
    }

//! Define powers-of-two ThreadLoad specializations for the given Cache load modifier
#  define _CUB_LOAD_ALL(cub_modifier, ptx_modifier) \
    _CUB_LOAD_16(cub_modifier, ptx_modifier)        \
    _CUB_LOAD_8(cub_modifier, ptx_modifier)         \
    _CUB_LOAD_4(cub_modifier, ptx_modifier)         \
    _CUB_LOAD_2(cub_modifier, ptx_modifier)         \
    _CUB_LOAD_1(cub_modifier, ptx_modifier)

//! Define powers-of-two ThreadLoad specializations for the various Cache load modifiers
_CUB_LOAD_ALL(LOAD_CA, ca)
_CUB_LOAD_ALL(LOAD_CG, cg)
_CUB_LOAD_ALL(LOAD_CS, cs)
_CUB_LOAD_ALL(LOAD_CV, cv)
_CUB_LOAD_ALL(LOAD_LDG, global.nc)

// Macro cleanup
#  undef _CUB_LOAD_ALL
#  undef _CUB_LOAD_1
#  undef _CUB_LOAD_2
#  undef _CUB_LOAD_4
#  undef _CUB_LOAD_8
#  undef _CUB_LOAD_16

#  if __cccl_ptx_isa >= 900
//! Define a 32 byte ThreadLoad specialization for LOAD_LDG. 32 byte accesses need the global state space, which only
//! LOAD_LDG names, and sm_100 or newer, thus older architectures load the word with two 16 byte accesses
template <>
_CCCL_DEVICE _CCCL_FORCEINLINE ulonglong4_32a ThreadLoad<LOAD_LDG, ulonglong4_32a const*>(ulonglong4_32a const* ptr)
{
  ulonglong4_32a retval;
  NV_IF_ELSE_TARGET(
    NV_PROVIDES_SM_100,
    (asm volatile("ld.global.nc.v4.u64 {%0, %1, %2, %3}, [%4];" : "=l"(retval.x),
                  "=l"(retval.y),
                  "=l"(retval.z),
                  "=l"(retval.w) : "l"(ptr));),
    (asm volatile("ld.global.nc.v2.u64 {%0, %1}, [%2];" : "=l"(retval.x), "=l"(retval.y) : "l"(ptr));
     asm volatile("ld.global.nc.v2.u64 {%0, %1}, [%2];" : "=l"(retval.z), "=l"(retval.w) : "l"(&ptr->z));))
  return retval;
}
#  endif // __cccl_ptx_isa >= 900

namespace detail
{
//! The words in which ThreadLoad loads a ``T`` with a PTX cache modifier: the widest words of at most 16 bytes that
//! ``T`` is aligned to, or 32 byte words with LOAD_LDG if ``T`` is aligned to them
#  if __cccl_ptx_isa >= 900
template <CacheLoadModifier Modifier, typename T>
using thread_load_word_t = ::cuda::std::
  conditional_t<Modifier == LOAD_LDG && alignof(T) % 32 == 0, ulonglong4_32a, typename UnitWord<T>::DeviceWord>;
#  else // ^^^ __cccl_ptx_isa >= 900 ^^^ / vvv __cccl_ptx_isa < 900 vvv
template <CacheLoadModifier Modifier, typename T>
using thread_load_word_t = typename UnitWord<T>::DeviceWord;
#  endif // __cccl_ptx_isa < 900
} // namespace detail

//! ThreadLoad definition for LOAD_DEFAULT modifier on iterator types
//! deprecated [Since 3.3]
template <typename RandomAccessIterator>
CCCL_DEPRECATED_BECAUSE("Use *itr instead") _CCCL_DEVICE_API
_CCCL_FORCEINLINE cub::detail::it_value_t<RandomAccessIterator>
ThreadLoad(RandomAccessIterator itr,
           detail::constant_t<LOAD_DEFAULT> /*modifier*/,
           ::cuda::std::false_type /*is_pointer*/)
{
  return *itr;
}

//! ThreadLoad definition for LOAD_DEFAULT modifier on pointer types
//! deprecated [Since 3.3]
template <typename T>
CCCL_DEPRECATED_BECAUSE("Use *itr instead") _CCCL_DEVICE_API _CCCL_FORCEINLINE T
ThreadLoad(const T* ptr, detail::constant_t<LOAD_DEFAULT> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  return *ptr;
}

//! ThreadLoad definition for LOAD_VOLATILE modifier on primitive pointer types
//! deprecated [Since 3.3]
template <typename T>
CCCL_DEPRECATED_BECAUSE("Use ThreadLoad<LOAD_VOLATILE>(ptr) instead") _CCCL_DEVICE_API _CCCL_FORCEINLINE T
ThreadLoadVolatilePointer(const T* ptr, ::cuda::std::true_type /*is_primitive*/)
{
  T retval = *reinterpret_cast<const volatile T*>(ptr);
  return retval;
}

//! ThreadLoad definition for LOAD_VOLATILE modifier on non-primitive pointer types
//! deprecated [Since 3.3]
template <typename T>
CCCL_DEPRECATED_BECAUSE("Use ThreadLoad<LOAD_VOLATILE>(ptr) instead") _CCCL_DEVICE_API _CCCL_FORCEINLINE T
ThreadLoadVolatilePointer(const T* ptr, ::cuda::std::false_type /*is_primitive*/)
{
  // Word type for memcpying
  using VolatileWord              = typename UnitWord<T>::VolatileWord;
  constexpr int VOLATILE_MULTIPLE = sizeof(T) / sizeof(VolatileWord);

  T retval;
  VolatileWord* words = reinterpret_cast<VolatileWord*>(&retval);
  _CUB::UnrolledCopy<VOLATILE_MULTIPLE>(reinterpret_cast<const volatile VolatileWord*>(ptr), words);
  return retval;
}

//! ThreadLoad definition for LOAD_VOLATILE modifier on pointer types
//! deprecated [Since 3.3]
template <typename T>
CCCL_DEPRECATED_BECAUSE("Use ThreadLoad<LOAD_VOLATILE>(ptr) instead") _CCCL_DEVICE_API _CCCL_FORCEINLINE T
ThreadLoad(const T* ptr, detail::constant_t<LOAD_VOLATILE> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  return _CUB::ThreadLoadVolatilePointer(ptr, detail::bool_constant_v<detail::is_primitive_v<T>>);
}

//! ThreadLoad definition for generic modifiers on pointer types
//! deprecated [Since 3.3]
template <typename T, CacheLoadModifier MODIFIER>
CCCL_DEPRECATED_BECAUSE("Use ThreadLoad(ptr) instead") _CCCL_DEVICE_API _CCCL_FORCEINLINE T
ThreadLoad(T const* ptr, detail::constant_t<MODIFIER> /*modifier*/, ::cuda::std::true_type /*is_pointer*/)
{
  using DeviceWord              = typename UnitWord<T>::DeviceWord;
  constexpr int DEVICE_MULTIPLE = sizeof(T) / sizeof(DeviceWord);

  DeviceWord words[DEVICE_MULTIPLE];
  _CUB::UnrolledThreadLoad<DEVICE_MULTIPLE, CacheLoadModifier(MODIFIER)>(
    reinterpret_cast<const DeviceWord*>(ptr), words);
  return *reinterpret_cast<T*>(words);
}

template <CacheLoadModifier MODIFIER, typename RandomAccessIterator>
_CCCL_DEVICE_API _CCCL_FORCEINLINE detail::it_value_t<RandomAccessIterator> ThreadLoad(RandomAccessIterator itr)
{
  using T = detail::it_value_t<RandomAccessIterator>;
  if constexpr (!THRUST_NS_QUALIFIER::is_contiguous_iterator_v<RandomAccessIterator> || MODIFIER == LOAD_DEFAULT)
  {
    return *itr;
  }
  else if constexpr (MODIFIER == LOAD_VOLATILE && detail::is_primitive_v<T>)
  {
    return *reinterpret_cast<const volatile T*>(::cuda::std::to_address(itr));
  }
  else if constexpr (MODIFIER == LOAD_VOLATILE)
  {
    using LoadWord              = typename UnitWord<T>::VolatileWord;
    constexpr int LOAD_MULTIPLE = sizeof(T) / sizeof(LoadWord);
    alignas(T) LoadWord words[LOAD_MULTIPLE]; // avoid default constructing a T
    _CUB::UnrolledCopy<LOAD_MULTIPLE>(reinterpret_cast<const volatile LoadWord*>(::cuda::std::to_address(itr)), words);
    // TODO(bgruber): start the lifetime of T here once cuda::std::start_lifetime_as is available
    return *reinterpret_cast<T*>(words);
  }
  else
  {
    using LoadWord = detail::thread_load_word_t<MODIFIER, T>;
    // NOLINTNEXTLINE(bugprone-sizeof-expression)
    constexpr int LOAD_MULTIPLE = sizeof(T) / sizeof(LoadWord);
    LoadWord words[LOAD_MULTIPLE];
    _CUB::UnrolledThreadLoad<LOAD_MULTIPLE, MODIFIER>(
      reinterpret_cast<const LoadWord*>(::cuda::std::to_address(itr)), words);
    return *reinterpret_cast<T*>(words);
  }
}

#endif // _CCCL_DOXYGEN_INVOKED

CUB_NAMESPACE_END
