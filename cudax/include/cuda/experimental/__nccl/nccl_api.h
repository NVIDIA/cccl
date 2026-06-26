//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___NCCL_NCCL_API_H
#define _CUDA_EXPERIMENTAL___NCCL_NCCL_API_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/device_ref.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/msg_storage.h>
#include <cuda/std/__functional/operations_traits.h>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstdint>
#include <cuda/std/source_location>

#include <cuda/experimental/__nccl/shared_library.h>

#if _CCCL_HOSTED()
#  include <cstdio> // snprintf
#  include <exception> // uncaught_exceptions
#endif // _CCCL_HOSTED()

#define _CCCL_NCCL()     _CCCL_VERSION_INVALID()
#define _CCCL_HAS_NCCL() 0

#if __has_include(<nccl.h>)
#  include <nccl.h>

#  undef _CCCL_HAS_NCCL
#  define _CCCL_HAS_NCCL() 1

#  if !defined(NCCL_MAJOR) || !defined(NCCL_MINOR)
#    error "Unsupported NCCL version which doesn't define NCCL_MAJOR and/or NCCL_MINOR"
#  endif // No NCCL_MAJOR or NCCL_MINOR

#  undef _CCCL_NCCL
#  define _CCCL_NCCL() (NCCL_MAJOR, NCCL_MINOR)

#  include <cuda/experimental/__nccl/abi_compatible.h>
#endif // __has_include(<nccl.h>)

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

extern "C" {
struct ncclComm;
} // extern "C"

// Taken from nccl.h.in
#define _CCCL_NCCL_MAKE_VERSION(_MAJOR, _MINOR) \
  (((_MAJOR) <= 2 && (_MINOR) <= 8) ? (_MAJOR) * 1000 + (_MINOR) * 100 : (_MAJOR) * 10000 + (_MINOR) * 100)
#define _CCCL_NCCL_VERSION(...) _CCCL_VERSION_COMPARE(_CCCL_NCCL_, _CCCL_NCCL, __VA_ARGS__)

namespace cuda::experimental::__nccl
{
[[nodiscard]] _CCCL_HOST_API inline __shared_library& __nccl_lib()
{
  static auto __lib = __shared_library{
#if _CCCL_OS(WINDOWS)
    /*__lib_path=*/"nccl.dll"
#elif _CCCL_OS(APPLE)
    /*__lib_path=*/"libnccl.dylib"
#elif _CCCL_OS(LINUX)
    /*__lib_path=*/"libnccl.so"
#else
#  error "Unknown nccl library name for platform, please report a bug at https://github.com/NVIDIA/cccl/issues"
#endif
  };

  return __lib;
}

#if _CCCL_HAS_NCCL()
#  define _CCCL_LOAD_NCCL_SYMBOL_ENABLE_ABI_CHECK_IF(__cond, __symbol, ...)                                          \
    ::cuda::experimental::__nccl::__nccl_lib().load_symbol<__VA_ARGS__>(#__symbol);                                  \
    static_assert(                                                                                                   \
      _CCCL_PP_IIF(__cond)(                                                                                          \
        (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<decltype(&::__symbol), __VA_ARGS__>()), true), \
      #__symbol " and " #__VA_ARGS__ " are not ABI compatible")

#  define _CCCL_LOAD_NCCL_SYMBOL(__symbol, ...)                                                                       \
    ::cuda::experimental::__nccl::__nccl_lib().load_symbol<__VA_ARGS__>(#__symbol);                                   \
    static_assert(::cuda::experimental::__nccl::__abi_detail::__abi_compatible<decltype(&::__symbol), __VA_ARGS__>(), \
                  #__symbol " and " #__VA_ARGS__ " are not ABI compatible")

#else // ^^^ _CCCL_HAS_NCCL() ^^^ / vvv !_CCCL_HAS_NCCL() vvv

#  define _CCCL_LOAD_NCCL_SYMBOL_ENABLE_ABI_CHECK_IF(__cond, __symbol, ...) \
    ::cuda::experimental::__nccl::__nccl_lib().load_symbol<__VA_ARGS__>(#__symbol)

#  define _CCCL_LOAD_NCCL_SYMBOL(__symbol, ...) \
    ::cuda::experimental::__nccl::__nccl_lib().load_symbol<__VA_ARGS__>(#__symbol)

#endif // ^^^ !_CCCL_HAS_NCCL() ^^^

#if _CCCL_NCCL_VERSION(>=, 2, 28)
#  define _CCCL_HAS_NCCL_2_28() 1
#else // ^^^ nccl 2.28+ ^^^ / vvv nccl.2.27- vvv
#  define _CCCL_HAS_NCCL_2_28() 0
#endif // ^^^ nccl 2.27- ^^^

// NCCL forward decls
// ==========================================================================================

enum __ncclResult_t // NOLINT(performance-enum-size)
{
  __ncclSuccess,
  __ncclUnhandledCudaError,
  __ncclSystemError,
  __ncclInternalError,
  __ncclInvalidArgument,
  __ncclInvalidUsage,
  __ncclRemoteError,
  __ncclInProgress,
  __ncclTimeout,
  __ncclNumResults
};

#if _CCCL_HAS_NCCL()
static_assert(__ncclSuccess == ::ncclSuccess);
static_assert(__ncclUnhandledCudaError == ::ncclUnhandledCudaError);
static_assert(__ncclSystemError == ::ncclSystemError);
static_assert(__ncclInternalError == ::ncclInternalError);
static_assert(__ncclInvalidArgument == ::ncclInvalidArgument);
static_assert(__ncclInvalidUsage == ::ncclInvalidUsage);
#  if _CCCL_NCCL_VERSION(>=, 2, 13)
static_assert(__ncclRemoteError == ::ncclRemoteError);
#  endif // NCCL 2.13+
#  if _CCCL_NCCL_VERSION(>=, 2, 14)
static_assert(__ncclInProgress == ::ncclInProgress);
#  endif // NCCL 2.14+
#  if _CCCL_NCCL_VERSION(>=, 2, 30)
static_assert(__ncclTimeout == ::ncclTimeout);
#  endif // NCCL 2.30+
#endif // _CCCL_HAS_NCCL

enum __ncclDataType_t // NOLINT(performance-enum-size)
{
  __ncclInt8       = 0,
  __ncclChar       = __ncclInt8,
  __ncclUint8      = 1,
  __ncclInt32      = 2,
  __ncclInt        = __ncclInt32,
  __ncclUint32     = 3,
  __ncclInt64      = 4,
  __ncclUint64     = 5,
  __ncclFloat16    = 6,
  __ncclHalf       = __ncclFloat16,
  __ncclFloat32    = 7,
  __ncclFloat      = __ncclFloat32,
  __ncclFloat64    = 8,
  __ncclDouble     = __ncclFloat64,
  __ncclBfloat16   = 9,
  __ncclFloat8e4m3 = 10,
  __ncclFloat8e5m2 = 11,
  __ncclNumTypes   = 12
};

#if _CCCL_HAS_NCCL()
// Do not check NumTypes. If NCCL adds new types after these values, we don't care (until we
// support them)
static_assert(__ncclInt8 == ::ncclInt8);
static_assert(__ncclChar == ::ncclChar);
static_assert(__ncclUint8 == ::ncclUint8);
static_assert(__ncclInt32 == ::ncclInt32);
static_assert(__ncclInt == ::ncclInt);
static_assert(__ncclUint32 == ::ncclUint32);
static_assert(__ncclInt64 == ::ncclInt64);
static_assert(__ncclUint64 == ::ncclUint64);
static_assert(__ncclFloat16 == ::ncclFloat16);
static_assert(__ncclHalf == ::ncclHalf);
static_assert(__ncclFloat32 == ::ncclFloat32);
static_assert(__ncclFloat == ::ncclFloat);
static_assert(__ncclFloat64 == ::ncclFloat64);
static_assert(__ncclDouble == ::ncclDouble);
#  if (_CCCL_NCCL_VERSION(>=, 2, 10) && defined(__CUDA_BF16_TYPES_EXIST__)) || _CCCL_NCCL_VERSION(>=, 2, 24)
static_assert(__ncclBfloat16 == ::ncclBfloat16);
#  endif // NCCL [2.10 - 2.24) and cuda_bf16. included, or NCCL 2.24+
#  if _CCCL_NCCL_VERSION(>=, 2, 24)
static_assert(__ncclFloat8e4m3 == ::ncclFloat8e4m3);
static_assert(__ncclFloat8e5m2 == ::ncclFloat8e5m2);
#  endif // NCCL 2.24+
#endif // _CCCL_HAS_NCCL

enum __ncclRedOp_dummy_t // NOLINT(performance-enum-size)
{
  __ncclNumOps_dummy = 5
};

enum __ncclRedOp_t // NOLINT(performance-enum-size)
{
  __ncclSum  = 0,
  __ncclProd = 1,
  __ncclMax  = 2,
  __ncclMin  = 3,
  __ncclAvg  = 4,
  /* ncclNumOps: The number of built-in ncclRedOp_t values. Also
   * serves as the least possible value for dynamic ncclRedOp_t's
   * as constructed by ncclRedOpCreate*** functions. */
  __ncclNumOps = 5,
  /* ncclMaxRedOp: The largest valid value for ncclRedOp_t.
   * It is defined to be the largest signed value (since compilers
   * are permitted to use signed enums) that won't grow
   * sizeof(ncclRedOp_t) when compared to previous NCCL versions to
   * maintain ABI compatibility. */
  __ncclMaxRedOp = 0x7fffffff >> (32 - (8 * sizeof(__ncclRedOp_dummy_t)))
};

#if _CCCL_HAS_NCCL()
// Do not check NumOps or MaxRedOp. These aren't guaranteed to be in older versions
static_assert(__ncclSum == ::ncclSum);
static_assert(__ncclProd == ::ncclProd);
static_assert(__ncclMax == ::ncclMax);
static_assert(__ncclMin == ::ncclMin);
static_assert(__ncclAvg == ::ncclAvg);
#endif // _CCCL_HAS_NCCL

using __ncclComm_t = ::ncclComm*;

// Helpers and concepts
// ==========================================================================================

struct __no_nccl_type
{};

template <class _Tp>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __nccl_type_of() noexcept
{
  if constexpr (::cuda::std::is_same_v<_Tp, bool>)
  {
    if constexpr (sizeof(bool) == sizeof(char))
    {
      return __ncclChar;
    }
    else if constexpr (sizeof(bool) == sizeof(::cuda::std::int32_t))
    {
      // Apparently ancient Visual studio used int
      return __ncclInt32;
    }
    else
    {
      static_assert(!::cuda::std::is_same_v<_Tp, _Tp>, "Unknown platform boolean size");
    }
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::int8_t>)
  {
    return __ncclInt8;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, char> && !::cuda::std::is_same_v<char, ::cuda::std::int8_t>)
  {
    return __ncclChar;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint8_t>)
  {
    return __ncclUint8;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::int32_t>)
  {
    return __ncclInt32;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint32_t>)
  {
    return __ncclUint32;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::int64_t>)
  {
    return __ncclInt64;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::uint64_t>)
  { // NOLINT(bugprone-branch-clone)
    return __ncclUint64;
  }
  // On some platforms size_t != uint64_t
  else if constexpr ((sizeof(::cuda::std::size_t) == sizeof(::cuda::std::uint64_t))
                     && (alignof(::cuda::std::size_t) == alignof(::cuda::std::uint64_t))
                     && ::cuda::std::is_same_v<_Tp, ::cuda::std::size_t>)
  {
    return __ncclUint64;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, float>)
  {
    return __ncclFloat;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, double>)
  {
    return __ncclDouble;
  }
#if _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__half>)
  {
    return __ncclHalf;
  }
#endif // _CCCL_HAS_NVFP16()
#if _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_bfloat16>)
  {
    return __ncclBfloat16;
  }
#endif // _CCCL_HAS_NVBF16()
#if _CCCL_HAS_NVFP8()
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_fp8_e4m3>)
  {
    return __ncclFloat8e4m3;
  }
  else if constexpr (::cuda::std::is_same_v<_Tp, ::__nv_fp8_e5m2>)
  {
    return __ncclFloat8e5m2;
  }
#endif // _CCCL_HAS_NVFP8()
  else
  {
    return __no_nccl_type{};
  }
  _CCCL_UNREACHABLE();
}

template <class _Tp>
inline constexpr __ncclDataType_t __nccl_type_of_v =
  ::cuda::experimental::__nccl::__nccl_type_of<::cuda::std::remove_cvref_t<_Tp>>();

template <class _Tp>
_CCCL_CONCEPT __has_nccl_type_of = _CCCL_REQUIRES_EXPR((_Tp), )(
  _Same_as(__ncclDataType_t)::cuda::experimental::__nccl::__nccl_type_of<::cuda::std::remove_cvref_t<_Tp>>());

// ------------------------------------------------------------------------------------------

struct __no_nccl_redop
{};

template <class _Op>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __nccl_redop_of() noexcept
{
  if constexpr (::cuda::std::__is_plus_op_v<_Op>)
  {
    return __ncclSum;
  }
  else if constexpr (::cuda::std::__is_multiplies_op_v<_Op>)
  {
    return __ncclProd;
  }
  else if constexpr (::cuda::__is_cuda_maximum_v<_Op>)
  {
    return __ncclMax;
  }
  else if constexpr (::cuda::__is_cuda_minimum_v<_Op>)
  {
    return __ncclMin;
  }
  else
  {
    return __no_nccl_redop{};
  }
  _CCCL_UNREACHABLE();
}

template <class _Op>
inline constexpr __ncclRedOp_t __nccl_redop_of_v =
  ::cuda::experimental::__nccl::__nccl_redop_of<::cuda::std::remove_cvref_t<_Op>>();

template <class _Op>
_CCCL_CONCEPT __has_nccl_redop_of = _CCCL_REQUIRES_EXPR((_Op), )(
  _Same_as(__ncclRedOp_t)::cuda::experimental::__nccl::__nccl_redop_of<::cuda::std::remove_cvref_t<_Op>>());

// API wrappers
// ==========================================================================================

[[nodiscard]] _CCCL_HOST_API inline const char* __ncclGetLastErrorNoThrow(__ncclComm_t __comm) noexcept
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclGetLastError, const char* (*) (__ncclComm_t));

  return __fn(__comm);
}

[[nodiscard]] _CCCL_HOST_API inline const char* __ncclGetErrorStringNoThrow(__ncclResult_t __result) noexcept
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclGetErrorString, const char* (*) (__ncclResult_t));

  return __fn(__result);
}

// ==========================================================================================

#if _CCCL_HOSTED()

class nccl_error final : public ::std::runtime_error
{
  [[nodiscard]] _CCCL_HOST_API static const char* __format_nccl_error(
    ::cuda::__msg_storage& __msg_buffer,
    __ncclResult_t __result,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc) noexcept
  {
    static_cast<void>(::snprintf(
      __msg_buffer.__buffer,
      ::cuda::__msg_storage::__size,
      "%s:%d %s%s%s(%d): %s",
      __loc.file_name(),
      __loc.line(),
      __api ? __api : "",
      __api ? " " : "",
      ::cuda::experimental::__nccl::__ncclGetErrorStringNoThrow(__result),
      static_cast<::cuda::std::int32_t>(__result),
      __msg));

    return __msg_buffer.__buffer;
  }

public:
  _CCCL_HOST_API nccl_error(
    __ncclResult_t __result,
    const char* __msg,
    const char* __api                  = nullptr,
    ::cuda::std::source_location __loc = ::cuda::std::source_location::current(),
    ::cuda::__msg_storage __msg_buffer = {}) noexcept
      : ::std::runtime_error{__format_nccl_error(__msg_buffer, __result, __msg, __api, __loc)}
      , __result_{__result}
  {}

  [[nodiscard]] _CCCL_HOST_API constexpr __ncclResult_t status() const noexcept
  {
    return __result_;
  }

private:
  __ncclResult_t __result_;
};

#else // ^^^ _CCCL_HOSTED() ^^^ / vvv !_CCCL_HOSTED() vvv

class nccl_error final
{};

#endif // ^^^ !_CCCL_HOSTED() ^^^

// ==========================================================================================

[[nodiscard]] _CCCL_HOST_API inline int __ncclCommCount(__ncclComm_t __comm)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclCommCount, __ncclResult_t (*)(__ncclComm_t, int*));

  int __count{};

  if (const auto __ret = __fn(__comm, &__count); __ret != __ncclSuccess)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclCommCount", "ncclCommCount");
  }
  return __count;
}

[[nodiscard]] _CCCL_HOST_API inline int __ncclCommUserRank(__ncclComm_t __comm)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclCommUserRank, __ncclResult_t (*)(__ncclComm_t, int*));

  int __rank{};

  if (const auto __ret = __fn(__comm, &__rank); __ret != __ncclSuccess)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclCommUserRank", "ncclCommUserRank");
  }
  return __rank;
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::device_ref __ncclCommCuDevice(__ncclComm_t __comm)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclCommCuDevice, __ncclResult_t (*)(__ncclComm_t, int*));

  int __device{};

  if (const auto __ret = __fn(__comm, &__device); __ret != __ncclSuccess)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclCommCuDevice", "ncclCommCuDevice");
  }

  return {__device};
}

// ==========================================================================================

_CCCL_HOST_API inline void __ncclGroupStart()
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclGroupStart, __ncclResult_t (*)());

  if (const auto __ret = __fn(); __ret != __ncclSuccess)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclGroupStart", "ncclGroupStart");
  }
}

[[nodiscard]] _CCCL_HOST_API inline __ncclResult_t __ncclGroupEndNoThrow() noexcept
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(ncclGroupEnd, __ncclResult_t (*)());

  return __fn();
}

_CCCL_HOST_API inline void __ncclGroupEnd()
{
  if (const auto __ret = ::cuda::experimental::__nccl::__ncclGroupEndNoThrow();
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclGroupEnd", "ncclGroupEnd");
  }
}

class __ensure_nccl_group
{
public:
  _CCCL_HOST_API __ensure_nccl_group()
      : __uncaught_on_entry_{::std::uncaught_exceptions()}
  {
    ::cuda::experimental::__nccl::__ncclGroupStart();
  }

  _CCCL_HIDE_FROM_ABI __ensure_nccl_group(const __ensure_nccl_group&) = delete;
  _CCCL_HIDE_FROM_ABI void operator=(const __ensure_nccl_group&)      = delete;
  _CCCL_HIDE_FROM_ABI __ensure_nccl_group(__ensure_nccl_group&&)      = delete;
  _CCCL_HIDE_FROM_ABI void operator=(__ensure_nccl_group&&)           = delete;

  _CCCL_HOST_API ~__ensure_nccl_group() noexcept(false)
  {
    if (::std::uncaught_exceptions() > __uncaught_on_entry_)
    {
      static_cast<void>(::cuda::experimental::__nccl::__ncclGroupEndNoThrow());
    }
    else
    {
      ::cuda::experimental::__nccl::__ncclGroupEnd();
    }
  }

private:
  // This is needed in case __ensure_nccl_group is constructed *inside* a catch block (or
  // anywhere an exception is already in the process of being handled). We need to identify
  // exactly the situation where we are unwinding as a result of a new exception during the
  // lifetime of this object.
  int __uncaught_on_entry_{};
};

// ==========================================================================================

_CCCL_HOST_API inline void __ncclAllReduce(
  const void* __sendbuff,
  void* __recvbuff,
  ::cuda::std::size_t __count,
  __ncclDataType_t __datatype,
  __ncclRedOp_t __op,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
    ncclAllReduce,
    __ncclResult_t (*)(
      const void*, void*, ::cuda::std::size_t, __ncclDataType_t, __ncclRedOp_t, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __recvbuff, __count, __datatype, __op, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclAllReduce", "ncclAllReduce");
  }
}

_CCCL_HOST_API inline void __ncclReduce(
  const void* __sendbuff,
  void* __recvbuff,
  ::cuda::std::size_t __count,
  __ncclDataType_t __datatype,
  __ncclRedOp_t __op,
  int __root,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
    ncclReduce,
    __ncclResult_t (*)(
      const void*, void*, ::cuda::std::size_t, __ncclDataType_t, __ncclRedOp_t, int, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __recvbuff, __count, __datatype, __op, __root, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclReduce", "ncclReduce");
  }
}

_CCCL_HOST_API inline void __ncclAllGather(
  const void* __sendbuff,
  void* __recvbuff,
  ::cuda::std::size_t __sendcount,
  __ncclDataType_t __datatype,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
    ncclAllGather,
    __ncclResult_t (*)(const void*, void*, ::cuda::std::size_t, __ncclDataType_t, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __recvbuff, __sendcount, __datatype, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclAllGather", "ncclAllGather");
  }
}

_CCCL_HOST_API inline void __ncclBroadcast(
  const void* __sendbuff,
  void* __recvbuff,
  ::cuda::std::size_t __count,
  __ncclDataType_t __datatype,
  int __root,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
    ncclBroadcast,
    __ncclResult_t (*)(const void*, void*, ::cuda::std::size_t, __ncclDataType_t, int, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __recvbuff, __count, __datatype, __root, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclBroadcast", "ncclBroadcast");
  }
}

_CCCL_HOST_API inline void __ncclGather(
  const void* __sendbuff,
  void* __recvbuff,
  ::cuda::std::size_t __sendcount,
  __ncclDataType_t __datatype,
  int __root,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  // ncclGather only since 2.28.
  //
  // TODO(jfaibussowit): If gather doesn't exist, we could try and implement it ourselves. The
  // NCCL docs show an example implementation
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-one-gather
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL_ENABLE_ABI_CHECK_IF(
    _CCCL_HAS_NCCL_2_28(),
    ncclGather,
    __ncclResult_t (*)(const void*, void*, ::cuda::std::size_t, __ncclDataType_t, int, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __recvbuff, __sendcount, __datatype, __root, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclGather", "ncclGather");
  }
}

_CCCL_HOST_API inline void __ncclAlltoAll(
  const void* __sendbuff,
  void* __recvbuff,
  ::cuda::std::size_t __count,
  __ncclDataType_t __datatype,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  // ncclAllToAll only since 2.28
  //
  // TODO(jfaibussowit): If all-to-all doesn't exist, we could try and implement it
  // ourselves. The NCCL docs show an example implementation
  // https://docs.nvidia.com/deeplearning/nccl/user-guide/docs/usage/p2p.html#all-to-all
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL_ENABLE_ABI_CHECK_IF(
    _CCCL_HAS_NCCL_2_28(),
    ncclAlltoAll,
    __ncclResult_t (*)(const void*, void*, ::cuda::std::size_t, __ncclDataType_t, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __recvbuff, __count, __datatype, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclAlltoAll", "ncclAlltoAll");
  }
}

// ==========================================================================================

_CCCL_HOST_API inline void __ncclSend(
  const void* __sendbuff,
  ::cuda::std::size_t __count,
  __ncclDataType_t __datatype,
  int __peer,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
    ncclSend, __ncclResult_t (*)(const void*, ::cuda::std::size_t, __ncclDataType_t, int, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__sendbuff, __count, __datatype, __peer, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclSend", "ncclSend");
  }
}

_CCCL_HOST_API inline void __ncclRecv(
  void* __recvbuff,
  ::cuda::std::size_t __count,
  __ncclDataType_t __datatype,
  int __peer,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
    ncclRecv, __ncclResult_t (*)(void*, ::cuda::std::size_t, __ncclDataType_t, int, __ncclComm_t, ::CUstream));

  if (const auto __ret = __fn(__recvbuff, __count, __datatype, __peer, __comm, __stream.get());
      __ret != __ncclSuccess && __ret != __ncclInProgress)
  {
    _CCCL_THROW(::cuda::experimental::__nccl::nccl_error, __ret, "Error in ncclRecv", "ncclRecv");
  }
}

// Clean up
#undef _CCCL_LOAD_NCCL_SYMBOL
#undef _CCCL_LOAD_NCCL_SYMBOL_ENABLE_ABI_CHECK_IF
#undef _CCCL_HAS_NCCL_2_28
} // namespace cuda::experimental::__nccl

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___NCCL_NCCL_API_H
