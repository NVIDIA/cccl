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
#include <cuda/__driver/driver_api.h>
#include <cuda/__functional/operator_properties.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstddef/types.h>
#include <cuda/std/__exception/msg_storage.h>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_cv.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/cstdint>
#include <cuda/std/source_location>

#include <cuda/experimental/__nccl/shared_library.h>

#if _CCCL_HOSTED()
#  include <cstdio>
#endif // _CCCL_HOSTED()

#if __has_include(<nccl.h>)
#  include <nccl.h>
#  define _CCCL_HAS_NCCL() 1

#  include <cuda/std/__type_traits/decay.h>
#  include <cuda/std/__type_traits/is_enum.h>
#  include <cuda/std/__type_traits/is_function.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/remove_pointer.h>
#  include <cuda/std/__type_traits/underlying_type.h>
#else // ^^^ __has_include(<nccl.h>) ^^^ / vvv !__has_include(<nccl.h>) vvv
#  define _CCCL_HAS_NCCL() 0
#endif // !__has_include(<nccl.h>)

#include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)

extern "C" {
struct ncclComm;
} // extern "C"

namespace cuda::experimental::__nccl
{
[[nodiscard]] _CCCL_HOST_API inline __shared_library& __nccl_lib()
{
  static auto __lib = __shared_library{
#if _CCCL_OS(WINDOWS)
    /*__lib_path=*/"nccl.dll"
#elif _CCCL_OS(LINUX)
    /*__lib_path=*/"libnccl.so"
#endif
  };

  return __lib;
}

#if _CCCL_HAS_NCCL()
namespace __abi_detail
{
template <class _Tp, class _Up>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible() noexcept;

template <class _R1, class... _Args1, class _R2, class... _Args2>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible_func(_R1 (*)(_Args1...), _R2 (*)(_Args2...)) noexcept
{
  if constexpr (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_R1, _R2>()
                && (sizeof...(_Args1) == sizeof...(_Args2)))
  {
    return (::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_Args1, _Args2>() && ...);
  }
  return false;
}

template <class _Tp, class _Up>
[[nodiscard]] _CCCL_HOST_API constexpr bool __abi_compatible() noexcept
{
  using _RawTp = ::cuda::std::remove_cv_t<_Tp>;
  using _RawUp = ::cuda::std::remove_cv_t<_Up>;

  if constexpr (::cuda::std::is_same_v<_RawTp, _RawUp>)
  {
    // Equal types are obviously ABI compatible
    return true;
  }
  else if constexpr (::cuda::std::is_function_v<_RawTp> && ::cuda::std::is_function_v<_RawUp>)
  {
    // Functions need all arguments checked
    return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible_func(
      ::cuda::std::decay_t<_RawTp>{}, ::cuda::std::decay_t<_RawUp>{});
  }
  else if constexpr (::cuda::std::is_enum_v<_RawTp> || ::cuda::std::is_enum_v<_RawUp>)
  {
    // If either side is an enum, we need to unwrap to check whether the underlying types
    // match. These must match *exactly*, otherwise we perform the moral equivalent of a
    // bitcast when we reinterpret them
    if constexpr (::cuda::std::is_enum_v<_RawTp> && ::cuda::std::is_enum_v<_RawUp>)
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::underlying_type_t<_RawTp>,
                                                                          ::cuda::std::underlying_type_t<_RawUp>>();
    }
    else if constexpr (::cuda::std::is_enum_v<_RawTp>)
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::underlying_type_t<_RawTp>,
                                                                          _RawUp>();
    }
    else
    {
      return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<_RawTp,
                                                                          ::cuda::std::underlying_type_t<_RawUp>>();
    }
  }
  else if constexpr (::cuda::std::is_pointer_v<_RawTp> && ::cuda::std::is_pointer_v<_RawUp>)
  {
    // Note the &&. If one is a pointer but the other is not, that's an error
    return ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<::cuda::std::remove_pointer_t<_RawTp>,
                                                                        ::cuda::std::remove_pointer_t<_RawUp>>();
  }
  else
  {
    return false;
  }
}

#  if 0
// NOLINTBEGIN
enum Foo_enum
{
};

using FooEnum = unsigned int;

struct Foo_st;
using FooStruct = Foo_st*;
// NOLINTEND

static_assert(__abi_compatible<int, int>());
static_assert(!__abi_compatible<int, float>());
static_assert(__abi_compatible<const char**, const char* const*>());
static_assert(!__abi_compatible<const char*, const int*>());
static_assert(::cuda::std::is_same_v<FooEnum, ::cuda::std::underlying_type_t<Foo_enum>>);
static_assert(__abi_compatible<FooEnum, Foo_enum>());
static_assert(__abi_compatible<FooEnum, Foo_enum>());
static_assert(__abi_compatible<FooEnum*, Foo_enum*>());
static_assert(!__abi_compatible<FooEnum*, Foo_enum>());
static_assert(__abi_compatible<int (*)(Foo_st*), int (*)(FooStruct)>());
#  endif // 0
} // namespace __abi_detail

#  define _CCCL_LOAD_NCCL_SYMBOL(__symbol_name, ...)                                                           \
    ::cuda::experimental::__nccl::__nccl_lib().load_symbol<__VA_ARGS__>(#__symbol_name);                       \
    static_assert(                                                                                             \
      ::cuda::experimental::__nccl::__abi_detail::__abi_compatible<decltype(&::__symbol_name), __VA_ARGS__>(), \
      #__symbol_name " and " #__VA_ARGS__ " are not ABI compatible")

#else // ^^^ _CCCL_HAS_NCCL() ^^^ / vvv !_CCCL_HAS_NCCL() vvv
#  define _CCCL_LOAD_NCCL_SYMBOL(__symbol, ...) \
    ::cuda::experimental::__nccl::__nccl_lib().load_symbol<__VA_ARGS__>(#__symbol)
#endif // ^^^ !_CCCL_HAS_NCCL() ^^^

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

template <class _Tp>
[[nodiscard]] _CCCL_API constexpr __ncclDataType_t __nccl_type_of() noexcept
{
  if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::int8_t>)
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
  else if constexpr (::cuda::std::is_same_v<_Tp, ::cuda::std::size_t>
                     && (sizeof(::cuda::std::size_t) == sizeof(::cuda::std::uint64_t)))
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
    static_assert(::cuda::std::__always_false_v<_Tp>, "Unsupported type for NCCL");
  }
}

template <class _Tp>
_CCCL_GLOBAL_CONSTANT __ncclDataType_t __nccl_type_of_v =
  ::cuda::experimental::__nccl::__nccl_type_of<::cuda::std::remove_cvref_t<_Tp>>();

template <class _Tp>
_CCCL_CONCEPT __has_nccl_type_of = _CCCL_REQUIRES_EXPR((_Tp), )(__nccl_type_of_v<_Tp>);

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t __size_of(__ncclDataType_t __datatype)
{
  switch (__datatype)
  {
    case __ncclInt8:
      return sizeof(::cuda::std::int8_t);
    case __ncclUint8:
      return sizeof(::cuda::std::uint8_t);
    case __ncclInt32:
      return sizeof(::cuda::std::int32_t);
    case __ncclUint32:
      return sizeof(::cuda::std::uint32_t);
    case __ncclInt64:
      return sizeof(::cuda::std::int64_t);
    case __ncclUint64:
      return sizeof(::cuda::std::uint64_t);
    case __ncclFloat16:
#if _CCCL_HAS_NVFP16()
      return sizeof(::__half);
#else // ^^^ _CCCL_HAS_NVFP16() ^^^ / vvv !_CCCL_HAS_NVFP16() vvv
      _CCCL_THROW(::std::invalid_argument, "Unsupported NCCL datatype: __ncclFloat16");
#endif // _CCCL_HAS_NVFP16()
    case __ncclFloat32:
      return sizeof(float);
    case __ncclFloat64:
      return sizeof(double);
    case __ncclBfloat16:
#if _CCCL_HAS_NVBF16()
      return sizeof(::__nv_bfloat16);
#else // ^^^ _CCCL_HAS_NVBF16() ^^^ / vvv !_CCCL_HAS_NVBF16() vvv
      _CCCL_THROW(::std::invalid_argument, "Unsupported NCCL datatype: __ncclBfloat16");
#endif // _CCCL_HAS_NVBF16()
    case __ncclFloat8e4m3:
#if _CCCL_HAS_NVFP8()
      return sizeof(::__nv_fp8_e4m3);
#else // ^^^ _CCCL_HAS_NVFP8() ^^^ / vvv !_CCCL_HAS_NVFP8() vvv
      _CCCL_THROW(::std::invalid_argument, "Unsupported NCCL datatype: __ncclFloat8e4m3");
#endif // _CCCL_HAS_NVFP8()
    case __ncclFloat8e5m2:
#if _CCCL_HAS_NVFP8()
      return sizeof(::__nv_fp8_e5m2);
#else // ^^^ _CCCL_HAS_NVFP8() ^^^ / vvv !_CCCL_HAS_NVFP8() vvv
      _CCCL_THROW(::std::invalid_argument, "Unsupported NCCL datatype: __ncclFloat8e5m2");
#endif // _CCCL_HAS_NVFP8()
    case __ncclNumTypes:
      _CCCL_THROW(::std::invalid_argument, "Unhandled NCCL datatype: __ncclNumTypes");
  }

  _CCCL_THROW(::std::invalid_argument, "Unhandled NCCL datatype value");
}

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

template <class _Op>
[[nodiscard]] _CCCL_API constexpr __ncclRedOp_t __nccl_redop_of() noexcept
{
  if constexpr (::cuda::__is_cuda_std_plus_v<_Op>)
  {
    return __ncclSum;
  }
  else if constexpr (::cuda::__is_cuda_std_multiplies_v<_Op>)
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
    static_assert(::cuda::std::__always_false_v<_Op>, "Unsupported nccl reduction operator");
  }
}

template <class _Op>
_CCCL_GLOBAL_CONSTANT __ncclRedOp_t __nccl_redop_of_v =
  ::cuda::experimental::__nccl::__nccl_redop_of<::cuda::std::remove_cvref_t<_Op>>();

template <class _Op>
_CCCL_CONCEPT __has_nccl_redop = _CCCL_REQUIRES_EXPR((_Op), )(__nccl_redop_of_v<_Op>);

using __ncclComm_t = ::ncclComm*;

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
  [[nodiscard]] static const char* __format_nccl_error(
    ::cuda::__msg_storage& __msg_buffer,
    const __ncclResult_t __result,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc) noexcept
  {
    ::snprintf(
      __msg_buffer.__buffer,
      __msg_buffer.__size, // NOLINT(readability-static-accessed-through-instance)
      "%s:%d %s%s%s(%d): %s",
      __loc.file_name(),
      __loc.line(),
      __api ? __api : "",
      __api ? " " : "",
      ::cuda::experimental::__nccl::__ncclGetErrorStringNoThrow(__result),
      static_cast<::cuda::std::int32_t>(__result),
      __msg);

    return __msg_buffer.__buffer;
  }

public:
  nccl_error(const __ncclResult_t __result,
             const char* __msg,
             const char* __api                  = nullptr,
             ::cuda::std::source_location __loc = ::cuda::std::source_location::current(),
             ::cuda::__msg_storage __msg_buffer = {}) noexcept
      : ::std::runtime_error{__format_nccl_error(__msg_buffer, __result, __msg, __api, __loc)}
      , __result_{__result}
  {}

  [[nodiscard]] constexpr __ncclResult_t status() const noexcept
  {
    return __result_;
  }

private:
  __ncclResult_t __result_;
};

#else // ^^^ _CCCL_HOSTED() ^^^ / vvv !_CCCL_HOSTED() vvv

class nccl_error;

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
  {
    ::cuda::experimental::__nccl::__ncclGroupStart();
  }

  _CCCL_HIDE_FROM_ABI __ensure_nccl_group(const __ensure_nccl_group&) = delete;
  _CCCL_HIDE_FROM_ABI void operator=(const __ensure_nccl_group&)      = delete;
  _CCCL_HIDE_FROM_ABI __ensure_nccl_group(__ensure_nccl_group&&)      = delete;
  _CCCL_HIDE_FROM_ABI void operator=(__ensure_nccl_group&&)           = delete;

  _CCCL_HOST_API ~__ensure_nccl_group() noexcept(false)
  {
    ::cuda::experimental::__nccl::__ncclGroupEnd();
  }
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
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
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
  static auto* const __fn = _CCCL_LOAD_NCCL_SYMBOL(
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

#undef _CCCL_LOAD_NCCL_SYMBOL

// ==========================================================================================

// Extensions to the NCCL API

_CCCL_HOST_API inline void __ncclGatherv(
  const void* __sendbuf,
  ::cuda::std::size_t __sendcount,
  void* __recvbuf,
  __ncclDataType_t __datatype,
  const ::cuda::std::size_t __h_recvcounts[],
  const ::cuda::std::size_t __h_displs[],
  int __root,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  // Technically group is only needed on the root rank since the ncclSend() is local the the
  // sender (in case of multiple local sends, the caller needs to call the group API). But it
  // cannot hurt to insert the group here just in case, because the NCCL docs don't say
  // anything about whether *all* ranks in a communication need to have the group setup if one
  // or some of them do.
  const auto _      = __ensure_nccl_group{};
  const auto __rank = ::cuda::experimental::__nccl::__ncclCommUserRank(__comm);

  if (__rank == __root)
  {
    const auto __size      = ::cuda::experimental::__nccl::__ncclCommCount(__comm);
    const auto __data_size = ::cuda::experimental::__nccl::__size_of(__datatype);

    for (int __peer = 0; __peer < __size; ++__peer)
    {
      const auto __count = __h_recvcounts[__peer];

      if (__count == 0)
      {
        continue;
      }

      auto* const __recv_ptr = static_cast<char*>(__recvbuf) + (__h_displs[__peer] * __data_size);

      if (__peer == __root)
      {
        // Unclear whether CUDA driver also makes this optimization
        if (__sendbuf != __recv_ptr)
        {
          ::cuda::__driver::__memcpyAsync(__recv_ptr, __sendbuf, __sendcount * __data_size, __stream.get());
        }
      }
      else
      {
        ::cuda::experimental::__nccl::__ncclRecv(__recv_ptr, __count, __datatype, __peer, __comm, __stream);
      }
    }
  }
  else if (__sendcount != 0)
  {
    ::cuda::experimental::__nccl::__ncclSend(__sendbuf, __sendcount, __datatype, __root, __comm, __stream);
  }
}

_CCCL_HOST_API inline void __ncclAlltoAllv(
  const void* __sendbuf,
  const ::cuda::std::size_t __h_send_counts[],
  const ::cuda::std::size_t __h_send_displs[],
  void* __recvbuf,
  const ::cuda::std::size_t __h_recv_counts[],
  const ::cuda::std::size_t __h_recv_displs[],
  __ncclDataType_t __datatype,
  __ncclComm_t __comm,
  ::cuda::stream_ref __stream)
{
  const auto _           = __ensure_nccl_group{};
  const auto __rank      = ::cuda::experimental::__nccl::__ncclCommUserRank(__comm);
  const auto __size      = ::cuda::experimental::__nccl::__ncclCommCount(__comm);
  const auto __data_size = ::cuda::experimental::__nccl::__size_of(__datatype);

  for (int __peer = 0; __peer < __size; ++__peer)
  {
    const auto __sendcount = __h_send_counts[__peer];
    const auto __recvcount = __h_recv_counts[__peer];

    if (__peer == __rank)
    {
      if (__sendcount != __recvcount)
      {
        _CCCL_THROW(::cuda::experimental::__nccl::nccl_error,
                    __ncclInvalidArgument,
                    "Mismatched self-copy count in ncclAlltoAllv",
                    "ncclAlltoAllv");
      }

      if (__sendcount == 0)
      {
        continue;
      }

      const auto __size_bytes      = __sendcount * __data_size;
      const auto* const __send_ptr = static_cast<const char*>(__sendbuf) + (__h_send_displs[__peer] * __data_size);
      auto* const __recv_ptr       = static_cast<char*>(__recvbuf) + (__h_recv_displs[__peer] * __data_size);

      if (__send_ptr != __recv_ptr)
      {
        ::cuda::__driver::__memcpyAsync(__recv_ptr, __send_ptr, __size_bytes, __stream.get());
      }
      continue;
    }

    if (__recvcount != 0)
    {
      auto* const __recv_ptr = static_cast<char*>(__recvbuf) + (__h_recv_displs[__peer] * __data_size);

      ::cuda::experimental::__nccl::__ncclRecv(__recv_ptr, __recvcount, __datatype, __peer, __comm, __stream);
    }

    if (__sendcount != 0)
    {
      const auto* const __send_ptr = static_cast<const char*>(__sendbuf) + (__h_send_displs[__peer] * __data_size);

      ::cuda::experimental::__nccl::__ncclSend(__send_ptr, __sendcount, __datatype, __peer, __comm, __stream);
    }
  }
}
} // namespace cuda::experimental::__nccl

// NOLINTEND(bugprone-reserved-identifier)

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___NCCL_NCCL_API_H
