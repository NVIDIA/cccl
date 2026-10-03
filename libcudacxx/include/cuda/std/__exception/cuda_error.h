//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___EXCEPTION_CUDA_ERROR_H
#define _CUDA_STD___EXCEPTION_CUDA_ERROR_H

// IWYU pragma: always_keep

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__driver/entry_point.h>
#include <cuda/std/__cstring/memcpy.h>
#include <cuda/std/__exception/exception_macros.h>
#include <cuda/std/__exception/msg_storage.h>
#include <cuda/std/__host_stdlib/cstdio>
#include <cuda/std/__host_stdlib/stdexcept>
#include <cuda/std/__type_traits/always_false.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_trivially_copyable.h>
#include <cuda/std/__utility/typeid.h>
#include <cuda/std/source_location>
#include <cuda/std/string_view>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

#if _CCCL_HAS_CTK()
using __cuda_error_t = ::cudaError_t;
#else
using __cuda_error_t = int;
#endif

#if _CCCL_HOSTED()
/**
 * @brief The rules `cuda_error` applies to a status value of type `_Status`. The defaults are the rules
 * every CUDA library's status enumeration follows: zero is success, the value is the code, and no text is
 * known. A library whose status is a struct, or that can describe its codes, specializes
 * @ref cuda_status_traits and may inherit these defaults for the members it keeps.
 */
template <class _Status>
struct cuda_status_defaults
{
  [[nodiscard]] _CCCL_HOST_API static constexpr bool failed(const _Status __status) noexcept
  {
    return static_cast<long long>(__status) != 0;
  }
  [[nodiscard]] _CCCL_HOST_API static constexpr long long raw_code(const _Status __status) noexcept
  {
    return static_cast<long long>(__status);
  }
  [[nodiscard]] _CCCL_HOST_API static constexpr const char* text(const _Status) noexcept
  {
    return nullptr;
  }
};

/**
 * @brief Customization point for status types that `cuda_error` can carry. The primary template applies
 * @ref cuda_status_defaults, so any status enumeration works unchanged. Specialize it to supply text, or
 * to carry a struct status such as cuFile's:
 *
 * @code
 * template <> struct cuda::cuda_status_traits<cufftResult> : cuda::cuda_status_defaults<cufftResult>
 * {
 *   static const char* text(cufftResult r) noexcept { return my_cufft_text(r); }
 * };
 * template <> struct cuda::cuda_status_traits<CUfileError_t>
 * {
 *   static bool failed(CUfileError_t s) noexcept { return s.err != CU_FILE_SUCCESS; }
 *   static long long raw_code(CUfileError_t s) noexcept { return s.err; }
 *   static const char* text(CUfileError_t s) noexcept { return cufileop_status_error(s.err); }
 * };
 * @endcode
 */
template <class _Status>
struct cuda_status_traits : cuda_status_defaults<_Status>
{};

#  if _CCCL_HAS_CTK()
template <>
struct cuda_status_traits<::cudaError_t> : cuda_status_defaults<::cudaError_t>
{
  [[nodiscard]] _CCCL_HOST_API static const char* text(const ::cudaError_t __status)
  {
    return ::cuda::__driver::__getErrorString(__status);
  }
};

template <>
struct cuda_status_traits<::CUresult> : cuda_status_defaults<::CUresult>
{
  [[nodiscard]] _CCCL_HOST_API static const char* text(const ::CUresult __status)
  {
    return ::cuda::__driver::__getErrorString(__status);
  }
};
#  endif // _CCCL_HAS_CTK()

namespace __detail
{
inline constexpr long long __cuda_error_unknown = 999; // ::cudaErrorUnknown, spelled out so the header needs no CTK

[[nodiscard]] _CCCL_HOST_API inline char* __format_cuda_error(
  ::cuda::__msg_storage& __msg_buffer,
  const ::cuda::std::source_location& __loc,
  const char* __api,
  const ::cuda::std::string_view __status_type,
  const char* __text,
  const long long __raw_code,
  const char* __msg) noexcept
{
  // file:line api status_type(code): text: msg   -- the `api ` and `text: ` parts appear only when known
  ::snprintf(
    __msg_buffer.__buffer,
    ::cuda::__msg_storage::__size,
    "%s:%d %s%s%.*s(%lld): %s%s%s",
    __loc.file_name(),
    __loc.line(),
    __api ? __api : "",
    __api ? " " : "",
    static_cast<int>(__status_type.size()),
    __status_type.data(),
    __raw_code,
    __text ? __text : "",
    __text ? ": " : "",
    __msg);
  return __msg_buffer.__buffer;
}
} // namespace __detail

/**
 * @brief Exception thrown when a CUDA error is encountered.
 *
 * The exception carries the failing status object itself (`status<Status>()`, with `holds<Status>()` and
 * `status_type()` to ask what it is), its code as the API reported it (`raw_code()`), and where it was
 * raised (`location()`). Any status enumeration can be thrown; a struct status up to sixteen trivially
 * copyable bytes as well, see @ref cuda_status_traits. `status()` keeps its historical meaning: the
 * status seen as a CUDA Runtime error code.
 */
class cuda_error : public ::std::runtime_error
{
  static constexpr ::cuda::std::size_t __status_capacity = 16;

  long long __raw_code_;
  ::cuda::std::__type_info_ptr __type_;
  ::cuda::std::string_view __status_type_;
  ::cuda::std::source_location __loc_;
  alignas(8) unsigned char __status_bytes_[__status_capacity] = {}; // the status object, byte for byte

  template <class _Status>
  _CCCL_HOST_API void __store(const _Status& __status) noexcept
  {
    static_assert(::cuda::std::is_trivially_copyable_v<_Status>,
                  "cuda_error: a status type must be trivially copyable");
    static_assert(sizeof(_Status) <= __status_capacity, "cuda_error: a status type must fit in sixteen bytes");
    ::cuda::std::memcpy(__status_bytes_, &__status, sizeof(_Status));
  }

  _CCCL_HOST_API cuda_error(
    const long long __raw_code,
    const ::cuda::std::__type_info_ptr __type,
    const ::cuda::std::string_view __status_type,
    const char* __text,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc,
    __msg_storage __msg_buffer = {})
      : ::std::runtime_error(
          ::cuda::__detail::__format_cuda_error(__msg_buffer, __loc, __api, __status_type, __text, __raw_code, __msg))
      , __raw_code_(__raw_code)
      , __type_(__type)
      , __status_type_(__status_type)
      , __loc_(__loc)
  {}

  template <class _Status>
  [[nodiscard]] _CCCL_HOST_API static ::cuda::std::string_view __name_of() noexcept
  {
    const auto __pretty = ::cuda::std::__pretty_nameof<_Status>();
    return ::cuda::std::string_view(__pretty.data(), __pretty.size());
  }

  // A runtime status with a caller-supplied text: `__throw_cuda_error<_Error>` uses it where the driver may
  // not be loadable yet.
  _CCCL_HOST_API cuda_error(
    const __cuda_error_t __status,
    const char* __text,
    const char* __msg,
    const char* __api,
    const ::cuda::std::source_location& __loc)
      : cuda_error{static_cast<long long>(__status),
                   &_CCCL_TYPEID(__cuda_error_t),
                   __name_of<__cuda_error_t>(),
                   __text,
                   __msg,
                   __api,
                   __loc}
  {
    __store(__status);
  }

public:
  //! @brief Constructs from a status of any type with usable @ref cuda_status_traits: `cudaError_t`,
  //! `CUresult`, any other CUDA library's status enumeration, or a struct status with a specialization.
  template <class _Status>
  _CCCL_HOST_API cuda_error(const _Status __status,
                            const char* __msg,
                            const char* __api                         = nullptr,
                            const ::cuda::std::source_location& __loc = ::cuda::std::source_location::current())
      : cuda_error{cuda_status_traits<_Status>::raw_code(__status),
                   &_CCCL_TYPEID(_Status),
                   __name_of<_Status>(),
                   cuda_status_traits<_Status>::text(__status),
                   __msg,
                   __api,
                   __loc}
  {
    __store(__status);
  }

  //! @brief The status seen as a CUDA Runtime error code. Exact for a `cudaError_t`. A `CUresult` is converted
  //! numerically, which is how the runtime reports driver-originated failures. Any other status type reports
  //! `cudaErrorUnknown`; use `status<Status>()` or `raw_code()` for the exact value.
  [[nodiscard]] _CCCL_HOST_API __cuda_error_t status() const noexcept
  {
#  if _CCCL_HAS_CTK()
    const bool __cuda_family = holds<::cudaError_t>() || holds<::CUresult>();
#  else // ^^^ _CCCL_HAS_CTK() ^^^ / vvv !_CCCL_HAS_CTK() vvv
    const bool __cuda_family = holds<int>();
#  endif // ^^^ !_CCCL_HAS_CTK() ^^^
    return static_cast<__cuda_error_t>(__cuda_family ? __raw_code_ : __detail::__cuda_error_unknown);
  }

  //! @brief Whether the stored status came from a value of type `_Status`.
  template <class _Status>
  [[nodiscard]] _CCCL_HOST_API bool holds() const noexcept
  {
    return *__type_ == _CCCL_TYPEID(_Status);
  }

  //! @brief The stored status object, exactly as it was passed in. Precondition: `holds<_Status>()`.
  template <class _Status>
  [[nodiscard]] _CCCL_HOST_API _Status status() const noexcept
  {
    _CCCL_VERIFY(holds<_Status>(), "cuda_error::status<Status>(): the stored status is of another type");
    _Status __status{};
    ::cuda::std::memcpy(&__status, __status_bytes_, sizeof(_Status));
    return __status;
  }

  //! @brief The status code as the API reported it, through `cuda_status_traits<Status>::raw_code`.
  [[nodiscard]] _CCCL_HOST_API constexpr long long raw_code() const noexcept
  {
    return __raw_code_;
  }

  //! @brief The name of the type the status came from, e.g. "CUresult".
  [[nodiscard]] _CCCL_HOST_API constexpr ::cuda::std::string_view status_type() const noexcept
  {
    return __status_type_;
  }

  //! @brief Where the error was raised.
  [[nodiscard]] _CCCL_HOST_API constexpr const ::cuda::std::source_location& location() const noexcept
  {
    return __loc_;
  }

  template <int _Error>
  [[noreturn]] friend _CCCL_HOST_API void
  __throw_cuda_error(const char* __msg, const char* __api, const ::cuda::std::source_location& __loc)
  {
    [[maybe_unused]] const char* __error_str{};
    if constexpr (_Error == /*::cudaErrorInvalidValue*/ 1)
    {
      __error_str = "invalid value";
    }
    else if constexpr (_Error == /*::cudaErrorInitializationError*/ 3)
    {
      __error_str = "initialization error";
    }
    else if constexpr (_Error == /*::cudaErrorNotSupported*/ 801)
    {
      __error_str = "operation not supported";
    }
    else if constexpr (_Error == /*::cudaErrorUnknown*/ 999)
    {
      __error_str = "unknown error";
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<decltype(_Error)>, "unknown _Error");
    }
    _CCCL_THROW(::cuda::cuda_error, static_cast<__cuda_error_t>(_Error), __error_str, __msg, __api, __loc);
  }

  [[noreturn]] friend _CCCL_HOST_API void
  __throw_cuda_error(int __error, const char* __msg, const char* __api, const ::cuda::std::source_location& __loc)
  {
    _CCCL_THROW(::cuda::cuda_error, static_cast<__cuda_error_t>(__error), __msg, __api, __loc);
  }
};
#endif // _CCCL_HOSTED()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_STD___EXCEPTION_CUDA_ERROR_H
