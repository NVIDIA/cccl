//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DRIVER_ENTRY_POINT_H
#define _CUDA___DRIVER_ENTRY_POINT_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)
#  if _CCCL_HOSTED()
#    if _CCCL_OS(WINDOWS)
#      include <windows.h>
#    else
#      include <dlfcn.h>
#    endif
#  endif // _CCCL_HOSTED()

#  include <cuda.h>
#  include <driver_types.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

enum class __driver_error_source
{
  __success,
  __driver_load,
  __entry_point_lookup,
  __driver_init,
  __api_call
};

struct __driver_status
{
  ::CUresult __status_;
  __driver_error_source __source_;
  const char* __api_;
  const char* __message_;

  [[nodiscard]] _CCCL_HOST_API constexpr operator ::cudaError_t() const noexcept
  {
    return static_cast<::cudaError_t>(__status_);
  }

  [[nodiscard]] _CCCL_HOST_API constexpr bool __ok() const noexcept
  {
    return __status_ == ::CUDA_SUCCESS;
  }
};

struct __driver_entry_point_result
{
  void* __fn_;
  __driver_status __status_;
};

template <class _FnPtr>
struct __driver_function_result
{
  _FnPtr __fn_;
  __driver_status __status_;

  _CCCL_HOST_API constexpr __driver_function_result(_FnPtr __fn, __driver_status __status) noexcept
      : __fn_(__fn)
      , __status_(__status)
  {}

  _CCCL_HOST_API explicit __driver_function_result(__driver_entry_point_result __result) noexcept
      : __fn_(reinterpret_cast<_FnPtr>(__result.__fn_))
      , __status_(__result.__status_)
  {}
};

// _FN must be a stable cached result: it is referenced more than once. Arguments are evaluated only on success.
#  define _CCCLRT_CALL_DRIVER_FUNCTION(_FN, ...)                                                                      \
    ((_FN).__status_.__ok() ? ::cuda::__driver::__driver_api_status((_FN).__fn_(__VA_ARGS__), (_FN).__status_.__api_) \
                            : (_FN).__status_)

// Only the function-pointer cast is typed; lookup and its diagnostics remain non-template implementations.
#  define _CCCLRT_GET_DRIVER_FUNCTION_TYPED(_FnPtr, ...)      \
    ::cuda::__driver::__driver_function_result<_FnPtr>        \
    {                                                         \
      ::cuda::__driver::__get_driver_entry_point(__VA_ARGS__) \
    }

#  define _CCCLRT_GET_DRIVER_FUNCTION_TYPED_NO_INIT(_FnPtr, ...)      \
    ::cuda::__driver::__driver_function_result<_FnPtr>                \
    {                                                                 \
      ::cuda::__driver::__get_driver_entry_point_no_init(__VA_ARGS__) \
    }

[[nodiscard]] _CCCL_HOST_API constexpr __driver_status __driver_success(const char* __api) noexcept
{
  return {::CUDA_SUCCESS, __driver_error_source::__success, __api, nullptr};
}

[[nodiscard]] _CCCL_HOST_API constexpr __driver_status
__driver_api_status(::CUresult __status, const char* __api) noexcept
{
  return {__status, __driver_error_source::__api_call, __api, nullptr};
}

[[nodiscard]] _CCCL_HOST_API constexpr __driver_status
__driver_api_status(__driver_status __status, const char*) noexcept
{
  return __status;
}

#  if _CCCL_HOSTED()

//! @brief Gets the cuGetProcAddress function pointer without throwing.
//!
//! @return The function pointer and driver-loading status.
//! @note Library and symbol loading failures are cached and are not retried.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline __driver_function_result<decltype(&cuGetProcAddress)>
__getProcAddressFn() noexcept
{
  constexpr auto __fn_name = "cuGetProcAddress_v2";

#    if _CCCL_OS(WINDOWS)
  static const auto __driver_library = ::LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (__driver_library == nullptr)
  {
    return {nullptr,
            {::CUDA_ERROR_UNKNOWN, __driver_error_source::__driver_load, nullptr, "Failed to load nvcuda.dll"}};
  }
  static const auto __fn = ::GetProcAddress(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    return {nullptr,
            {::CUDA_ERROR_NOT_INITIALIZED,
             __driver_error_source::__driver_load,
             __fn_name,
             "Failed to get cuGetProcAddress from nvcuda.dll"}};
  }
#    else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
  constexpr auto __driver_library_name = _CCCL_OS(ANDROID) ? "libcuda.so" : "libcuda.so.1";
  static const auto __driver_library   = ::dlopen(__driver_library_name, RTLD_NOW);
  if (__driver_library == nullptr)
  {
    return {nullptr,
            {::CUDA_ERROR_UNKNOWN, __driver_error_source::__driver_load, nullptr, "Failed to load libcuda.so.1"}};
  }
  static const auto __fn = ::dlsym(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    return {nullptr,
            {::CUDA_ERROR_NOT_INITIALIZED,
             __driver_error_source::__driver_load,
             __fn_name,
             "Failed to get cuGetProcAddress from libcuda.so.1"}};
  }
#    endif // ^^^ !_CCCL_OS(WINDOWS) ^^^

  return {reinterpret_cast<decltype(&cuGetProcAddress)>(__fn), ::cuda::__driver::__driver_success(__fn_name)};
}

#  else // ^^^ _CCCL_HOSTED() ^^^ / vvv !_CCCL_HOSTED() vvv

[[nodiscard]]
_CCCL_PUBLIC_HOST_API inline __driver_function_result<decltype(&cuGetProcAddress)>
__getProcAddressFn(decltype(cuGetProcAddress)* __ptr = nullptr, bool __set = false) noexcept
{
  static decltype(cuGetProcAddress)* __fn = __ptr;

  if (__set)
  {
    __fn = __ptr;
  }

  if (__fn == nullptr)
  {
    return {nullptr,
            {::CUDA_ERROR_NOT_INITIALIZED,
             __driver_error_source::__driver_load,
             "cuGetProcAddress_v2",
             "Failed to get cuGetProcAddress"}};
  }
  return {__fn, ::cuda::__driver::__driver_success("cuGetProcAddress_v2")};
}

#  endif // !_CCCL_HOSTED()

//! @brief Makes the driver version from major and minor version.
[[nodiscard]] _CCCL_HOST_API constexpr int __make_version(int __major, int __minor) noexcept
{
  _CCCL_ASSERT(__major >= 2, "invalid major CUDA Driver version");
  _CCCL_ASSERT(__minor >= 0 && __minor < 100, "invalid minor CUDA Driver version");
  return __major * 1000 + __minor * 10;
}

//! @brief Get a driver function pointer for a given API name and optionally specific CUDA version without initializing
//!        the CUDA driver.
//!
//! @param[in] __name Name of the symbol to get the driver entry point for.
//! @param[in] __major The major CTK version to get the symbol version for. Defaults to 12.
//! @param[in] __minor The minor CTK version to get the symbol version for. Defaults to 0.
//!
//! @return The address of the symbol and lookup status.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline __driver_entry_point_result __get_driver_entry_point_no_init(
  const char* __name,
  int __major = 12,
  int __minor = 0) noexcept // NOLINT(bugprone-exception-escape)
{
  const auto __get_proc_addr_fn = ::cuda::__driver::__getProcAddressFn();
  if (!__get_proc_addr_fn.__status_.__ok())
  {
    return {nullptr, __get_proc_addr_fn.__status_};
  }

  void* __fn{};
  ::CUdriverProcAddressQueryResult __result{};
  const ::CUresult __status = __get_proc_addr_fn.__fn_(
    __name, &__fn, ::cuda::__driver::__make_version(__major, __minor), ::CU_GET_PROC_ADDRESS_DEFAULT, &__result);
  if (__status == ::CUDA_SUCCESS && __result == ::CU_GET_PROC_ADDRESS_SUCCESS)
  {
    return {__fn, ::cuda::__driver::__driver_success(__name)};
  }

  if (__status == ::CUDA_ERROR_INVALID_VALUE)
  {
    return {nullptr,
            {::CUDA_ERROR_INVALID_VALUE,
             __driver_error_source::__entry_point_lookup,
             __name,
             "Driver version is too low to use this API"}};
  }
  if (__result == ::CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT)
  {
    return {nullptr,
            {::CUDA_ERROR_NOT_SUPPORTED,
             __driver_error_source::__entry_point_lookup,
             __name,
             "Driver does not support this API"}};
  }
  return {nullptr,
          {::CUDA_ERROR_UNKNOWN, __driver_error_source::__entry_point_lookup, __name, "Failed to access driver API"}};
}

[[nodiscard]] _CCCL_HOST_API inline const char* __getErrorString(::cudaError_t __error) noexcept
{
  // cuGetErrorString doesn't require the driver to be initialized.
  static const auto __driver_fn =
    _CCCLRT_GET_DRIVER_FUNCTION_TYPED_NO_INIT(decltype(&::cuGetErrorString), "cuGetErrorString");

  // Error formatting must not replace the original error if lookup or cuGetErrorString fails.
  const char* __ret{};
  (void) _CCCLRT_CALL_DRIVER_FUNCTION(__driver_fn, static_cast<::CUresult>(__error), &__ret);
  return (__ret != nullptr) ? __ret : "unrecognized error code";
}

//! @brief Initializes the CUDA Driver.
//!
//! @return Initialization status, including lookup failure provenance.
[[nodiscard]] _CCCL_HOST_API inline __driver_status __init() noexcept // NOLINT(bugprone-exception-escape)
{
  constexpr auto __symbol_name = "cuInit";
  const auto __driver_fn       = _CCCLRT_GET_DRIVER_FUNCTION_TYPED_NO_INIT(decltype(&::cuInit), __symbol_name);
  if (!__driver_fn.__status_.__ok())
  {
    return __driver_fn.__status_;
  }
  const auto __status = __driver_fn.__fn_(0);
  if (__status != ::CUDA_SUCCESS)
  {
    return {__status, __driver_error_source::__driver_init, __symbol_name, "Failed to initialize CUDA Driver"};
  }
  return ::cuda::__driver::__driver_success(__symbol_name);
}

//! @brief Gets a driver entry point after initializing the CUDA Driver, without throwing.
//!
//! @param[in] __name Name of the symbol to get the driver entry point for.
//! @param[in] __major The major CTK version to get the symbol version for. Defaults to 12.
//! @param[in] __minor The minor CTK version to get the symbol version for. Defaults to 0.
//! @return The address of the symbol and initialization or lookup status.
//! @note Driver initialization status is cached, including failures; initialization is not retried.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline __driver_entry_point_result
__get_driver_entry_point(const char* __name, int __major = 12, int __minor = 0) noexcept
{
  static const auto __init_status = ::cuda::__driver::__init();
  if (!__init_status.__ok())
  {
    return {nullptr, __init_status};
  }
  return ::cuda::__driver::__get_driver_entry_point_no_init(__name, __major, __minor);
}

// Lookup and invocation both return status; throwing is the wrapper's responsibility.
#  define _CCCLRT_GET_DRIVER_FUNCTION(function_name) \
    _CCCLRT_GET_DRIVER_FUNCTION_TYPED(decltype(&::function_name), #function_name)

#  define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
    _CCCLRT_GET_DRIVER_FUNCTION_TYPED(decltype(&::versioned_fn_name), #function_name, major, minor)

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DRIVER_ENTRY_POINT_H
