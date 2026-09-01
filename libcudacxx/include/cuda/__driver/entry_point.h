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
#  include <cuda/std/source_location>

#  if _CCCL_HOSTED()
#    if _CCCL_OS(WINDOWS)
#      include <windows.h>
#    else
#      include <dlfcn.h>
#    endif
#  endif // _CCCL_HOSTED()

#  include <cuda.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// We can't use ordinary cuda_error's constructor here, because it uses cuGetErrorString to get the error string which
// might be unavailable if we fail to dlopen the driver or query the cuGetProcAddress function. Instead, we hardcode
// several known error strings that are used in these APIs.
template <::cudaError_t _Error>
[[noreturn]] _CCCL_HOST_API void __throw_cuda_error(
  const char* __msg,
  const char* __api                         = nullptr,
  const ::cuda::std::source_location& __loc = ::cuda::std::source_location::current());

_CCCL_END_NAMESPACE_CUDA

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

#  if _CCCL_HOSTED()
//! @brief Gets the cuGetProcAddress function pointer.
template <class = void>
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline auto __getProcAddressFn() -> decltype(cuGetProcAddress)*
{
  const char* __fn_name = "cuGetProcAddress_v2";
#    if _CCCL_OS(WINDOWS)
  static auto __driver_library = ::LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (__driver_library == nullptr)
  {
    ::cuda::__throw_cuda_error<::cudaErrorUnknown>("Failed to load nvcuda.dll");
  }
  static void* __fn = ::GetProcAddress(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    ::cuda::__throw_cuda_error<::cudaErrorInitializationError>("Failed to get cuGetProcAddress from nvcuda.dll");
  }
#    else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
#      if _CCCL_OS(ANDROID)
  const char* __driver_library_name = "libcuda.so";
#      else // ^^^ _CCCL_OS(ANDROID) ^^^ / vvv !_CCCL_OS(ANDROID) vvv
  const char* __driver_library_name = "libcuda.so.1";
#      endif // ^^^ !_CCCL_OS(ANDROID) ^^^
  static void* __driver_library = ::dlopen(__driver_library_name, RTLD_NOW);
  if (__driver_library == nullptr)
  {
    ::cuda::__throw_cuda_error<::cudaErrorUnknown>("Failed to load libcuda.so.1");
  }
  static void* __fn = ::dlsym(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    ::cuda::__throw_cuda_error<::cudaErrorInitializationError>("Failed to get cuGetProcAddress from libcuda.so.1");
  }
#    endif // ^^^ !_CCCL_OS(WINDOWS) ^^^
  return reinterpret_cast<decltype(cuGetProcAddress)*>(__fn);
}
#  else // ^^^ _CCCL_HOSTED() ^^^ / vvv !_CCCL_HOSTED() vvv
[[nodiscard]]
_CCCL_PUBLIC_HOST_API inline auto
__getProcAddressFn(decltype(cuGetProcAddress)* __ptr = nullptr, bool __set = false) noexcept
  -> decltype(cuGetProcAddress)*
{
  static decltype(cuGetProcAddress)* __fn = __ptr;

  if (__set)
  {
    __fn = __ptr;
  }

  return __fn;
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
//! @param __name Name of the symbol to get the driver entry point for.
//! @param __major The major CTK version to get the symbol version for. Defaults to 12.
//! @param __minor The major CTK version to get the symbol version for. Defaults to 0.
//!
//! @return The address of the symbol.
//!
//! @throws @c cuda::cuda_error if the symbol cannot be obtained.
template <class = void>
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline void* __get_driver_entry_point_no_init(
  const char* __name, [[maybe_unused]] int __major = 12, [[maybe_unused]] int __minor = 0)
{
  static auto __get_proc_addr_fn = ::cuda::__driver::__getProcAddressFn();

  void* __fn;
  ::CUdriverProcAddressQueryResult __result;
  ::CUresult __status = __get_proc_addr_fn(
    __name, &__fn, ::cuda::__driver::__make_version(__major, __minor), ::CU_GET_PROC_ADDRESS_DEFAULT, &__result);
  if (__status != ::CUDA_SUCCESS || __result != ::CU_GET_PROC_ADDRESS_SUCCESS)
  {
    if (__status == ::CUDA_ERROR_INVALID_VALUE)
    {
      ::cuda::__throw_cuda_error<::cudaErrorInvalidValue>("Driver version is too low to use this API", __name);
    }
    if (__result == ::CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT)
    {
      ::cuda::__throw_cuda_error<::cudaErrorNotSupported>("Driver does not support this API", __name);
    }
    ::cuda::__throw_cuda_error<::cudaErrorUnknown>("Failed to access driver API", __name);
  }
  return __fn;
}

[[nodiscard]] _CCCL_HOST_API inline const char* __getErrorString(::cudaError_t __error)
{
  // cuGetErrorString doesn't require the driver to be initialized.
  static const auto __driver_fn = reinterpret_cast<decltype(::cuGetErrorString)*>(
    ::cuda::__driver::__get_driver_entry_point_no_init("cuGetErrorString"));

  // We can emulate the cudaGetErrorString behaviour by falling back to the "unrecognized error code" string.
  const char* __ret{};
  (void) __driver_fn(static_cast<::CUresult>(__error), &__ret);
  return (__ret != nullptr) ? __ret : "unrecognized error code";
}

// Forward declare __init, so we can use it in __get_driver_entry_point.
[[nodiscard]] _CCCL_HOST_API inline bool __init();

//! @brief Get a driver function pointer for a given API name and optionally specific CUDA version. This function also
//!        initializes the CUDA Driver.
//!
//! @param __name Name of the symbol to get the driver entry point for.
//! @param __major The major CTK version to get the symbol version for. Defaults to 12.
//! @param __minor The major CTK version to get the symbol version for. Defaults to 0.
//!
//! @return The address of the symbol.
//!
//! @throws @c cuda::cuda_error if the symbol cannot be obtained or the CUDA driver failed to initialize.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline void*
__get_driver_entry_point(const char* __name, [[maybe_unused]] int __major = 12, [[maybe_unused]] int __minor = 0)
{
  [[maybe_unused]] static auto __init = ::cuda::__driver::__init();
  return ::cuda::__driver::__get_driver_entry_point_no_init(__name, __major, __minor);
}

// Get the driver function by name using this macro
#  define _CCCLRT_GET_DRIVER_FUNCTION(function_name) \
    reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name))

#  define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
    reinterpret_cast<decltype(::versioned_fn_name)*>(                                           \
      ::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DRIVER_ENTRY_POINT_H
