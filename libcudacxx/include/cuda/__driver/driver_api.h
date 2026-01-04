//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___DRIVER_DRIVER_API_H
#define _CUDA___DRIVER_DRIVER_API_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__internal/namespaces.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_same.h>
#  if _CCCL_OS(WINDOWS)
#    include <windows.h>
#  else
#    include <dlfcn.h>
#  endif

#  include <stdexcept>

#  include <cuda.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

// Get the driver function by name using this macro
#  define _CCCLRT_GET_DRIVER_FUNCTION(function_name) \
    reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name))

#  define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
    reinterpret_cast<decltype(::versioned_fn_name)*>(                                           \
      ::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

// cudaGetDriverEntryPoint function is deprecated
_CCCL_SUPPRESS_DEPRECATED_PUSH

//! @brief Gets the cuGetProcAddress function pointer.
[[nodiscard]] _CCCL_PUBLIC_HOST_API inline auto __getProcAddressFn() -> decltype(cuGetProcAddress)*
{
  const char* __fn_name = "cuGetProcAddress_v2";
#  if _CCCL_OS(WINDOWS)
  static auto __driver_library = ::LoadLibraryExA("nvcuda.dll", nullptr, LOAD_LIBRARY_SEARCH_SYSTEM32);
  if (__driver_library == nullptr)
  {
    ::cuda::__throw_cuda_error(::cudaErrorUnknown, "Failed to load nvcuda.dll");
  }
  static void* __fn = ::GetProcAddress(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    ::cuda::__throw_cuda_error(::cudaErrorInitializationError, "Failed to get cuGetProcAddress from nvcuda.dll");
  }
#  else // ^^^ _CCCL_OS(WINDOWS) ^^^ / vvv !_CCCL_OS(WINDOWS) vvv
#    if _CCCL_OS(ANDROID)
  const char* __driver_library_name = "libcuda.so";
#    else // ^^^ _CCCL_OS(ANDROID) ^^^ / vvv !_CCCL_OS(ANDROID) vvv
  const char* __driver_library_name = "libcuda.so.1";
#    endif // ^^^ !_CCCL_OS(ANDROID) ^^^
  static void* __driver_library = ::dlopen(__driver_library_name, RTLD_NOW);
  if (__driver_library == nullptr)
  {
    ::cuda::__throw_cuda_error(::cudaErrorUnknown, "Failed to load libcuda.so.1");
  }
  static void* __fn = ::dlsym(__driver_library, __fn_name);
  if (__fn == nullptr)
  {
    ::cuda::__throw_cuda_error(::cudaErrorInitializationError, "Failed to get cuGetProcAddress from libcuda.so.1");
  }
#  endif // ^^^ !_CCCL_OS(WINDOWS) ^^^
  return reinterpret_cast<decltype(cuGetProcAddress)*>(__fn);
}

_CCCL_SUPPRESS_DEPRECATED_POP

//! @brief Makes the driver version from major and minor version.
[[nodiscard]] _CCCL_HOST_API constexpr int __make_version(int __major, int __minor) noexcept
{
  _CCCL_ASSERT(__major >= 12, "invalid major CUDA Driver version");
  _CCCL_ASSERT(__minor >= 0 && __minor < 100, "invalid minor CUDA Driver version");
  return __major * 1000 + __minor * 10;
}

//! @brief Gets the driver entry point.
//!
//! @param __get_proc_addr_fn Pointer to cuGetProcAddress function.
//! @param __name Name of the symbol to get the driver entry point for.
//! @param __major The major CTK version to get the symbol version for.
//! @param __minor The major CTK version to get the symbol version for.
//!
//! @return The address of the symbol.
//!
//! @throws @c cuda::cuda_error if the symbol cannot be obtained.
[[nodiscard]] _CCCL_HOST_API inline void* __get_driver_entry_point_impl(
  decltype(cuGetProcAddress)* __get_proc_addr_fn, const char* __name, int __major, int __minor)
{
  void* __fn;
  ::CUdriverProcAddressQueryResult __result;
  ::CUresult __status = __get_proc_addr_fn(
    __name, &__fn, ::cuda::__driver::__make_version(__major, __minor), ::CU_GET_PROC_ADDRESS_DEFAULT, &__result);
  if (__status != ::CUDA_SUCCESS || __result != ::CU_GET_PROC_ADDRESS_SUCCESS)
  {
    if (__status == ::CUDA_ERROR_INVALID_VALUE)
    {
      ::cuda::__throw_cuda_error(::cudaErrorInvalidValue, "Driver version is too low to use this API", __name);
    }
    if (__result == ::CU_GET_PROC_ADDRESS_VERSION_NOT_SUFFICIENT)
    {
      ::cuda::__throw_cuda_error(::cudaErrorNotSupported, "Driver does not support this API", __name);
    }
    else
    {
      ::cuda::__throw_cuda_error(::cudaErrorUnknown, "Failed to access driver API", __name);
    }
  }
  return __fn;
}

//! @brief CUDA Driver API call wrapper. Calls a given CUDA Driver API and checks the return value.
//!
//! @param __fn A CUDA Driver function.
//! @param __err_msg Error message describing the call if the all fails.
//! @param __args The arguments to the @c __fn call.
//!
//! @throws @c cuda::cuda_error if the function call doesn't return CUDA_SUCCESS.
template <typename Fn, typename... Args>
_CCCL_HOST_API inline void __call_driver_fn(Fn __fn, const char* __err_msg, Args... __args)
{
  ::CUresult __status = __fn(__args...);
  if (__status != ::CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<::cudaError_t>(__status), __err_msg);
  }
}

//! @brief Initializes the CUDA Driver.
//!
//! @param __get_proc_addr_fn The pointer to cuGetProcAddress function.
//!
//! @return A dummy bool value.
//!
//! @warning This function should be called only once from __get_driver_entry_point function.
[[nodiscard]] _CCCL_HOST_API inline bool __init(decltype(cuGetProcAddress)* __get_proc_addr_fn)
{
  auto __driver_fn = reinterpret_cast<decltype(::cuInit)*>(
    ::cuda::__driver::__get_driver_entry_point_impl(__get_proc_addr_fn, "cuInit", 12, 0));
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to initialize CUDA Driver", 0);
  return true;
}

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
  // Get cuGetProcAddress function and call cuInit(0) only on the first call
  static auto __get_proc_addr_fn      = ::cuda::__driver::__getProcAddressFn();
  [[maybe_unused]] static auto __init = ::cuda::__driver::__init(__get_proc_addr_fn);
  return ::cuda::__driver::__get_driver_entry_point_impl(__get_proc_addr_fn, __name, __major, __minor);
}

//! @brief Converts CUdevice to ordinal device id.
//!
//! @note Currently, CUdevice value is the same as the ordinal device id. But that might change in the future.
[[nodiscard]] _CCCL_HOST_API inline int __cudevice_to_ordinal(::CUdevice __dev) noexcept
{
  return static_cast<int>(__dev);
}

// Version management

[[nodiscard]] _CCCL_HOST_API inline int __getVersion()
{
  static int __version = []() {
    int __v;
    auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDriverGetVersion);
    ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to check CUDA driver version", &__v);
    return __v;
  }();
  return __version;
}

[[nodiscard]] _CCCL_HOST_API inline bool __version_at_least(int __major, int __minor)
{
  return ::cuda::__driver::__getVersion() >= ::cuda::__driver::__make_version(__major, __minor);
}

[[nodiscard]] _CCCL_HOST_API inline bool __version_below(int __major, int __minor)
{
  return ::cuda::__driver::__getVersion() < ::cuda::__driver::__make_version(__major, __minor);
}

// Device management

[[nodiscard]] _CCCL_HOST_API inline ::CUdevice __deviceGet(int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGet);
  ::CUdevice __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device", &__result, __ordinal);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::CUdevice __deviceGetAttribute(::CUdevice_attribute __attr, ::CUdevice __device)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetAttribute);
  int __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device attribute", &__result, __attr, __device);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline int __deviceGetCount()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetCount);
  int __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device count", &__result);
  return __result;
}

_CCCL_HOST_API inline void __deviceGetName(char* __name_out, int __len, int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetName);

  // TODO CUdevice is just an int, we probably could just cast, but for now do the safe thing
  ::CUdevice __dev = __deviceGet(__ordinal);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to query the name of a device", __name_out, __len, __dev);
}

// Primary context management

[[nodiscard]] _CCCL_HOST_API inline ::CUcontext __primaryCtxRetain(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRetain);
  ::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to retain context for a device", &__result, __dev);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __primaryCtxReleaseNoThrow(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRelease);
  return static_cast<::cudaError_t>(__driver_fn(__dev));
}

[[nodiscard]] _CCCL_HOST_API inline bool __isPrimaryCtxActive(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxGetState);
  int __result;
  unsigned int __dummy;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to check the primary ctx state", __dev, &__dummy, &__result);
  return __result == 1;
}

// Context management

_CCCL_HOST_API inline void __ctxPush(::CUcontext __ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to push context", __ctx);
}

_CCCL_HOST_API inline ::CUcontext __ctxPop()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  ::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to pop context", &__result);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::CUcontext __ctxGetCurrent()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetCurrent);
  ::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get current context", &__result);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::CUdevice __ctxGetDevice()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetDevice);
  ::CUdevice __result{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get current context", &__result);
  return __result;
}

// Memory management

_CCCL_HOST_API inline void
__memcpyAsync(void* __dst, const void* __src, ::cuda::std::size_t __count, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemcpyAsync);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to perform a memcpy",
    reinterpret_cast<::CUdeviceptr>(__dst),
    reinterpret_cast<::CUdeviceptr>(__src),
    __count,
    __stream);
}

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline void __memcpyAsyncWithAttributes(
  void* __dst, const void* __src, ::cuda::std::size_t __count, ::CUstream __stream, ::CUmemcpyAttributes __attributes)
{
  static auto __driver_fn    = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemcpyBatchAsync, cuMemcpyBatchAsync, 13, 0);
  ::cuda::std::size_t __zero = 0;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to perform a memcpy with attributes",
    reinterpret_cast<::CUdeviceptr*>(&__dst),
    reinterpret_cast<::CUdeviceptr*>(&__src),
    &__count,
    1,
    &__attributes,
    &__zero,
    1,
    __stream);
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <typename _Tp>
_CCCL_HOST_API void __memsetAsync(void* __dst, _Tp __value, ::cuda::std::size_t __count, ::CUstream __stream)
{
  if constexpr (sizeof(_Tp) == 1)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD8Async);
    ::cuda::__driver::__call_driver_fn(
      __driver_fn, "Failed to perform a memset", reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream);
  }
  else if constexpr (sizeof(_Tp) == 2)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD16Async);
    ::cuda::__driver::__call_driver_fn(
      __driver_fn, "Failed to perform a memset", reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream);
  }
  else if constexpr (sizeof(_Tp) == 4)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD32Async);
    ::cuda::__driver::__call_driver_fn(
      __driver_fn, "Failed to perform a memset", reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream);
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "Unsupported type for memset");
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__mempoolCreateNoThrow(::CUmemoryPool* __pool, ::CUmemPoolProps* __props)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolCreate);
  return static_cast<::cudaError_t>(__driver_fn(__pool, __props));
}

_CCCL_HOST_API inline void __mempoolSetAttribute(::CUmemoryPool __pool, ::CUmemPool_attribute __attr, void* __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolSetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to set attribute for a memory pool", __pool, __attr, __value);
}

_CCCL_HOST_API inline ::cuda::std::size_t __mempoolGetAttribute(::CUmemoryPool __pool, ::CUmemPool_attribute __attr)
{
  ::cuda::std::size_t __value = 0;
  static auto __driver_fn     = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get attribute for a memory pool", __pool, __attr, &__value);
  return __value;
}

_CCCL_HOST_API inline void __mempoolDestroy(::CUmemoryPool __pool)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolDestroy);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to destroy a memory pool", __pool);
}

_CCCL_HOST_API inline ::CUdeviceptr
__mallocFromPoolAsync(::cuda::std::size_t __bytes, ::CUmemoryPool __pool, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocFromPoolAsync);
  ::CUdeviceptr __result  = 0;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to allocate memory from a memory pool", &__result, __bytes, __pool, __stream);
  return __result;
}

_CCCL_HOST_API inline void __mempoolTrimTo(::CUmemoryPool __pool, ::cuda::std::size_t __min_bytes_to_keep)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolTrimTo);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to trim a memory pool", __pool, __min_bytes_to_keep);
}

_CCCL_HOST_API inline ::cudaError_t __freeAsyncNoThrow(::CUdeviceptr __dptr, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFreeAsync);
  return static_cast<::cudaError_t>(__driver_fn(__dptr, __stream));
}

_CCCL_HOST_API inline void
__mempoolSetAccess(::CUmemoryPool __pool, ::CUmemAccessDesc* __descs, ::cuda::std::size_t __count)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolSetAccess);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to set access of a memory pool", __pool, __descs, __count);
}

_CCCL_HOST_API inline ::CUmemAccess_flags __mempoolGetAccess(::CUmemoryPool __pool, ::CUmemLocation* __location)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAccess);
  ::CUmemAccess_flags __flags;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get access of a memory pool", &__flags, __pool, __location);
  return __flags;
}

_CCCL_HOST_API inline ::cudaError_t
__mempoolGetAccessNoThrow(::CUmemAccess_flags& __flags, ::CUmemoryPool __pool, ::CUmemLocation* __location) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAccess);
  return static_cast<::cudaError_t>(__driver_fn(&__flags, __pool, __location));
}

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline ::CUmemoryPool
__getDefaultMemPool(CUmemLocation __location, CUmemAllocationType_enum __allocation_type)
{
  static auto __driver_fn =
    _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemGetDefaultMemPool, cuMemGetDefaultMemPool, 13, 0);
  ::CUmemoryPool __result = nullptr;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to get default memory pool", &__result, &__location, __allocation_type);
  return __result;
}
#  else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
_CCCL_HOST_API inline ::CUmemoryPool __deviceGetDefaultMemPool(::CUdevice __device)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetDefaultMemPool);
  ::CUmemoryPool __result = nullptr;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get default memory pool", &__result, __device);
  return __result;
}
#  endif // ^^^ _CCCL_CTK_BELOW(13, 0) ^^^

_CCCL_HOST_API inline ::CUdeviceptr __mallocManaged(::cuda::std::size_t __bytes, unsigned int __flags)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocManaged);
  ::CUdeviceptr __result  = 0;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to allocate managed memory", &__result, __bytes, __flags);
  return __result;
}

_CCCL_HOST_API inline void* __mallocHost(::cuda::std::size_t __bytes)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocHost);
  void* __result          = nullptr;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to allocate host memory", &__result, __bytes);
  return __result;
}

_CCCL_HOST_API inline ::cudaError_t __freeNoThrow(::CUdeviceptr __dptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFree);
  return static_cast<::cudaError_t>(__driver_fn(__dptr));
}

_CCCL_HOST_API inline ::cudaError_t __freeHostNoThrow(void* __dptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFreeHost);
  return static_cast<::cudaError_t>(__driver_fn(__dptr));
}

// Unified Addressing

// TODO: we don't want to have these functions here, refactoring expected
template <::CUpointer_attribute _Attr>
[[nodiscard]] _CCCL_API _CCCL_CONSTEVAL auto __pointer_attribute_value_type_t_impl() noexcept
{
  if constexpr (_Attr == ::CU_POINTER_ATTRIBUTE_CONTEXT)
  {
    return ::CUcontext{};
  }
  else if constexpr (_Attr == ::CU_POINTER_ATTRIBUTE_MEMORY_TYPE)
  {
    return ::CUmemorytype{};
  }
  else if constexpr (_Attr == ::CU_POINTER_ATTRIBUTE_DEVICE_POINTER || _Attr == ::CU_POINTER_ATTRIBUTE_HOST_POINTER)
  {
    return static_cast<void*>(nullptr);
  }
  else if constexpr (_Attr == ::CU_POINTER_ATTRIBUTE_IS_MANAGED || _Attr == ::CU_POINTER_ATTRIBUTE_MAPPED)
  {
    return bool{};
  }
  else if constexpr (_Attr == ::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL)
  {
    return int{};
  }
  else if constexpr (_Attr == ::CU_POINTER_ATTRIBUTE_MEMPOOL_HANDLE)
  {
    return ::CUmemoryPool{};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<decltype(_Attr)>, "not implemented attribute");
  }
}

template <::CUpointer_attribute _Attr>
using __pointer_attribute_value_type_t = decltype(::cuda::__driver::__pointer_attribute_value_type_t_impl<_Attr>());

template <::CUpointer_attribute _Attr>
[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__pointerGetAttributeNoThrow(__pointer_attribute_value_type_t<_Attr>& __result, const void* __ptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuPointerGetAttribute);
  ::cudaError_t __status{};
  if constexpr (::cuda::std::is_same_v<__pointer_attribute_value_type_t<_Attr>, bool>)
  {
    int __result2{};
    __status = static_cast<::cudaError_t>(__driver_fn(&__result2, _Attr, reinterpret_cast<::CUdeviceptr>(__ptr)));
    __result = static_cast<bool>(__result2);
  }
  else
  {
    __status =
      static_cast<::cudaError_t>(__driver_fn((void*) &__result, _Attr, reinterpret_cast<::CUdeviceptr>(__ptr)));
  }
  return __status;
}

template <::cuda::std::size_t _Np>
[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__pointerGetAttributesNoThrow(::CUpointer_attribute (&__attrs)[_Np], void* (&__results)[_Np], const void* __ptr)
{
  static const auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuPointerGetAttributes);
  return static_cast<::cudaError_t>(
    __driver_fn(static_cast<unsigned>(_Np), __attrs, __results, reinterpret_cast<::CUdeviceptr>(__ptr)));
}

// Stream management

_CCCL_HOST_API inline void
__streamAddCallback(::CUstream __stream, ::CUstreamCallback __cb, void* __data, unsigned __flags = 0)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamAddCallback);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to add a stream callback", __stream, __cb, __data, __flags);
}

[[nodiscard]] _CCCL_HOST_API inline ::CUstream __streamCreateWithPriority(unsigned __flags, int __priority)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamCreateWithPriority);
  ::CUstream __stream;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to create a stream", &__stream, __flags, __priority);
  return __stream;
}

_CCCL_HOST_API inline ::cudaError_t __streamSynchronizeNoThrow(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamSynchronize);
  return static_cast<::cudaError_t>(__driver_fn(__stream));
}

_CCCL_HOST_API inline void __streamSynchronize(::CUstream __stream)
{
  cudaError_t __status = __streamSynchronizeNoThrow(__stream);
  if (__status != cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__status, "Failed to synchronize a stream");
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::CUcontext __streamGetCtx(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetCtx);
  ::CUcontext __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get context from a stream", __stream, &__result);
  return __result;
}

#  if _CCCL_CTK_AT_LEAST(12, 5)
struct __ctx_from_stream
{
  enum class __kind
  {
    __device,
    __green
  };

  __kind __ctx_kind_;
  union
  {
    ::CUcontext __ctx_device_;
    ::CUgreenCtx __ctx_green_;
  };
};

[[nodiscard]] _CCCL_HOST_API inline __ctx_from_stream __streamGetCtx_v2(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetCtx, cuStreamGetCtx_v2, 12, 5);

  ::CUcontext __ctx   = nullptr;
  ::CUgreenCtx __gctx = nullptr;
  __ctx_from_stream __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get context from a stream", __stream, &__ctx, &__gctx);
  if (__gctx)
  {
    __result.__ctx_kind_  = __ctx_from_stream::__kind::__green;
    __result.__ctx_green_ = __gctx;
  }
  else
  {
    __result.__ctx_kind_   = __ctx_from_stream::__kind::__device;
    __result.__ctx_device_ = __ctx;
  }
  return __result;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

// TODO: make this available since CUDA 12.8
#  if _CCCL_CTK_AT_LEAST(13, 0)
[[nodiscard]] _CCCL_HOST_API inline ::CUdevice __streamGetDevice(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetDevice, cuStreamGetDevice, 12, 8);
  ::CUdevice __result{};
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the device of the stream", __stream, &__result);
  return __result;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

_CCCL_HOST_API inline void __streamWaitEvent(::CUstream __stream, ::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamWaitEvent);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to make a stream wait for an event", __stream, __evnt, ::CU_EVENT_WAIT_DEFAULT);
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __streamQueryNoThrow(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamQuery);
  return static_cast<::cudaError_t>(__driver_fn(__stream));
}

[[nodiscard]] _CCCL_HOST_API inline int __streamGetPriority(::CUstream __stream)
{
  int __priority;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetPriority);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the priority of a stream", __stream, &__priority);
  return __priority;
}

[[nodiscard]] _CCCL_HOST_API inline unsigned long long __streamGetId(::CUstream __stream)
{
  unsigned long long __id;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetId);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the ID of a stream", __stream, &__id);
  return __id;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __streamDestroyNoThrow(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamDestroy);
  return static_cast<::cudaError_t>(__driver_fn(__stream));
}

// Event management

[[nodiscard]] _CCCL_HOST_API inline ::CUevent __eventCreate(unsigned __flags)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventCreate);
  ::CUevent __evnt;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to create a CUDA event", &__evnt, __flags);
  return __evnt;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __eventDestroyNoThrow(::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventDestroy);
  return static_cast<::cudaError_t>(__driver_fn(__evnt));
}

[[nodiscard]] _CCCL_HOST_API inline float __eventElapsedTime(::CUevent __start, ::CUevent __end)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventElapsedTime);
  float __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get event elapsed time", &__result, __start, __end);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __eventQueryNoThrow(::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventQuery);
  return static_cast<::cudaError_t>(__driver_fn(__evnt));
}

_CCCL_HOST_API inline void __eventRecord(::CUevent __evnt, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventRecord);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to record an event", __evnt, __stream);
}

_CCCL_HOST_API inline void __eventSynchronize(::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventSynchronize);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to synchronize an event", __evnt);
}

// Library management

[[nodiscard]] _CCCL_HOST_API inline ::CUfunction __kernelGetFunction(::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetFunction);
  ::CUfunction __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel function", &__result, __kernel);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline int
__kernelGetAttribute(::CUfunction_attribute __attr, ::CUkernel __kernel, ::CUdevice __dev)
{
  int __value;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel attribute", &__value, __attr, __kernel, __dev);
  return __value;
}

#  if _CCCL_CTK_AT_LEAST(12, 3)
[[nodiscard]] _CCCL_HOST_API inline const char* __kernelGetName(::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetName, cuKernelGetName, 12, 3);
  const char* __name;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel name", &__name, __kernel);
  return __name;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 3)

[[nodiscard]] _CCCL_HOST_API inline ::CUlibrary __libraryLoadData(
  const void* __code,
  ::CUjit_option* __jit_opts,
  void** __jit_opt_vals,
  unsigned __njit_opts,
  ::CUlibraryOption* __lib_opts,
  void** __lib_opt_vals,
  unsigned __nlib_opts)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryLoadData);
  ::CUlibrary __result;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn,
    "Failed to load a library from data",
    &__result,
    __code,
    __jit_opts,
    __jit_opt_vals,
    __njit_opts,
    __lib_opts,
    __lib_opt_vals,
    __nlib_opts);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::CUkernel __libraryGetKernel(::CUlibrary __library, const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetKernel);
  ::CUkernel __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get kernel from library", &__result, __library, __name);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __libraryUnloadNoThrow(::CUlibrary __library)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryUnload);
  return static_cast<::cudaError_t>(__driver_fn(__library));
}

#  if _CCCL_CTK_AT_LEAST(12, 5)
[[nodiscard]] _CCCL_HOST_API inline ::CUlibrary __kernelGetLibrary(::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetLibrary, cuKernelGetLibrary, 12, 5);
  ::CUlibrary __lib;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the library from kernel", &__lib, __kernel);
  return __lib;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__libraryGetKernelNoThrow(::CUkernel& __kernel, ::CUlibrary __lib, const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetKernel);
  return static_cast<cudaError_t>(__driver_fn(&__kernel, __lib, __name));
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__libraryGetGlobalNoThrow(::CUdeviceptr& __dptr, ::cuda::std::size_t& __nbytes, ::CUlibrary __lib, const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetGlobal);
  return static_cast<::cudaError_t>(__driver_fn(&__dptr, &__nbytes, __lib, __name));
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__libraryGetManagedNoThrow(::CUdeviceptr& __dptr, ::cuda::std::size_t& __nbytes, ::CUlibrary __lib, const char* __name)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetManaged);
  return static_cast<::cudaError_t>(__driver_fn(&__dptr, &__nbytes, __lib, __name));
}

// Execution control

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__functionGetAttributeNoThrow(int& __value, ::CUfunction_attribute __attr, ::CUfunction __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncGetAttribute);
  return static_cast<::cudaError_t>(__driver_fn(&__value, __attr, __kernel));
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __functionLoadNoThrow(::CUfunction __kernel) noexcept
{
  static auto __driver_fn = reinterpret_cast<::CUresult(CUDAAPI*)(::CUfunction)>(
    ::cuda::__driver::__get_driver_entry_point("cuFuncLoad", 12, 4));
  return static_cast<::cudaError_t>(__driver_fn(__kernel));
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__functionSetAttributeNoThrow(::CUfunction __kernel, ::CUfunction_attribute __attr, int __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncSetAttribute);
  return static_cast<::cudaError_t>(__driver_fn(__kernel, __attr, __value));
}

_CCCL_HOST_API inline void __launchHostFunc(::CUstream __stream, ::CUhostFn __fn, void* __data)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchHostFunc);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to launch host function", __stream, __fn, __data);
}

_CCCL_HOST_API inline void
__launchKernel(::CUlaunchConfig& __config, ::CUfunction __kernel, void* __args[], void* __extra[] = nullptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchKernelEx);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to launch kernel", &__config, __kernel, __args, __extra);
}

// Graph management

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddKernelNode(
  ::CUgraph __graph, const ::CUgraphNode __deps[], ::cuda::std::size_t __ndeps, ::CUDA_KERNEL_NODE_PARAMS& __node_params)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphAddKernelNode);
  ::CUgraphNode __result;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to add a node to a graph", &__result, __graph, __deps, __ndeps, &__node_params);
  return __result;
}

_CCCL_HOST_API inline void
__graphKernelNodeSetAttribute(::CUgraphNode __node, ::CUkernelNodeAttrID __id, const ::CUkernelNodeAttrValue& __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphKernelNodeSetAttribute);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to set kernel node parameters", __node, __id, &__value);
}

// Peer Context Memory Access

[[nodiscard]] _CCCL_HOST_API inline bool __deviceCanAccessPeer(::CUdevice __dev, ::CUdevice __peer_dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceCanAccessPeer);
  int __result;
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to query if device can access peer's memory", &__result, __dev, __peer_dev);
  return static_cast<bool>(__result);
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__deviceCanAccessPeerNoThrow(int& __result, ::CUdevice __dev, ::CUdevice __peer_dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceCanAccessPeer);
  return static_cast<::cudaError_t>(__driver_fn(&__result, __dev, __peer_dev));
}

// Green contexts

#  if _CCCL_CTK_AT_LEAST(12, 5)
// Add actual resource description input once exposure is ready
[[nodiscard]] _CCCL_HOST_API inline ::CUgreenCtx __greenCtxCreate(::CUdevice __dev)
{
  ::CUgreenCtx __result;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, cuGreenCtxCreate, 12, 5);
  ::cuda::__driver::__call_driver_fn(
    __driver_fn, "Failed to create a green context", &__result, nullptr, __dev, ::CU_GREEN_CTX_DEFAULT_STREAM);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __greenCtxDestroyNoThrow(::CUgreenCtx __green_ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxDestroy, cuGreenCtxDestroy, 12, 5);
  return static_cast<::cudaError_t>(__driver_fn(__green_ctx));
}

[[nodiscard]] _CCCL_HOST_API inline ::CUcontext __ctxFromGreenCtx(::CUgreenCtx __green_ctx)
{
  ::CUcontext __result;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuCtxFromGreenCtx, cuCtxFromGreenCtx, 12, 5);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to convert a green context", &__result, __green_ctx);
  return __result;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

#  if _CCCL_CTK_AT_LEAST(13, 0)
[[nodiscard]] _CCCL_HOST_API inline unsigned long long __greenCtxGetId(::CUgreenCtx __green_ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxGetId, cuGreenCtxGetId, 13, 0);
  unsigned long long __id;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get the ID of a green context", __green_ctx, &__id);
  return __id;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t
__cutensormap_size_bytes(::cuda::std::size_t __num_items, ::CUtensorMapDataType __data_type)
{
  constexpr auto __max_size = ::cuda::std::numeric_limits<::cuda::std::size_t>::max();
  switch (__data_type)
  {
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT8:
#  if _CCCL_CTK_AT_LEAST(12, 8)
    case ::CU_TENSOR_MAP_DATA_TYPE_16U6_ALIGN16B:
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
      return __num_items;
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      if (__num_items > __max_size / 2)
      {
        _CCCL_THROW(::std::invalid_argument{"Number of items must be less than or equal to 2^64 / 2"});
      }
      return __num_items * 2;
    case ::CU_TENSOR_MAP_DATA_TYPE_INT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32_FTZ:
    case ::CU_TENSOR_MAP_DATA_TYPE_TFLOAT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_TFLOAT32_FTZ:
      if (__num_items > __max_size / 4)
      {
        _CCCL_THROW(::std::invalid_argument{"Number of items must be less than or equal to 2^64 / 4"});
      }
      return __num_items * 4;
    case ::CU_TENSOR_MAP_DATA_TYPE_INT64:
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
      if (__num_items > __max_size / 8)
      {
        _CCCL_THROW(::std::invalid_argument{"Number of items must be less than or equal to 2^64 / 8"});
      }
      return __num_items * 8;
#  if _CCCL_CTK_AT_LEAST(12, 8)
    case ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B:
    case ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B:
      return __num_items / 2;
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
  }
  return 0; // MSVC workaround
}

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap __tensorMapEncodeTiled(
  ::CUtensorMapDataType __tensorDataType,
  ::cuda::std::uint32_t __tensorRank,
  void* __globalAddress,
  const ::cuda::std::uint64_t* __globalDim,
  const ::cuda::std::uint64_t* __globalStrides,
  const ::cuda::std::uint32_t* __boxDim,
  const ::cuda::std::uint32_t* __elementStrides,
  ::CUtensorMapInterleave __interleave,
  ::CUtensorMapSwizzle __swizzle,
  ::CUtensorMapL2promotion __l2Promotion,
  ::CUtensorMapFloatOOBfill __oobFill)
{
  ::CUtensorMap __tensorMap{};
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuTensorMapEncodeTiled);
  __call_driver_fn(
    __driver_fn,
    "Failed to encode TMA descriptor",
    &__tensorMap,
    __tensorDataType,
    __tensorRank,
    __globalAddress,
    __globalDim,
    __globalStrides,
    __boxDim,
    __elementStrides,
    __interleave,
    __swizzle,
    __l2Promotion,
    __oobFill);
  // workaround for nvbug 5736804
  if (::cuda::__driver::__version_below(13, 2))
  {
    const auto __tensor_req_size                = __globalDim[__tensorRank - 1] * __globalStrides[__tensorRank - 1];
    ::cuda::std::size_t __tensor_req_size_bytes = 0;
    __tensor_req_size_bytes   = ::cuda::__driver::__cutensormap_size_bytes(__tensor_req_size, __tensorDataType);
    const auto __tensorMapPtr = reinterpret_cast<::cuda::std::uint64_t*>(static_cast<void*>(&__tensorMap));
    if (__tensor_req_size_bytes < 128 * 1024) // 128 KiB
    {
      __tensorMapPtr[1] &= ~(::cuda::std::uint64_t{1} << 21); // clear the bit
    }
    else
    {
      __tensorMapPtr[1] |= ::cuda::std::uint64_t{1} << 21; // set the bit
    }
  }
  return __tensorMap;
}

#  undef _CCCLRT_GET_DRIVER_FUNCTION
#  undef _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DRIVER_DRIVER_API_H
