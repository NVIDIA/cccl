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

#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__internal/namespaces.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/pod_tuple.h>
#  include <cuda/std/source_location>

#  include <cuda.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

template <class _Tp>
struct [[nodiscard]] __driver_call_result
{
  _Tp __value_;
  ::CUresult __error_;
};

template <>
struct [[nodiscard]] __driver_call_result<void>
{
  ::CUresult __error_;
};

template <class _Tp>
[[nodiscard]] _CCCL_HOST_API constexpr _Tp __check_and_extract_result(
  __driver_call_result<_Tp> __result, ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
{
  if (__result.__error_ != ::CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(__result.__error_, "", nullptr, __loc);
  }
  return __result.__value_;
}

_CCCL_HOST_API inline void __check_and_extract_result(
  __driver_call_result<void> __result, ::cuda::std::source_location __loc = ::cuda::std::source_location::current())
{
  if (__result.__error_ != ::CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(__result.__error_, "", nullptr, __loc);
  }
}

#  define _CCCL_TRY_DRIVER_API(...) ::cuda::__driver::__check_and_extract_result(::cuda::__driver::__VA_ARGS__)
#  define _CCCL_ASSERT_DRIVER_API(...)                                                                     \
    do                                                                                                     \
    {                                                                                                      \
      _CCCL_ASSERT((::cuda::__driver::__VA_ARGS__).__error_ == ::CUDA_SUCCESS, "CUDA Driver call failed"); \
    } while (0)

// Get the driver function by name using this macro
#  define _CCCLRT_GET_DRIVER_FUNCTION(function_name) \
    reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name))

#  define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
    reinterpret_cast<decltype(::versioned_fn_name)*>(                                           \
      ::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

// cudaGetDriverEntryPoint function is deprecated
_CCCL_SUPPRESS_DEPRECATED_PUSH

//! @brief Gets the cuGetProcAddress function pointer.
[[nodiscard]] _CCCL_HOST_API inline auto __getProcAddressFn() -> decltype(cuGetProcAddress)*
{
  // TODO switch to dlopen of libcuda.so instead of the below
  void* __fn;
  ::cudaDriverEntryPointQueryResult __result;
  ::cudaError_t __status = ::cudaGetDriverEntryPoint("cuGetProcAddress", &__fn, ::cudaEnableDefault, &__result);
  if (__status != ::cudaSuccess || __result != ::cudaDriverEntryPointSuccess)
  {
    ::cuda::__throw_cuda_error(::cudaErrorUnknown, "Failed to get cuGetProcAddress");
  }
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
  if (auto __status = __driver_fn(0); __status != ::CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(__status, "Failed to initialize CUDA Driver");
  }
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
[[nodiscard]] _CCCL_HOST_API inline void*
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

_CCCL_HOST_API inline __driver_call_result<int> __getVersion() noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDriverGetVersion);
  __driver_call_result<int> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<bool> __version_at_least(int __major, int __minor) noexcept
{
  __driver_call_result<bool> __ret{};
  auto __version = ::cuda::__driver::__getVersion();
  __ret.__error_ = __version.__error_;
  if (__ret.__error_ == ::CUDA_SUCCESS)
  {
    __ret.__value_ = __version.__value_ >= ::cuda::__driver::__make_version(__major, __minor);
  }
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<bool> __version_below(int __major, int __minor) noexcept
{
  __driver_call_result<bool> __ret{};
  auto __version = ::cuda::__driver::__getVersion();
  __ret.__error_ = __version.__error_;
  if (__ret.__error_ == ::CUDA_SUCCESS)
  {
    __ret.__value_ = __version.__value_ < ::cuda::__driver::__make_version(__major, __minor);
  }
  return __ret;
}

// Device management

_CCCL_HOST_API inline __driver_call_result<::CUdevice> __deviceGet(int __ordinal) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGet);
  __driver_call_result<::CUdevice> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __ordinal);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<int>
__deviceGetAttribute(::CUdevice_attribute __attr, ::CUdevice __device) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetAttribute);
  __driver_call_result<int> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __attr, __device);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<int> __deviceGetCount() noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetCount);
  __driver_call_result<int> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __deviceGetName(char* __name, int __len, ::CUdevice __device) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetName);
  return {__driver_fn(__name, __len, __device)};
}

// Primary context management

_CCCL_HOST_API inline __driver_call_result<::CUcontext> __primaryCtxRetain(::CUdevice __dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRetain);
  __driver_call_result<::CUcontext> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __dev);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __primaryCtxRelease(::CUdevice __dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRelease);
  return {__driver_fn(__dev)};
}

_CCCL_HOST_API inline __driver_call_result<bool> __isPrimaryCtxActive(::CUdevice __dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxGetState);
  __driver_call_result<bool> __ret{};
  int __active{};
  unsigned __dummy{};
  __ret.__error_ = __driver_fn(__dev, &__dummy, &__active);
  __ret.__value_ = __active == 1;
  return __ret;
}

// Context management

_CCCL_HOST_API inline __driver_call_result<void> __ctxPush(::CUcontext __ctx) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  return {__driver_fn(__ctx)};
}

_CCCL_HOST_API inline __driver_call_result<::CUcontext> __ctxPop() noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  __driver_call_result<::CUcontext> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<::CUcontext> __ctxGetCurrent() noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetCurrent);
  __driver_call_result<::CUcontext> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<::CUdevice> __ctxGetDevice() noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetDevice);
  __driver_call_result<::CUdevice> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_);
  return __ret;
}

// Memory management

_CCCL_HOST_API inline __driver_call_result<void>
__memcpyAsync(void* __dst, const void* __src, ::cuda::std::size_t __count, ::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemcpyAsync);
  return {
    __driver_fn(reinterpret_cast<::CUdeviceptr>(__dst), reinterpret_cast<::CUdeviceptr>(__src), __count, __stream)};
}

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline __driver_call_result<void> __memcpyAsyncWithAttributes(
  void* __dst, const void* __src, size_t __count, ::CUstream __stream, ::CUmemcpyAttributes __attributes) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemcpyBatchAsync, cuMemcpyBatchAsync, 13, 0);
  ::cuda::std::size_t __zero{};
  return {__driver_fn(
    reinterpret_cast<::CUdeviceptr*>(&__dst),
    reinterpret_cast<::CUdeviceptr*>(&__src),
    &__count,
    1,
    &__attributes,
    &__zero,
    1,
    __stream)};
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

template <class _Tp>
_CCCL_HOST_API __driver_call_result<void>
__memsetAsync(void* __dst, _Tp __value, size_t __count, ::CUstream __stream) noexcept
{
  if constexpr (sizeof(_Tp) == 1)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD8Async);
    return {__driver_fn(reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream)};
  }
  else if constexpr (sizeof(_Tp) == 2)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD16Async);
    return {__driver_fn(reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream)};
  }
  else if constexpr (sizeof(_Tp) == 4)
  {
    static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD32Async);
    return {__driver_fn(reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream)};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_Tp>, "Unsupported type for memset");
    return {};
  }
}

_CCCL_HOST_API inline __driver_call_result<::CUmemoryPool> __mempoolCreate(::CUmemPoolProps* __props) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolCreate);
  __driver_call_result<::CUmemoryPool> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __props);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void>
__mempoolSetAttribute(::CUmemoryPool __pool, ::CUmemPool_attribute __attr, void* __value) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolSetAttribute);
  return {__driver_fn(__pool, __attr, __value)};
}

_CCCL_HOST_API inline __driver_call_result<::cuda::std::size_t>
__mempoolGetAttribute(::CUmemoryPool __pool, ::CUmemPool_attribute __attr) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAttribute);
  __driver_call_result<::cuda::std::size_t> __ret{};
  __ret.__error_ = __driver_fn(__pool, __attr, &__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __mempoolDestroy(::CUmemoryPool __pool) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolDestroy);
  return {__driver_fn(__pool)};
}

_CCCL_HOST_API inline __driver_call_result<void*>
__mallocFromPoolAsync(::cuda::std::size_t __bytes, ::CUmemoryPool __pool, ::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocFromPoolAsync);
  __driver_call_result<void*> __ret{};
  __ret.__error_ = __driver_fn(reinterpret_cast<::CUdeviceptr*>(&__ret.__value_), __bytes, __pool, __stream);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void>
__mempoolTrimTo(::CUmemoryPool __pool, ::cuda::std::size_t __min_bytes_to_keep) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolTrimTo);
  return {__driver_fn(__pool, __min_bytes_to_keep)};
}

_CCCL_HOST_API inline __driver_call_result<void> __freeAsync(void* __ptr, ::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFreeAsync);
  return {__driver_fn(reinterpret_cast<::CUdeviceptr>(__ptr), __stream)};
}

_CCCL_HOST_API inline __driver_call_result<void>
__mempoolSetAccess(::CUmemoryPool __pool, ::CUmemAccessDesc* __descs, ::cuda::std::size_t __count) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolSetAccess);
  return {__driver_fn(__pool, __descs, __count)};
}

_CCCL_HOST_API inline __driver_call_result<::CUmemAccess_flags>
__mempoolGetAccess(::CUmemoryPool __pool, ::CUmemLocation* __location) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemPoolGetAccess);
  __driver_call_result<::CUmemAccess_flags> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __pool, __location);
  return __ret;
}

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline __driver_call_result<::CUmemoryPool>
__getDefaultMemPool(CUmemLocation __location, CUmemAllocationType_enum __allocation_type) noexcept
{
  static auto __driver_fn =
    _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuMemGetDefaultMemPool, cuMemGetDefaultMemPool, 13, 0);
  __driver_call_result<::CUmemoryPool> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, &__location, __allocation_type);
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

_CCCL_HOST_API inline __driver_call_result<void*> __mallocManaged(::cuda::std::size_t __bytes, unsigned __flags) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocManaged);
  __driver_call_result<void*> __ret{};
  __ret.__error_ = __driver_fn(reinterpret_cast<::CUdeviceptr*>(&__ret.__value_), __bytes, __flags);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void*> __mallocHost(::cuda::std::size_t __bytes) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemAllocHost);
  __driver_call_result<void*> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __bytes);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __free(void* __ptr) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFree);
  return {__driver_fn(reinterpret_cast<::CUdeviceptr>(__ptr))};
}

_CCCL_HOST_API inline __driver_call_result<void> __freeHost(void* __dptr) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemFreeHost);
  return {__driver_fn(__dptr)};
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
_CCCL_HOST_API inline __driver_call_result<__pointer_attribute_value_type_t<_Attr>>
__pointerGetAttribute(const void* __ptr) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuPointerGetAttribute);
  __driver_call_result<__pointer_attribute_value_type_t<_Attr>> __ret{};
  if constexpr (::cuda::std::is_same_v<__pointer_attribute_value_type_t<_Attr>, bool>)
  {
    int __result2{};
    __ret.__error_ = __driver_fn(&__result2, _Attr, reinterpret_cast<::CUdeviceptr>(__ptr));
    __ret.__value_ = static_cast<bool>(__result2);
  }
  else
  {
    __ret.__error_ = __driver_fn((void*) &__ret.__value_, _Attr, reinterpret_cast<::CUdeviceptr>(__ptr));
  }
  return __ret;
}

template <::cuda::std::size_t _Np>
_CCCL_HOST_API inline __driver_call_result<void>
__pointerGetAttributes(::CUpointer_attribute (&__attrs)[_Np], void* (&__results)[_Np], const void* __ptr) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuPointerGetAttributes);
  return {__driver_fn(static_cast<unsigned>(_Np), __attrs, __results, reinterpret_cast<::CUdeviceptr>(__ptr))};
}

// Stream management

_CCCL_HOST_API inline __driver_call_result<void>
__streamAddCallback(::CUstream __stream, ::CUstreamCallback __cb, void* __data, unsigned __flags = 0) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamAddCallback);
  return {__driver_fn(__stream, __cb, __data, __flags)};
}

_CCCL_HOST_API inline __driver_call_result<::CUstream>
__streamCreateWithPriority(unsigned __flags, int __priority) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamCreateWithPriority);
  __driver_call_result<::CUstream> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __flags, __priority);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __streamSynchronize(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamSynchronize);
  return {__driver_fn(__stream)};
}

_CCCL_HOST_API inline __driver_call_result<::CUcontext> __streamGetCtx(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetCtx);
  __driver_call_result<::CUcontext> __ret{};
  __ret.__error_ = __driver_fn(__stream, &__ret.__value_);
  return __ret;
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

_CCCL_HOST_API inline __driver_call_result<__ctx_from_stream> __streamGetCtx_v2(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetCtx, cuStreamGetCtx_v2, 12, 5);

  ::CUcontext __ctx   = nullptr;
  ::CUgreenCtx __gctx = nullptr;
  __driver_call_result<__ctx_from_stream> __ret{};
  __ret.__error_ = __driver_fn(__stream, &__ctx, &__gctx);
  if (__gctx)
  {
    __ret.__value_.__ctx_kind_  = __ctx_from_stream::__kind::__green;
    __ret.__value_.__ctx_green_ = __gctx;
  }
  else
  {
    __ret.__value_.__ctx_kind_   = __ctx_from_stream::__kind::__device;
    __ret.__value_.__ctx_device_ = __ctx;
  }
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

// TODO: make this available since CUDA 12.8
#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline __driver_call_result<::CUdevice> __streamGetDevice(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetDevice, cuStreamGetDevice, 12, 8);
  __driver_call_result<::CUdevice> __ret{};
  __ret.__error_ = __driver_fn(__stream, &__ret.__value_);
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

_CCCL_HOST_API inline __driver_call_result<void> __streamWaitEvent(::CUstream __stream, ::CUevent __evnt) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamWaitEvent);
  return {__driver_fn(__stream, __evnt, ::CU_EVENT_WAIT_DEFAULT)};
}

_CCCL_HOST_API inline __driver_call_result<void> __streamQuery(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamQuery);
  return {__driver_fn(__stream)};
}

_CCCL_HOST_API inline __driver_call_result<int> __streamGetPriority(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetPriority);
  __driver_call_result<::CUdevice> __ret{};
  __ret.__error_ = __driver_fn(__stream, &__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<unsigned long long> __streamGetId(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetId);
  __driver_call_result<unsigned long long> __ret{};
  __ret.__error_ = __driver_fn(__stream, &__ret.__value_);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __streamDestroy(::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamDestroy);
  return {__driver_fn(__stream)};
}

// Event management

_CCCL_HOST_API inline __driver_call_result<::CUevent> __eventCreate(unsigned __flags) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventCreate);
  __driver_call_result<::CUevent> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __flags);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __eventDestroy(::CUevent __evnt) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventDestroy);
  return {__driver_fn(__evnt)};
}

_CCCL_HOST_API inline __driver_call_result<float> __eventElapsedTime(::CUevent __start, ::CUevent __end) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventElapsedTime);
  __driver_call_result<float> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __start, __end);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __eventQuery(::CUevent __evnt) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventQuery);
  return {__driver_fn(__evnt)};
}

_CCCL_HOST_API inline __driver_call_result<void> __eventRecord(::CUevent __evnt, ::CUstream __stream) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventRecord);
  return {__driver_fn(__evnt, __stream)};
}

_CCCL_HOST_API inline __driver_call_result<void> __eventSynchronize(::CUevent __evnt) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventSynchronize);
  return {__driver_fn(__evnt)};
}

// Library management

_CCCL_HOST_API inline __driver_call_result<::CUfunction> __kernelGetFunction(::CUkernel __kernel) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetFunction);
  __driver_call_result<::CUfunction> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __kernel);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<int>
__kernelGetAttribute(::CUfunction_attribute __attr, ::CUkernel __kernel, ::CUdevice __dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetAttribute);
  __driver_call_result<int> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __attr, __kernel, __dev);
  return __ret;
}

#  if _CCCL_CTK_AT_LEAST(12, 3)
_CCCL_HOST_API inline __driver_call_result<const char*> __kernelGetName(::CUkernel __kernel) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetName, cuKernelGetName, 12, 3);
  __driver_call_result<const char*> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __kernel);
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 3)

_CCCL_HOST_API inline __driver_call_result<::CUlibrary> __libraryLoadData(
  const void* __code,
  ::CUjit_option* __jit_opts,
  void** __jit_opt_vals,
  unsigned __njit_opts,
  ::CUlibraryOption* __lib_opts,
  void** __lib_opt_vals,
  unsigned __nlib_opts)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryLoadData);
  __driver_call_result<::CUlibrary> __ret{};
  __ret.__error_ = __driver_fn(
    &__ret.__value_, __code, __jit_opts, __jit_opt_vals, __njit_opts, __lib_opts, __lib_opt_vals, __nlib_opts);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<::CUkernel>
__libraryGetKernel(::CUlibrary __library, const char* __name) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetKernel);
  __driver_call_result<::CUkernel> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __library, __name);
  return __ret;
}

#  if _CCCL_CTK_AT_LEAST(12, 5)
_CCCL_HOST_API inline __driver_call_result<::CUlibrary> __kernelGetLibrary(::CUkernel __kernel) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetLibrary, cuKernelGetLibrary, 12, 5);
  __driver_call_result<::CUlibrary> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __kernel);
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

_CCCL_HOST_API inline __driver_call_result<::cuda::std::__tuple<void*, ::cuda::std::size_t>>
__libraryGetGlobal(::CUlibrary __lib, const char* __name) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetGlobal);
  __driver_call_result<::cuda::std::__tuple<void*, ::cuda::std::size_t>> __ret{};
  __ret.__error_ =
    __driver_fn(reinterpret_cast<::CUdeviceptr*>(&__ret.__value_.__val0), &__ret.__value_.__val1, __lib, __name);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<::cuda::std::__tuple<void*, ::cuda::std::size_t>>
__libraryGetManaged(::CUlibrary __lib, const char* __name) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryGetManaged);
  __driver_call_result<::cuda::std::__tuple<void*, ::cuda::std::size_t>> __ret{};
  __ret.__error_ =
    __driver_fn(reinterpret_cast<::CUdeviceptr*>(&__ret.__value_.__val0), &__ret.__value_.__val1, __lib, __name);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __libraryUnload(::CUlibrary __lib) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLibraryUnload);
  return {__driver_fn(__lib)};
}

// Execution control

_CCCL_HOST_API inline __driver_call_result<int>
__functionGetAttribute(::CUfunction_attribute __attr, ::CUfunction __kernel) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncGetAttribute);
  __driver_call_result<int> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __attr, __kernel);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __functionLoad(::CUfunction __kernel) noexcept
{
  static auto __driver_fn = reinterpret_cast<::CUresult(CUDAAPI*)(::CUfunction)>(
    ::cuda::__driver::__get_driver_entry_point("cuFuncLoad", 12, 4));
  return {__driver_fn(__kernel)};
}

_CCCL_HOST_API inline __driver_call_result<void>
__functionSetAttribute(::CUfunction __kernel, ::CUfunction_attribute __attr, int __value) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncSetAttribute);
  return {__driver_fn(__kernel, __attr, __value)};
}

_CCCL_HOST_API inline __driver_call_result<void>
__launchHostFunc(::CUstream __stream, ::CUhostFn __fn, void* __data) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchHostFunc);
  return {__driver_fn(__stream, __fn, __data)};
}

_CCCL_HOST_API inline __driver_call_result<void>
__launchKernel(::CUlaunchConfig& __config, ::CUfunction __kernel, void* __args[], void* __extra[] = nullptr) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchKernelEx);
  return {__driver_fn(&__config, __kernel, __args, __extra)};
}

// Graph management

_CCCL_HOST_API inline __driver_call_result<::CUgraphNode> __graphAddKernelNode(
  ::CUgraph __graph,
  const ::CUgraphNode __deps[],
  ::cuda::std::size_t __ndeps,
  ::CUDA_KERNEL_NODE_PARAMS& __node_params) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphAddKernelNode);
  __driver_call_result<::CUgraphNode> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __graph, __deps, __ndeps, &__node_params);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __graphKernelNodeSetAttribute(
  ::CUgraphNode __node, ::CUkernelNodeAttrID __id, const ::CUkernelNodeAttrValue& __value) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphKernelNodeSetAttribute);
  return {__driver_fn(__node, __id, &__value)};
}

// Peer Context Memory Access

_CCCL_HOST_API inline __driver_call_result<bool> __deviceCanAccessPeer(::CUdevice __dev, ::CUdevice __peer_dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceCanAccessPeer);
  __driver_call_result<bool> __ret{};
  int __result{};
  __ret.__error_ = __driver_fn(&__result, __dev, __peer_dev);
  __ret.__value_ = static_cast<bool>(__result);
  return __ret;
}

// Green contexts

#  if _CCCL_CTK_AT_LEAST(12, 5)
// Add actual resource description input once exposure is ready
_CCCL_HOST_API inline __driver_call_result<::CUgreenCtx> __greenCtxCreate(::CUdevice __dev) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, cuGreenCtxCreate, 12, 5);
  __driver_call_result<::CUgreenCtx> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, nullptr, __dev, ::CU_GREEN_CTX_DEFAULT_STREAM);
  return __ret;
}

_CCCL_HOST_API inline __driver_call_result<void> __greenCtxDestroy(::CUgreenCtx __green_ctx) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxDestroy, cuGreenCtxDestroy, 12, 5);
  return {__driver_fn(__green_ctx)};
}

_CCCL_HOST_API inline __driver_call_result<::CUcontext> __ctxFromGreenCtx(::CUgreenCtx __green_ctx) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuCtxFromGreenCtx, cuCtxFromGreenCtx, 12, 5);
  __driver_call_result<::CUcontext> __ret{};
  __ret.__error_ = __driver_fn(&__ret.__value_, __green_ctx);
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(12, 5)

#  if _CCCL_CTK_AT_LEAST(13, 0)
_CCCL_HOST_API inline __driver_call_result<unsigned long long> __greenCtxGetId(::CUgreenCtx __green_ctx) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxGetId, cuGreenCtxGetId, 13, 0);
  __driver_call_result<unsigned long long> __ret{};
  __ret.__error_ = __driver_fn(__green_ctx, &__ret.__value_);
  return __ret;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

_CCCL_HOST_API inline __driver_call_result<::CUtensorMap> __tensorMapEncodeTiled(
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
  ::CUtensorMapFloatOOBfill __oobFill) noexcept
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuTensorMapEncodeTiled);
  __driver_call_result<::CUtensorMap> __ret{};
  __ret.__error_ = __driver_fn(
    &__ret.__value_,
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
  return __ret;
}

#  undef _CCCLRT_GET_DRIVER_FUNCTION
#  undef _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC)

#endif // _CUDA___DRIVER_DRIVER_API_H
