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

#if _CCCL_HAS_CTK()

#  include <cuda/std/__exception/cuda_error.h>
#  include <cuda/std/__internal/namespaces.h>
#  include <cuda/std/__type_traits/always_false.h>

#  include <cuda.h>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_DRIVER

// Get the driver function by name using this macro
#  define _CCCLRT_GET_DRIVER_FUNCTION(function_name) \
    reinterpret_cast<decltype(::function_name)*>(::cuda::__driver::__get_driver_entry_point(#function_name))

#  define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
    reinterpret_cast<decltype(::versioned_fn_name)*>(                                           \
      ::cuda::__driver::__get_driver_entry_point(#function_name, major, minor))

//! @brief Get a driver function pointer for a given API name and optionally specific CUDA version
//!
//! For minor version compatibility request the 12.0 version of everything for now, unless requested otherwise
[[nodiscard]] _CCCL_HOST_API inline void*
__get_driver_entry_point(const char* __name, [[maybe_unused]] int __major = 12, [[maybe_unused]] int __minor = 0)
{
  void* __fn;
  ::cudaDriverEntryPointQueryResult __result;
#  if _CCCL_CTK_AT_LEAST(12, 5)
  ::cudaGetDriverEntryPointByVersion(__name, &__fn, __major * 1000 + __minor * 10, ::cudaEnableDefault, &__result);
#  else
  // Versioned get entry point not available before 12.5, but we don't need anything versioned before that
  ::cudaGetDriverEntryPoint(__name, &__fn, ::cudaEnableDefault, &__result);
#  endif
  if (__result != ::cudaDriverEntryPointSuccess)
  {
    if (__result == ::cudaDriverEntryPointVersionNotSufficent)
    {
      ::cuda::__throw_cuda_error(::cudaErrorNotSupported, "Driver does not support this API");
    }
    else
    {
      ::cuda::__throw_cuda_error(::cudaErrorUnknown, "Failed to access driver API");
    }
  }
  return __fn;
}

template <typename Fn, typename... Args>
_CCCL_HOST_API inline void __call_driver_fn(Fn __fn, const char* __err_msg, Args... __args)
{
  ::CUresult __status = __fn(__args...);
  if (__status != ::CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<::cudaError_t>(__status), __err_msg);
  }
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

// Device management

[[nodiscard]] _CCCL_HOST_API inline ::CUdevice __deviceGet(int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGet);
  ::CUdevice __result;
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get device", &__result, __ordinal);
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

_CCCL_HOST_API inline void __primaryCtxRelease(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRelease);
  // TODO we might need to ignore failure here
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to release context for a device", __dev);
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

// Memory management

_CCCL_HOST_API inline void __memcpyAsync(void* __dst, const void* __src, size_t __count, ::CUstream __stream)
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

template <typename _Tp>
_CCCL_HOST_API void __memsetAsync(void* __dst, _Tp __value, size_t __count, ::CUstream __stream)
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

// Stream management

_CCCL_HOST_API inline void __streamSynchronize(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamSynchronize);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to synchronize a stream", __stream);
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
  ::CUcontext __ctx       = nullptr;
  ::CUgreenCtx __gctx     = nullptr;
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

// Event management

_CCCL_HOST_API inline void __eventRecord(::CUevent __evnt, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventRecord);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to record CUDA event", __evnt, __stream);
}

// Destroy calls return error codes to let the calling code decide if the error should be ignored
[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __streamDestroyNoThrow(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamDestroy);
  return static_cast<::cudaError_t>(__driver_fn(__stream));
}

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t __eventDestroyNoThrow(::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventDestroy);
  return static_cast<::cudaError_t>(__driver_fn(__evnt));
}

_CCCL_HOST_API inline void __eventElapsedTime(::CUevent __start, ::CUevent __end, float* __ms)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventElapsedTime);
  ::cuda::__driver::__call_driver_fn(__driver_fn, "Failed to get CUDA event elapsed time", __ms, __start, __end);
}

// Library management

[[nodiscard]] _CCCL_HOST_API inline ::CUfunction __kernelGetFunction(::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetFunction);
  ::CUfunction __result;
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to get kernel function", &__result, __kernel);
  return __result;
}

[[nodiscard]] _CCCL_HOST_API inline int
__kernelGetAttribute(::CUfunction_attribute __attr, ::CUkernel __kernel, ::CUdevice __dev)
{
  int __value;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuKernelGetAttribute);
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to get kernel attribute", &__value, __attr, __kernel, __dev);
  return __value;
}

#  if _CCCL_CTK_AT_LEAST(12, 3)
[[nodiscard]] _CCCL_HOST_API inline const char* __kernelGetName(::CUkernel __kernel)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuKernelGetName, cuKernelGetName, 12, 3);
  const char* __name;
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to get kernel name", &__name, __kernel);
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
  _CUDA_DRIVER::__call_driver_fn(
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
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to get kernel from library", &__result, __library, __name);
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
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to get the library from kernel", &__lib, __kernel);
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

[[nodiscard]] _CCCL_HOST_API inline ::cudaError_t
__functionSetAttributeNoThrow(::CUfunction __kernel, ::CUfunction_attribute __attr, int __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuFuncSetAttribute);
  return static_cast<::cudaError_t>(__driver_fn(__kernel, __attr, __value));
}

_CCCL_HOST_API inline void
__launchKernel(::CUlaunchConfig& __config, ::CUfunction __kernel, void* __args[], void* __extra[] = nullptr)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuLaunchKernelEx);
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to launch kernel", &__config, __kernel, __args, __extra);
}

// Graph management

[[nodiscard]] _CCCL_HOST_API inline ::CUgraphNode __graphAddKernelNode(
  ::CUgraph __graph, const ::CUgraphNode __deps[], ::cuda::std::size_t __ndeps, ::CUDA_KERNEL_NODE_PARAMS& __node_params)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphAddKernelNode);
  ::CUgraphNode __result;
  _CUDA_DRIVER::__call_driver_fn(
    __driver_fn, "Failed to add a node to a graph", &__result, __graph, __deps, __ndeps, &__node_params);
  return __result;
}

_CCCL_HOST_API inline void
__graphKernelNodeSetAttribute(::CUgraphNode __node, ::CUkernelNodeAttrID __id, const ::CUkernelNodeAttrValue& __value)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuGraphKernelNodeSetAttribute);
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to set kernel node parameters", __node, __id, &__value);
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
  _CUDA_DRIVER::__call_driver_fn(__driver_fn, "Failed to get the ID of a green context", __green_ctx, &__id);
  return __id;
}
#  endif // _CCCL_CTK_AT_LEAST(13, 0)

#  undef _CCCLRT_GET_DRIVER_FUNCTION
#  undef _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED

_CCCL_END_NAMESPACE_CUDA_DRIVER

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK()

#endif // _CUDA___DRIVER_DRIVER_API_H
