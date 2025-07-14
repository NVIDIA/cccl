//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA__DRIVER_DRIVER_API
#define _CUDA__DRIVER_DRIVER_API

#include <cuda/std/__exception/cuda_error.h>

#include <cuda.h>

#include <cuda/std/__cccl/prologue.h>

// Get the driver function by name using this macro
#define _CCCLRT_GET_DRIVER_FUNCTION(function_name) \
  reinterpret_cast<decltype(function_name)*>(__get_driver_entry_point(#function_name))

#define _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(function_name, versioned_fn_name, major, minor) \
  reinterpret_cast<decltype(versioned_fn_name)*>(__get_driver_entry_point(#function_name, major, minor))

namespace cuda::__driver
{
//! @brief Get a driver function pointer for a given API name and optionally specific CUDA version
//!
//! For minor version compatibility request the 12.0 version of everything for now, unless requested otherwise
inline void*
__get_driver_entry_point(const char* __name, [[maybe_unused]] int __major = 12, [[maybe_unused]] int __minor = 0)
{
  void* __fn;
  ::cudaDriverEntryPointQueryResult __result;
#if _CCCL_CTK_AT_LEAST(12, 5)
  ::cudaGetDriverEntryPointByVersion(__name, &__fn, __major * 1000 + __minor * 10, ::cudaEnableDefault, &__result);
#else
  // Versioned get entry point not available before 12.5, but we don't need anything versioned before that
  ::cudaGetDriverEntryPoint(__name, &__fn, ::cudaEnableDefault, &__result);
#endif
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
inline void __call_driver_fn(Fn __fn, const char* __err_msg, Args... __args)
{
  ::CUresult __status = __fn(__args...);
  if (__status != ::CUDA_SUCCESS)
  {
    ::cuda::__throw_cuda_error(static_cast<::cudaError_t>(__status), __err_msg);
  }
}

inline int __getVersion()
{
  static int __version = []() {
    int __v;
    auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDriverGetVersion);
    __call_driver_fn(__driver_fn, "Failed to check CUDA driver version", &__v);
    return __v;
  }();
  return __version;
}

inline void __ctxPush(::CUcontext __ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPushCurrent);
  __call_driver_fn(__driver_fn, "Failed to push context", __ctx);
}

inline ::CUcontext __ctxPop()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxPopCurrent);
  ::CUcontext __result;
  __call_driver_fn(__driver_fn, "Failed to pop context", &__result);
  return __result;
}

inline ::CUcontext __ctxGetCurrent()
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuCtxGetCurrent);
  ::CUcontext __result;
  __call_driver_fn(__driver_fn, "Failed to get current context", &__result);
  return __result;
}

inline ::CUdevice __deviceGet(int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGet);
  ::CUdevice __result;
  __call_driver_fn(__driver_fn, "Failed to get device", &__result, __ordinal);
  return __result;
}

inline void __getName(char* __name_out, int __len, int __ordinal)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDeviceGetName);

  // TODO CUdevice is just an int, we probably could just cast, but for now do the safe thing
  ::CUdevice __dev = __deviceGet(__ordinal);
  __call_driver_fn(__driver_fn, "Failed to query the name of a device", __name_out, __len, __dev);
}

inline ::CUcontext __primaryCtxRetain(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRetain);
  ::CUcontext __result;
  __call_driver_fn(__driver_fn, "Failed to retain context for a device", &__result, __dev);
  return __result;
}

inline void __primaryCtxRelease(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxRelease);
  // TODO we might need to ignore failure here
  __call_driver_fn(__driver_fn, "Failed to release context for a device", __dev);
}

inline bool __isPrimaryCtxActive(::CUdevice __dev)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuDevicePrimaryCtxGetState);
  int __result;
  unsigned int __dummy;
  __call_driver_fn(__driver_fn, "Failed to check the primary ctx state", __dev, &__dummy, &__result);
  return __result == 1;
}

inline void __streamSynchronize(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamSynchronize);
  __call_driver_fn(__driver_fn, "Failed to synchronize a stream", __stream);
}

inline ::CUcontext __streamGetCtx(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetCtx);
  ::CUcontext __result;
  __call_driver_fn(__driver_fn, "Failed to get context from a stream", __stream, &__result);
  return __result;
}

#if _CCCL_CTK_AT_LEAST(12, 5)
struct __ctx_from_stream
{
  enum class __kind
  {
    __device,
    __green
  };

  __kind __ctx_kind;
  union
  {
    ::CUcontext __device;
    ::CUgreenCtx __green;
  } __ctx_ptr;
};

inline __ctx_from_stream __streamGetCtx_v2(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuStreamGetCtx, cuStreamGetCtx_v2, 12, 5);
  ::CUcontext __ctx       = nullptr;
  ::CUgreenCtx __gctx     = nullptr;
  __ctx_from_stream __result;
  __call_driver_fn(__driver_fn, "Failed to get context from a stream", __stream, &__ctx, &__gctx);
  if (__gctx)
  {
    __result.__ctx_kind        = __ctx_from_stream::__kind::__green;
    __result.__ctx_ptr.__green = __gctx;
  }
  else
  {
    __result.__ctx_kind         = __ctx_from_stream::__kind::__device;
    __result.__ctx_ptr.__device = __ctx;
  }
  return __result;
}
#endif // _CCCL_CTK_AT_LEAST(12, 5)

inline void __streamWaitEvent(::CUstream __stream, ::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamWaitEvent);
  __call_driver_fn(__driver_fn, "Failed to make a stream wait for an event", __stream, __evnt, ::CU_EVENT_WAIT_DEFAULT);
}

inline ::cudaError_t __streamQueryNoThrow(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamQuery);
  return static_cast<::cudaError_t>(__driver_fn(__stream));
}

inline int __streamGetPriority(::CUstream __stream)
{
  int __priority;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetPriority);
  __call_driver_fn(__driver_fn, "Failed to get the priority of a stream", __stream, &__priority);
  return __priority;
}

inline unsigned long long __streamGetId(::CUstream __stream)
{
  unsigned long long __id;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamGetId);
  __call_driver_fn(__driver_fn, "Failed to get the ID of a stream", __stream, &__id);
  return __id;
}

inline void __eventRecord(::CUevent __evnt, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventRecord);
  __call_driver_fn(__driver_fn, "Failed to record CUDA event", __evnt, __stream);
}

// Destroy calls return error codes to let the calling code decide if the error should be ignored
inline ::cudaError_t __streamDestroyNoThrow(::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuStreamDestroy);
  return static_cast<::cudaError_t>(__driver_fn(__stream));
}

inline ::cudaError_t __eventDestroyNoThrow(::CUevent __evnt)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventDestroy);
  return static_cast<::cudaError_t>(__driver_fn(__evnt));
}

inline void __eventElapsedTime(::CUevent __start, ::CUevent __end, float* __ms)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuEventElapsedTime);
  __call_driver_fn(__driver_fn, "Failed to get CUDA event elapsed time", __ms, __start, __end);
}

#if _CCCL_CTK_AT_LEAST(12, 5)
// Add actual resource description input once exposure is ready
inline ::CUgreenCtx __greenCtxCreate(::CUdevice __dev)
{
  ::CUgreenCtx __result;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxCreate, cuGreenCtxCreate, 12, 5);
  __call_driver_fn(
    __driver_fn, "Failed to create a green context", &__result, nullptr, __dev, ::CU_GREEN_CTX_DEFAULT_STREAM);
  return __result;
}

inline ::cudaError_t __greenCtxDestroyNoThrow(::CUgreenCtx __green_ctx)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuGreenCtxDestroy, cuGreenCtxDestroy, 12, 5);
  return static_cast<::cudaError_t>(__driver_fn(__green_ctx));
}

inline ::CUcontext __ctxFromGreenCtx(::CUgreenCtx __green_ctx)
{
  ::CUcontext __result;
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION_VERSIONED(cuCtxFromGreenCtx, cuCtxFromGreenCtx, 12, 5);
  __call_driver_fn(__driver_fn, "Failed to convert a green context", &__result, __green_ctx);
  return __result;
}
#endif // _CCCL_CTK_AT_LEAST(12, 5)

inline void __memcpyAsync(void* __dst, const void* __src, size_t __count, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemcpyAsync);
  __call_driver_fn(
    __driver_fn,
    "Failed to perform a memcpy",
    reinterpret_cast<::CUdeviceptr>(__dst),
    reinterpret_cast<::CUdeviceptr>(__src),
    __count,
    __stream);
}

inline void __memsetAsync(void* __dst, ::uint8_t __value, size_t __count, ::CUstream __stream)
{
  static auto __driver_fn = _CCCLRT_GET_DRIVER_FUNCTION(cuMemsetD8Async);
  __call_driver_fn(
    __driver_fn, "Failed to perform a memset", reinterpret_cast<::CUdeviceptr>(__dst), __value, __count, __stream);
}

} // namespace cuda::__driver

#undef _CCCLRT_GET_DRIVER_FUNCTION

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA__DRIVER_DRIVER_API
