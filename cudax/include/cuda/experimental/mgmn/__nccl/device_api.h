// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#ifndef _CUDA_EXPERIMENTAL_MGMN___NCCL_DEVICE_API_H
#define _CUDA_EXPERIMENTAL_MGMN___NCCL_DEVICE_API_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif

#if defined(__CUDACC__) && __has_include(<nccl_device.h>)
#  include <nccl_device.h>
#  if NCCL_VERSION_CODE >= NCCL_VERSION(2, 29, 2)
#    define _CUDAX_HAS_NCCL_DEVICE_API 1

#    include <cuda/experimental/mgmn/__nccl/nccl_api.h>

#    include <cuda/std/__cccl/prologue.h>

// NOLINTBEGIN(bugprone-reserved-identifier)
namespace cuda::experimental::__nccl
{
// Keep device support optional even when its SDK headers are installed. In
// particular, including sort must not add a link-time dependency on libnccl.
struct __device_api
{
  using __team_function                                      = ::ncclTeam_t (*)(::ncclComm_t);
  decltype(&::ncclCommQueryProperties) __query               = nullptr;
  __team_function __lsa                                      = nullptr;
  __team_function __world                                    = nullptr;
  decltype(&::ncclMemAlloc) __allocate                       = nullptr;
  decltype(&::ncclMemFree) __free                            = nullptr;
  decltype(&::ncclCommWindowRegister) __register_window      = nullptr;
  decltype(&::ncclCommWindowDeregister) __deregister_window  = nullptr;
  decltype(&::ncclDevCommCreate) __create                    = nullptr;
  decltype(&::ncclDevCommDestroy) __destroy                  = nullptr;
  decltype(&::ncclGinBarrierCreateRequirement) __gin_barrier = nullptr;

  [[nodiscard]] _CCCL_HOST_API constexpr bool __supports_lsa() const noexcept
  {
    return __query && __lsa && __allocate && __free && __register_window && __deregister_window && __create
        && __destroy;
  }

  [[nodiscard]] _CCCL_HOST_API constexpr bool __supports_gin() const noexcept
  {
    return __supports_lsa() && __world && __gin_barrier;
  }
};

[[nodiscard]] _CCCL_HOST_API inline const __device_api& __get_device_api()
{
  static const auto __api = [] {
    auto& __library = ::cuda::experimental::__nccl::__nccl_lib();
    __device_api __result;
    // Do not pass newer structure layouts to an older runtime. Newer runtimes
    // receive the original SDK initializers and perform their own LSA/GIN ABI
    // compatibility checks through ncclCommQueryProperties/ncclDevCommCreate.
    const auto __version_fn = __library.load_symbol<decltype(&::ncclGetVersion)>("ncclGetVersion", true);
    int __version           = 0;
    if (!__version_fn || __version_fn(&__version) != ::ncclSuccess || __version < NCCL_VERSION_CODE)
    {
      return __result;
    }
    __result.__query    = __library.load_symbol<decltype(__result.__query)>("ncclCommQueryProperties", true);
    __result.__lsa      = __library.load_symbol<__device_api::__team_function>("ncclTeamLsa", true);
    __result.__world    = __library.load_symbol<__device_api::__team_function>("ncclTeamWorld", true);
    __result.__allocate = __library.load_symbol<decltype(__result.__allocate)>("ncclMemAlloc", true);
    __result.__free     = __library.load_symbol<decltype(__result.__free)>("ncclMemFree", true);
    __result.__register_window =
      __library.load_symbol<decltype(__result.__register_window)>("ncclCommWindowRegister", true);
    __result.__deregister_window =
      __library.load_symbol<decltype(__result.__deregister_window)>("ncclCommWindowDeregister", true);
    __result.__create  = __library.load_symbol<decltype(__result.__create)>("ncclDevCommCreate", true);
    __result.__destroy = __library.load_symbol<decltype(__result.__destroy)>("ncclDevCommDestroy", true);
    __result.__gin_barrier =
      __library.load_symbol<decltype(__result.__gin_barrier)>("ncclGinBarrierCreateRequirement", true);
    return __result;
  }();
  return __api;
}
} // namespace cuda::experimental::__nccl
// NOLINTEND(bugprone-reserved-identifier)

#    include <cuda/std/__cccl/epilogue.h>
#  endif // NCCL 2.29.2+
#endif // CUDA compiler and device SDK

#endif // _CUDA_EXPERIMENTAL_MGMN___NCCL_DEVICE_API_H
