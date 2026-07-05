//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_EXPERIMENTAL___GROUP_INVOKE_ONE_CUH
#define _CUDA_EXPERIMENTAL___GROUP_INVOKE_ONE_CUH

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__ptx/instructions/elect_sync.h>
#include <cuda/hierarchy>
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__functional/invoke.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_void.h>
#include <cuda/std/__type_traits/remove_reference.h>
#include <cuda/std/__utility/forward.h>
#include <cuda/std/optional>

#include <cuda/experimental/__group/concepts.cuh>
#include <cuda/experimental/__group/fwd.cuh>

#include <cuda/std/__cccl/prologue.h>

#if !defined(_CCCL_DOXYGEN_INVOKED)

namespace cuda::experimental
{
template <class _Group>
[[nodiscard]] _CCCL_DEVICE_API bool __elect_one(const _Group& __group) noexcept
{
  if constexpr (__is_this_group_v<_Group> && ::cuda::std::is_same_v<typename _Group::level_type, warp_level>)
  {
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({ return ::cuda::ptx::elect_sync(~0u); }))
  }
  else if constexpr (!::cuda::std::is_same_v<typename _Group::unit_type, thread_level>)
  {
    // For groups whose unit is >= warp_level, we want to still execute the elect.sync PTX instruction by the root warp
    // to let the compiler enter the Uniform Data Path (UDP) when invoking the callable.
    NV_IF_TARGET(NV_PROVIDES_SM_90, ({
                   if (warp.is_root_rank(__group))
                   {
                     return ::cuda::ptx::elect_sync(~0u);
                   }
                   return false;
                 }))
  }
  return gpu_thread.is_root_rank(__group);
}

_CCCL_TEMPLATE(class _Group, class _Callable, class... _Args)
_CCCL_REQUIRES(is_group<_Group> _CCCL_AND ::cuda::std::is_invocable_v<_Callable, _Args...>
                 _CCCL_AND ::cuda::std::is_void_v<::cuda::std::invoke_result_t<_Callable, _Args...>>)
_CCCL_DEVICE_API void invoke_one(const _Group& __group, _Callable&& __callable, _Args&&... __args) noexcept(
  ::cuda::std::is_nothrow_invocable_v<_Callable, _Args...>)
{
  if (::cuda::experimental::__elect_one(__group))
  {
    ::cuda::std::invoke(::cuda::std::forward<_Callable>(__callable), ::cuda::std::forward<_Args>(__args)...);
  }
}

_CCCL_TEMPLATE(class _Group,
               class _Callable,
               class... _Args,
               class _InvokeResult = ::cuda::std::invoke_result_t<_Callable, _Args...>)
_CCCL_REQUIRES(is_group<_Group> _CCCL_AND ::cuda::std::is_invocable_v<_Callable, _Args...> _CCCL_AND(
  !::cuda::std::is_void_v<_InvokeResult>))
[[nodiscard]]
_CCCL_DEVICE_API auto invoke_one(const _Group& __group, _Callable&& __callable, _Args&&... __args) noexcept(
  ::cuda::std::is_nothrow_invocable_v<_Callable, _Args...>)
{
  using _Ret = ::cuda::std::optional<::cuda::std::conditional_t<::cuda::std::is_rvalue_reference_v<_InvokeResult>,
                                                                ::cuda::std::remove_reference_t<_InvokeResult>,
                                                                _InvokeResult>>;

  _Ret __ret{};
  if (::cuda::experimental::__elect_one(__group))
  {
    __ret = ::cuda::std::invoke(::cuda::std::forward<_Callable>(__callable), ::cuda::std::forward<_Args>(__args)...);
  }
  return __ret;
}
} // namespace cuda::experimental

#endif // !_CCCL_DOXYGEN_INVOKED

#include <cuda/std/__cccl/epilogue.h>

#endif // _CUDA_EXPERIMENTAL___GROUP_INVOKE_ONE_CUH
