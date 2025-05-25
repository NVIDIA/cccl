//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_RCVR_REF
#define __CUDAX_EXECUTION_RCVR_REF

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/is_nothrow_copy_constructible.h>
#include <cuda/std/__type_traits/is_same.h>

#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/type_traits.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

_CCCL_NV_DIAG_SUPPRESS(114) // function "foo" was referenced but not defined

namespace cuda::experimental::execution
{
template <class _Rcvr, class _Env = env_of_t<_Rcvr>>
struct _CCCL_TYPE_VISIBILITY_DEFAULT __rcvr_ref
{
  using receiver_concept = receiver_t;

  _CCCL_API explicit constexpr __rcvr_ref(_Rcvr& __rcvr) noexcept
      : __rcvr_{_CUDA_VSTD::addressof(__rcvr)}
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... _As>
  _CCCL_TRIVIAL_API void set_value(_As&&... __as) noexcept
  {
    static_cast<_Rcvr&&>(*__rcvr_).set_value(static_cast<_As&&>(__as)...);
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class _Error>
  _CCCL_TRIVIAL_API void set_error(_Error&& __err) noexcept
  {
    static_cast<_Rcvr&&>(*__rcvr_).set_error(static_cast<_Error&&>(__err));
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_TRIVIAL_API void set_stopped() noexcept
  {
    static_cast<_Rcvr&&>(*__rcvr_).set_stopped();
  }

  _CCCL_API auto get_env() const noexcept -> _Env
  {
    static_assert(_CUDA_VSTD::is_same_v<_Env, env_of_t<_Rcvr>>,
                  "get_env() must return the same type as env_of_t<_Rcvr>");
    return execution::get_env(*__rcvr_);
  }

private:
  _Rcvr* __rcvr_;
};

namespace __detail
{
template <class _Rcvr, size_t = sizeof(_Rcvr)>
_CCCL_API constexpr auto __is_type_complete(int) noexcept
{
  return true;
}
template <class _Rcvr>
_CCCL_API constexpr auto __is_type_complete(long) noexcept
{
  return false;
}
} // namespace __detail

// The __ref_rcvr function and its helpers are used to avoid wrapping a receiver in a
// __rcvr_ref when that is possible. The logic goes as follows:
//
// 1. If the receiver is an instance of __rcvr_ref, return it.
// 2. If the type is incomplete or an operation state, return a __rcvr_ref wrapping the
//    receiver.
// 3. If the receiver is nothrow copy constructible, return it.
// 4. Otherwise, return a __rcvr_ref wrapping the receiver.
template <class _Env = void, class _Rcvr>
[[nodiscard]] _CCCL_TRIVIAL_API constexpr auto __ref_rcvr(_Rcvr& __rcvr) noexcept
{
  if constexpr (_CUDA_VSTD::is_same_v<_Env, void>)
  {
    return execution::__ref_rcvr<env_of_t<_Rcvr>>(__rcvr);
  }
  else if constexpr (__is_specialization_of_v<_Rcvr, __rcvr_ref>)
  {
    return __rcvr;
  }
  else if constexpr (!__detail::__is_type_complete<_Rcvr>(0))
  {
    return __rcvr_ref<_Rcvr, _Env>{__rcvr};
  }
  else if constexpr (__is_operation_state<_Rcvr>)
  {
    return __rcvr_ref<_Rcvr, _Env>{__rcvr};
  }
  else if constexpr (_CUDA_VSTD::is_nothrow_copy_constructible_v<_Rcvr>)
  {
    return __rcvr;
  }
  else
  {
    return __rcvr_ref<_Rcvr, _Env>{__rcvr};
  }
  _CCCL_UNREACHABLE();
}

template <class _Rcvr, class _Env = env_of_t<_Rcvr>>
using __rcvr_ref_t _CCCL_NODEBUG_ALIAS = decltype(::cuda::experimental::execution::__ref_rcvr<_Env>(declval<_Rcvr&>()));

} // namespace cuda::experimental::execution

_CCCL_NV_DIAG_DEFAULT(114) // function "foo" was references but not defined

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_RCVR_REF
