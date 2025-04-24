//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION_STREAM_DOMAIN
#define __CUDAX_EXECUTION_STREAM_DOMAIN

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_callable.h>

#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__stream/stream_ref.cuh>

#include <cuda/experimental/__execution/prologue.cuh>

namespace cuda::experimental::execution
{
namespace __stream
{
// Forward declaration of the __adapt function
template <class _Sndr>
_CCCL_API constexpr auto __adapt(_Sndr, stream_ref) -> decltype(auto);

template <class _Sndr>
_CCCL_API constexpr auto __adapt(_Sndr) -> decltype(auto);
} // namespace __stream

//////////////////////////////////////////////////////////////////////////////////////////
// stream domain
struct stream_domain : default_domain
{
  _CUDAX_SEMI_PRIVATE :
  struct __default_apply_t
  {
    template <class _Sndr>
    _CCCL_API constexpr auto operator()(_Sndr&& __sndr) const
    {
      return __stream::__adapt(static_cast<_Sndr&&>(__sndr));
    }
  };

  template <class _Tag>
  struct __apply_t : __default_apply_t
  {};

public:
  _CCCL_TEMPLATE(class _Tag, class _Sndr, class... _Args)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_callable_v<__apply_t<_Tag>, _Sndr, _Args...>)
  _CCCL_TRIVIAL_HOST_API static constexpr auto apply_sender(_Tag, _Sndr&& __sndr, _Args&&... __args) noexcept(
    _CUDA_VSTD::__is_nothrow_callable_v<__apply_t<_Tag>, _Sndr, _Args...>)
    -> _CUDA_VSTD::__call_result_t<__apply_t<_Tag>, _Sndr, _Args...>
  {
    return __apply_t<_Tag>{}(static_cast<_Sndr&&>(__sndr), static_cast<_Args&&>(__args)...);
  }

  _CCCL_TEMPLATE(class _Sndr, class... _Env)
  _CCCL_REQUIRES(_CUDA_VSTD::__is_callable_v<__apply_t<tag_of_t<_Sndr>>, _Sndr, const _Env&...>)
  _CCCL_TRIVIAL_API static constexpr auto transform_sender(_Sndr&& __sndr, const _Env&... __env) noexcept(
    _CUDA_VSTD::__is_nothrow_callable_v<__apply_t<tag_of_t<_Sndr>>, _Sndr, const _Env&...>) -> decltype(auto)
  {
    return __apply_t<tag_of_t<_Sndr>>{}(static_cast<_Sndr&&>(__sndr), __env...);
  }
};

} // namespace cuda::experimental::execution

#include <cuda/experimental/__execution/epilogue.cuh>

#endif // __CUDAX_EXECUTION_STREAM_DOMAIN
