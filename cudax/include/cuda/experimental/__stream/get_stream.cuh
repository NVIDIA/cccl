//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDAX__STREAM_GET_STREAM
#define _CUDAX__STREAM_GET_STREAM

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda_runtime_api.h>

#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__cuda/api_wrapper.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/stream_ref>

#include <cuda/experimental/__stream/stream.cuh>

namespace cuda::experimental
{
// clang-format off
template <class _Tp>
_CCCL_CONCEPT __has_member_get_stream =
  _CCCL_REQUIRES_EXPR((_Tp), const _Tp& __t)
  (
    requires(!_CUDA_VSTD::convertible_to<_Tp, ::cuda::stream_ref>),
    _Same_as(::cuda::stream_ref) __t.get_stream()
  );
// clang-format on

//! @brief `get_stream` is a customization point object that queries a type `T` for an associated stream
struct get_stream_t
{
  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES((_CUDA_VSTD::convertible_to<_Tp, ::cuda::stream_ref>) )
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref operator()(const _Tp& __t) const
    noexcept(noexcept(static_cast<::cuda::stream_ref>(__t)))
  {
    return static_cast<::cuda::stream_ref>(__t);
  } // namespace __get_stream

  _CCCL_TEMPLATE(class _Tp)
  _CCCL_REQUIRES(__has_member_get_stream<_Tp>)
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref operator()(const _Tp& __t) const
    noexcept(noexcept(__t.get_stream()))
  {
    return __t.get_stream();
  }

  _CCCL_TEMPLATE(class _Env,
                 class _Ret = decltype(_CUDA_VSTD::declval<const _Env&>().query(_CUDA_VSTD::declval<get_stream_t>())))
  _CCCL_REQUIRES(_CCCL_TRAIT(_CUDA_VSTD::is_convertible, _Ret, ::cuda::stream_ref))
  _CCCL_NODISCARD _CCCL_HIDE_FROM_ABI constexpr ::cuda::stream_ref operator()(const _Env& __env) const noexcept
  {
    static_assert(noexcept(__env.query(*this)), "");
    return __env.query(*this);
  }
};

_CCCL_GLOBAL_CONSTANT auto get_stream = get_stream_t{};

} // namespace cuda::experimental

#endif // _CUDAX__STREAM_GET_STREAM
