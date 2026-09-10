//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA_STD___PSTL_CUDA_MDSPAN_COPY_H
#define _CUDA_STD___PSTL_CUDA_MDSPAN_COPY_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_BACKEND_CUDA()

#  include <cuda/__functional/call_or.h>
#  include <cuda/__mdspan/__copy/mdspan_d2d.h>
#  include <cuda/__mdspan/host_device_accessor.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__stream/get_stream.h>
#  include <cuda/__stream/stream_ref.h>
#  include <cuda/std/__concepts/concept_macros.h>
#  include <cuda/std/__pstl/cuda/ensure_current_context.h>
#  include <cuda/std/__pstl/dispatch.h>
#  include <cuda/std/__type_traits/is_assignable.h>
#  include <cuda/std/__type_traits/is_constructible.h>
#  include <cuda/std/__type_traits/is_execution_policy.h>
#  include <cuda/std/__type_traits/remove_cvref.h>
#  include <cuda/std/__utility/forward.h>
#  include <cuda/std/mdspan>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA_STD_EXECUTION
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

template <>
struct __pstl_dispatch<__pstl_algorithm::__mdspan_copy, __execution_backend::__cuda>
{
  template <class _Policy,
            class _TpIn,
            class _ExtentsIn,
            class _LayoutPolicyIn,
            class _AccessorPolicyIn,
            class _TpOut,
            class _ExtentsOut,
            class _LayoutPolicyOut,
            class _AccessorPolicyOut>
  _CCCL_HOST_API void _CCCL_STATIC_CALL_OPERATOR(
    const _Policy& __policy,
    const ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, ::cuda::device_accessor<_AccessorPolicyIn>>& __src,
    const ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, ::cuda::device_accessor<_AccessorPolicyOut>>& __dst)
  {
    const auto __stream = ::cuda::__call_or(::cuda::get_stream, ::cuda::stream_ref{cudaStream_t{}}, __policy);
    [[maybe_unused]] const auto __ctx = ::cuda::std::execution::__pstl_ensure_current_ctx_for(__policy);

    using __src_mdspan_t _CCCL_NODEBUG = ::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>;
    using __dst_mdspan_t _CCCL_NODEBUG =
      ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>;
    const __src_mdspan_t __device_src{__src.data_handle(), __src.mapping(), __src.accessor()};
    const __dst_mdspan_t __device_dst{__dst.data_handle(), __dst.mapping(), __dst.accessor()};

    ::cuda::copy(__device_src, __device_dst, __stream);
    __stream.sync();
  }
};

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
_CCCL_END_NAMESPACE_CUDA_STD_EXECUTION

_CCCL_BEGIN_NAMESPACE_CUDA_STD
_CCCL_BEGIN_NAMESPACE_ARCH_DEPENDENT

_CCCL_TEMPLATE(
  class _ExecutionPolicy,
  class _TpIn,
  class _ExtentsIn,
  class _LayoutPolicyIn,
  class _AccessorPolicyIn,
  class _TpOut,
  class _ExtentsOut,
  class _LayoutPolicyOut,
  class _AccessorPolicyOut)
_CCCL_REQUIRES(
  is_execution_policy_v<remove_cvref_t<_ExecutionPolicy>> _CCCL_AND
    is_assignable_v<typename ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>::reference,
                    typename ::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>::reference>
      _CCCL_AND is_constructible_v<_ExtentsIn, _ExtentsOut>)
_CCCL_HOST_API void copy(_ExecutionPolicy&& __policy,
                         const ::cuda::device_mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, _AccessorPolicyIn>& __src,
                         const ::cuda::device_mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, _AccessorPolicyOut>& __dst)
{
  using __src_accessor_t _CCCL_NODEBUG = ::cuda::device_accessor<_AccessorPolicyIn>;
  using __dst_accessor_t _CCCL_NODEBUG = ::cuda::device_accessor<_AccessorPolicyOut>;
  using __src_base_t _CCCL_NODEBUG     = ::cuda::std::mdspan<_TpIn, _ExtentsIn, _LayoutPolicyIn, __src_accessor_t>;
  using __dst_base_t _CCCL_NODEBUG     = ::cuda::std::mdspan<_TpOut, _ExtentsOut, _LayoutPolicyOut, __dst_accessor_t>;

  ::cuda::std::copy(::cuda::std::forward<_ExecutionPolicy>(__policy),
                    static_cast<const __src_base_t&>(__src),
                    static_cast<const __dst_base_t&>(__dst));
}

_CCCL_END_NAMESPACE_ARCH_DEPENDENT
_CCCL_END_NAMESPACE_CUDA_STD

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_BACKEND_CUDA()
#endif // _CUDA_STD___PSTL_CUDA_MDSPAN_COPY_H
