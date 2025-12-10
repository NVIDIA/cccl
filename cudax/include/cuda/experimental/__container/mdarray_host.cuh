//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
#define __CUDAX__CONTAINER_MDARRAY_DEVICE_CUH

#include <cuda/std/detail/__config>

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__memory/allocator.h>
#include <cuda/std/__utility/delegate_constructors.h>

#include <cuda/experimental/__container/mdarray_base.cuh>
#include <cuda/experimental/__container/mdarray_utils.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy,
          typename _Allocator = ::cuda::std::allocator<_ElementType>>
class host_mdarray
    : public __base_mdarray<host_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
                            _Allocator,
                            ::cuda::host_mdspan,
                            _ElementType,
                            _Extents,
                            _LayoutPolicy>
{
  using __base_class =
    __base_mdarray<host_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
                   _Allocator,
                   ::cuda::host_mdspan,
                   _ElementType,
                   _Extents,
                   _LayoutPolicy>;

  friend __base_class;

  _CCCL_HOST_API static _Allocator __get_default_allocator()
  {
    return ::cuda::std::allocator<_ElementType>{};
  }

  _CCCL_HOST_API void __init()
  {
    ::cuda::experimental::__init_host_impl(this->view(), ::cuda::std::make_index_sequence<_Extents::rank()>{});
  }

  template <typename _ElementType2, typename _Extents2, typename _LayoutPolicy2, typename _Accessor2>
  _CCCL_HOST_API void __copy_from(::cuda::host_mdspan<_ElementType2, _Extents2, _LayoutPolicy2, _Accessor2> __mdspan_in)
  {
    ::cuda::experimental::__for_each_in_layout_host(__mdspan_in.mapping(), _CopyOp{__mdspan_in, this->view()});
  }

  template <typename _Extents2, typename _LayoutPolicy2>
  _CCCL_HOST_API void __copy_from(::cuda::device_mdspan<_ElementType, _Extents2, _LayoutPolicy2> __mdspan_in)
  {
    ::cuda::experimental::__copy_host_device(__mdspan_in, this->view(), ::cudaStream_t{nullptr});
  }

public:
  using view_type       = ::cuda::host_mdspan<_ElementType, _Extents, _LayoutPolicy>;
  using const_view_type = ::cuda::host_mdspan<const _ElementType, _Extents, _LayoutPolicy>;

  _CCCL_DELEGATE_CONSTRUCTORS(
    host_mdarray, __base_mdarray, host_mdarray, _Allocator, ::cuda::host_mdspan, _ElementType, _Extents, _LayoutPolicy);

  _CCCL_HIDE_FROM_ABI host_mdarray(const host_mdarray&) = default;
  _CCCL_HIDE_FROM_ABI host_mdarray(host_mdarray&&)      = default;
  _CCCL_HIDE_FROM_ABI ~host_mdarray() noexcept          = default;

  _CCCL_HIDE_FROM_ABI host_mdarray& operator=(const host_mdarray&)     = default;
  _CCCL_HIDE_FROM_ABI host_mdarray& operator=(host_mdarray&&) noexcept = default;
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
