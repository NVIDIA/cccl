//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_HOST_CUH
#define __CUDAX__CONTAINER_MDARRAY_HOST_CUH

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
#include <cuda/std/__concepts/concept_macros.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/remove_const.h>
#include <cuda/std/__utility/delegate_constructors.h>

#include <cuda/experimental/__container/mdarray_base.cuh>
#include <cuda/experimental/__container/mdarray_utils.cuh>

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _ElementType,
          typename _Extents,
          typename _LayoutPolicy,
          typename _Allocator = ::cuda::experimental::__host_allocator>
class host_mdarray
    : public __base_mdarray<host_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
                            _Allocator,
                            ::cuda::host_mdspan,
                            _ElementType,
                            _Extents,
                            _LayoutPolicy>
{
  static_assert(::cuda::has_property<_Allocator, ::cuda::mr::host_accessible>);

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
    return ::cuda::experimental::__host_allocator{};
  }

  _CCCL_HOST_API void __init()
  {
    ::cuda::experimental::__init_host_impl(this->view());
  }

  template <typename _ElementType2, typename _Extents2, typename _LayoutPolicy2, typename _Accessor2>
  _CCCL_HOST_API void __copy_from(::cuda::host_mdspan<_ElementType2, _Extents2, _LayoutPolicy2, _Accessor2> __mdspan_in)
  {
    ::cuda::experimental::__for_each_in_layout_host(__mdspan_in.mapping(), _CopyOp{__mdspan_in, this->view()});
  }

  // TODO: very inefficient but copying different layouts is also possible
  _CCCL_TEMPLATE(typename _ElementType2, typename _Extents2)
  _CCCL_REQUIRES(::cuda::std::is_same_v<::cuda::std::remove_const_t<_ElementType2>, _ElementType>)
  _CCCL_HOST_API void __copy_from(::cuda::device_mdspan<_ElementType2, _Extents2, _LayoutPolicy> __mdspan_in)
  {
    using __view_type = ::cuda::std::mdspan<_ElementType, _Extents2, _LayoutPolicy>;
    auto __view_in    = static_cast<__view_type>(__mdspan_in);
    auto __view_out   = static_cast<__view_type>(this->view());
    ::cuda::experimental::__copy_host_device(__view_in, __view_out, ::cudaStream_t{nullptr});
  }

  template <typename _Tp>
  [[nodiscard]] _CCCL_API decltype(auto) __access_single_element(_Tp&& __ref) const noexcept
  {
    return ::cuda::std::forward<_Tp>(__ref);
  }

public:
  _CCCL_DELEGATE_CONSTRUCTORS(
    host_mdarray, __base_mdarray, host_mdarray, _Allocator, ::cuda::host_mdspan, _ElementType, _Extents, _LayoutPolicy);

  _CCCL_HIDE_FROM_ABI host_mdarray(const host_mdarray&)     = default;
  _CCCL_HIDE_FROM_ABI host_mdarray(host_mdarray&&) noexcept = default;
  _CCCL_HIDE_FROM_ABI ~host_mdarray() noexcept              = default;

  _CCCL_HIDE_FROM_ABI host_mdarray& operator=(const host_mdarray&)     = default;
  _CCCL_HIDE_FROM_ABI host_mdarray& operator=(host_mdarray&&) noexcept = default;

  using __base_class::operator=;
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_HOST_CUH
