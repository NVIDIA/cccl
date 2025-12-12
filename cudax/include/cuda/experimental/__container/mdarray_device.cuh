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
#include <cuda/std/__type_traits/remove_const.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/__device/all_devices.h>
#include <cuda/__driver/driver_api.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/device_memory_pool.h>
#include <cuda/__memory_resource/shared_resource.h>
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
          typename _Allocator = ::cuda::mr::shared_resource<::cuda::device_memory_pool>>
class device_mdarray
    : public __base_mdarray<device_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
                            _Allocator,
                            ::cuda::device_mdspan,
                            _ElementType,
                            _Extents,
                            _LayoutPolicy>
{
  static_assert(::cuda::has_property<_Allocator, ::cuda::mr::device_accessible>);

  using __base_class =
    __base_mdarray<device_mdarray<_ElementType, _Extents, _LayoutPolicy, _Allocator>,
                   _Allocator,
                   ::cuda::device_mdspan,
                   _ElementType,
                   _Extents,
                   _LayoutPolicy>;

  using reference  = typename ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>::reference;
  using value_type = typename ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>::value_type;
  friend __base_class;

  _CCCL_HOST_API static _Allocator __get_default_allocator()
  {
    static auto __allocator = __construct_allocator<_Allocator>::__do(::cuda::__devices()[0]);
    return __allocator;
  }

  _CCCL_HOST_API void __init()
  {
    ::cuda::experimental::__init_device_impl(this->view(), ::cudaStream_t{nullptr});
  }

  template <typename _ElementType2, typename _Extents2, typename _LayoutPolicy2, typename _Accessor2>
  _CCCL_HOST_API void
  __copy_from(::cuda::device_mdspan<_ElementType2, _Extents2, _LayoutPolicy2, _Accessor2> __mdspan_in)
  {
    auto __temp_storage         = reinterpret_cast<void*>(0x1);
    size_t __temp_storage_bytes = 0;
    cub::DeviceCopy::Copy(__temp_storage, __temp_storage_bytes, __mdspan_in, this->view(), ::cudaStream_t{nullptr});
  }

  _CCCL_TEMPLATE(typename _ElementType2, typename _Extents2)
  _CCCL_REQUIRES(::cuda::std::is_same_v<::cuda::std::remove_const_t<_ElementType2>, _ElementType>)
  _CCCL_HOST_API void __copy_from(::cuda::host_mdspan<_ElementType2, _Extents2, _LayoutPolicy> __mdspan_in)
  {
    // TODO: check extents compatibility
    using __view_type = ::cuda::std::mdspan<_ElementType, _Extents2, _LayoutPolicy>;
    auto __view_in    = static_cast<__view_type>(__mdspan_in);
    auto __view_out   = static_cast<__view_type>(this->view());
    ::cuda::experimental::__copy_host_device(__view_in, __view_out, ::cudaStream_t{nullptr});
  }

  [[nodiscard]] _CCCL_API value_type __access_single_element(value_type& )
  {
    using __value_type = ::cuda::std::remove_reference_t<reference>;
    __value_type __value;
   // printf("__ref: %p\n", &__ref);
   // ::cuda::__driver::__memcpyAsync(&__value, &__ref, sizeof(reference), ::cudaStream_t{nullptr});
    return __value;
  }

public:
  using view_type       = ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>;
  using const_view_type = ::cuda::device_mdspan<const _ElementType, _Extents, _LayoutPolicy>;

  _CCCL_DELEGATE_CONSTRUCTORS(
    device_mdarray,
    __base_mdarray,
    device_mdarray,
    _Allocator,
    ::cuda::device_mdspan,
    _ElementType,
    _Extents,
    _LayoutPolicy);

  _CCCL_HIDE_FROM_ABI device_mdarray(const device_mdarray&) = default;
  _CCCL_HIDE_FROM_ABI device_mdarray(device_mdarray&&)      = default;
  _CCCL_HIDE_FROM_ABI ~device_mdarray() noexcept            = default;

  _CCCL_HIDE_FROM_ABI device_mdarray& operator=(const device_mdarray&)     = default;
  _CCCL_HIDE_FROM_ABI device_mdarray& operator=(device_mdarray&&) noexcept = default;

  using __base_class::operator=;
};
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_DEVICE_CUH
