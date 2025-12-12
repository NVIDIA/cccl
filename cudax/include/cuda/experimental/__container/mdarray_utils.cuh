//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX__CONTAINER_MDARRAY_UTILS_CUH
#define __CUDAX__CONTAINER_MDARRAY_UTILS_CUH

#include <cuda/std/detail/__config>

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/device/device_copy.cuh>
#include <cub/device/device_for.cuh>

#include <cuda/__device/device_ref.h>
#include <cuda/__mdspan/host_device_mdspan.h>
#include <cuda/__memory_resource/shared_resource.h>
#include <cuda/__stream/stream_ref.h>
#include <cuda/std/__cstdlib/aligned_alloc.h>
#include <cuda/std/__utility/integer_sequence.h>
#include <cuda/std/__utility/typeid.h>

#if _CCCL_HAS_EXCEPTIONS()
#  include <stdexcept>
#endif // _LIBCUDACXX_HAS_STRING

#include <cuda/std/__cccl/prologue.h>

namespace cuda::experimental
{
template <typename _Alloc>
struct __construct_allocator
{
  [[nodiscard]] _CCCL_HOST_API static _Alloc __do()
  {
    return _Alloc{};
  }

  [[nodiscard]] _CCCL_HOST_API static _Alloc __do(::cuda::device_ref)
  {
    return _Alloc{};
  }
};

template <typename _Resource>
struct __construct_allocator<::cuda::mr::shared_resource<_Resource>>
{
  [[nodiscard]] _CCCL_HOST_API static ::cuda::mr::shared_resource<_Resource> __do(::cuda::device_ref __device)
  {
    return ::cuda::mr::make_shared_resource<_Resource>(__device);
  }
};

template <typename _ElementType, typename _Extents, typename _LayoutPolicy>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__mapping_size_bytes(const ::cuda::host_mdspan<_ElementType, _Extents, _LayoutPolicy>& __mdspan) noexcept
{
  return __mdspan.mapping().required_span_size() * sizeof(_ElementType);
}

template <typename _ElementType, typename _Extents, typename _LayoutPolicy>
[[nodiscard]] _CCCL_HOST_API ::cuda::std::size_t
__mapping_size_bytes(const ::cuda::device_mdspan<_ElementType, _Extents, _LayoutPolicy>& __mdspan) noexcept
{
  return __mdspan.mapping().required_span_size() * sizeof(_ElementType);
}

template <int K, int Rank, typename _LayoutMapping, typename _OpType, typename... IndicesType>
_CCCL_HOST_API void __for_each_in_layout_host(
  const _LayoutMapping& __layout_mapping, _OpType __op, ::cuda::std::size_t& __index, IndicesType... __indices)
{
  // TODO: static_assert that the layout mapping is a valid layout mapping
  if constexpr (K == Rank)
  {
    __op(__index++, __indices...);
  }
  else if constexpr (::cuda::std::is_same_v<_LayoutMapping, ::cuda::std::layout_left>)
  {
    for (int i = 0; i < __layout_mapping.extents().extent(K); ++i)
    {
      ::cuda::experimental::__for_each_in_layout_host<K + 1, Rank>(__layout_mapping, __op, __index, i, __indices...);
    }
  }
  else
  {
    for (int i = 0; i < __layout_mapping.extents().extent(K); ++i)
    {
      ::cuda::experimental::__for_each_in_layout_host<K + 1, Rank>(__layout_mapping, __op, __index, __indices..., i);
    }
  }
}

class __host_allocator
{
public:
  __host_allocator()                        = default;
  __host_allocator(const __host_allocator&) = default;
  __host_allocator(__host_allocator&&)      = default;
  ~__host_allocator()                       = default;

  [[nodiscard]] friend bool operator==(const __host_allocator& __lhs, const __host_allocator& __rhs) noexcept
  {
    return &__lhs == &__rhs;
  }

  [[nodiscard]] void* allocate_sync(::cuda::std::size_t __size,
                                    ::cuda::std::size_t __align = alignof(cuda::std::max_align_t)) const
  {
    //_CCCL_ASSERT(__size % __align == 0, "Size must be divisible by alignment");
    //_CCCL_ASSERT(::cuda::is_power_of_two(__align), "Alignment must be a power of two");
    //_CCCL_ASSERT(__align % sizeof(void*) == 0, "Alignment must be a multiple of the size of void*");
    // auto __ptr = ::cuda::std::aligned_alloc(__align, __size);
    auto __ptr = ::cuda::std::malloc(__size);
    if (__ptr == nullptr)
    {
      _CCCL_THROW(::std::bad_alloc{});
    }
    return __ptr;
  }

  void
  deallocate_sync(void* __pv, ::cuda::std::size_t, ::cuda::std::size_t = alignof(cuda::std::max_align_t)) const noexcept
  {
    ::cuda::std::free(__pv);
  }

  [[nodiscard]] void* allocate(cuda::stream_ref, ::cuda::std::size_t __size, ::cuda::std::size_t __align) const
  {
    return allocate_sync(__size, __align);
  }

  [[nodiscard]] void* allocate(cuda::stream_ref, ::cuda::std::size_t __size) const
  {
    auto __ptr = ::cuda::std::malloc(__size);
    if (__ptr == nullptr)
    {
      _CCCL_THROW(::std::bad_alloc{});
    }
    return __ptr;
  }

  void deallocate(cuda::stream_ref, void* __pv, ::cuda::std::size_t __size, ::cuda::std::size_t __align) const noexcept
  {
    deallocate_sync(__pv, __size, __align);
  }

  void deallocate(cuda::stream_ref, void* __pv, ::cuda::std::size_t __size) const noexcept
  {
    deallocate_sync(__pv, __size);
  }

  _CCCL_HOST_API friend constexpr void get_property(const __host_allocator& __res, ::cuda::mr::host_accessible) noexcept
  {}
};

//----------------------------------------------------------------------------------------------------------------------
// Copy

template <typename _View1, typename _View2>
struct _CopyOp
{
  _View1 __view1_;
  _View2 __view2_;

  template <typename... _Indices>
  _CCCL_HOST_API void operator()(_Indices... __indices)
  {
    __view1_(__indices...) = __view2_(__indices...);
  }
};

template <typename _View1, typename _View2>
struct _CopyOpHostDevice
{
  using _LayoutPolicy1 = typename _View1::layout_type;
  using _LayoutPolicy2 = typename _View2::layout_type;

  _View1 __view1_;
  _View2 __view2_;
  ::cuda::stream_ref __stream_;

  template <typename... _Indices>
  _CCCL_HOST_API void operator()(::cuda::std::size_t, _Indices... __indices)
  {
    using _ElementType1 = typename _View1::element_type;
    using _Extents1     = typename _View1::extents_type;
    if constexpr (::cuda::std::is_same_v<_LayoutPolicy1, ::cuda::std::layout_right>)
    {
      auto __data_handle1 = &__view1_(__indices..., 0);
      auto __data_handle2 = &__view2_(__indices..., 0);
      ::cuda::__driver::__memcpyAsync(
        __data_handle2, __data_handle1, __view1_.extent(_Extents1::rank() - 1) * sizeof(_ElementType1), __stream_.get());
    }
    else if constexpr (::cuda::std::is_same_v<_LayoutPolicy1, ::cuda::std::layout_left>)
    {
      auto __data_handle1 = &__view1_(0, __indices...);
      auto __data_handle2 = &__view2_(0, __indices...);
      ::cuda::__driver::__memcpyAsync(
        __data_handle2, __data_handle1, __view1_.extent(0) * sizeof(_ElementType1), __stream_.get());
    }
  }
};

_CCCL_TEMPLATE(typename _ElementType1,
               typename _Extents1,
               typename _LayoutPolicy1,
               typename _ElementType2,
               typename _Extents2,
               typename _LayoutPolicy2)
_CCCL_REQUIRES(::cuda::std::is_same_v<::cuda::std::remove_const_t<_ElementType1>, _ElementType2>)
_CCCL_HOST_API void __copy_host_device(::cuda::std::mdspan<_ElementType1, _Extents1, _LayoutPolicy1> __mdspan_in,
                                       ::cuda::std::mdspan<_ElementType2, _Extents2, _LayoutPolicy2> __mdspan_out,
                                       ::cuda::stream_ref __stream = ::cudaStream_t{nullptr})
{
  if (!__mdspan_in.is_exhaustive())
  {
    _CCCL_THROW(::std::invalid_argument("Source and destination mdspans must be exhaustive"));
  }
  if (__mdspan_in.mapping() != __mdspan_out.mapping())
  {
    _CCCL_THROW(::std::invalid_argument("Source and destination mappings must be the same"));
  }
  if (__mdspan_in.is_exhaustive() && __mdspan_out.is_exhaustive())
  {
    ::cuda::__driver::__memcpyAsync(
      __mdspan_out.data_handle(),
      __mdspan_in.data_handle(),
      __mdspan_in.mapping().required_span_size() * sizeof(_ElementType1),
      __stream.get());
  }
  else if constexpr (::cuda::std::is_same_v<_LayoutPolicy1, ::cuda::std::layout_right>)
  {
    ::cuda::std::size_t __index = 0;
    ::cuda::experimental::__for_each_in_layout_host<0, _Extents1::rank() - 1>(
      __mdspan_in.mapping(), _CopyOpHostDevice{__mdspan_in, __mdspan_out, __stream}, __index);
  }
  else if constexpr (::cuda::std::is_same_v<_LayoutPolicy1, ::cuda::std::layout_left>)
  {
    ::cuda::std::size_t __index = 0;
    ::cuda::experimental::__for_each_in_layout_host<1, _Extents1::rank()>(
      __mdspan_in.mapping(), _CopyOpHostDevice{__mdspan_in, __mdspan_out, __stream}, __index);
  }
}

//----------------------------------------------------------------------------------------------------------------------
// Initialization

template <typename _View, typename _ElementType>
struct _InitOp
{
  _View __view_;

  template <typename _IndexType, typename... _Indices>
  _CCCL_HOST_DEVICE void operator()(_IndexType, _Indices... __indices)
  {
    __view_(__indices...) = _ElementType{};
  }
};

template <typename _ElementType, typename _Extents, typename _LayoutPolicy>
_CCCL_HOST_API void
__init_device_impl(device_mdspan<_ElementType, _Extents, _LayoutPolicy> __mdspan, ::cuda::stream_ref __stream)
{
  using _View = device_mdspan<_ElementType, _Extents, _LayoutPolicy>;
  if constexpr (::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_right>
                || ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_left>)
  {
    cub::DeviceFor::ForEachInLayout(__mdspan.mapping(), _InitOp<_View, _ElementType>{__mdspan}, __stream.get());
  }
  else
  {
    cub::DeviceFor::ForEachInExtents(__mdspan.extents(), _InitOp<_View, _ElementType>{__mdspan}, __stream.get());
  }
}

template <typename _ElementType, typename _Extents, typename _LayoutPolicy, typename... IndicesType>
_CCCL_HOST_API void __init_host_impl(host_mdspan<_ElementType, _Extents, _LayoutPolicy> __mdspan)
{
  using __view_type           = host_mdspan<_ElementType, _Extents, _LayoutPolicy>;
  ::cuda::std::size_t __index = 0;
  ::cuda::experimental::__for_each_in_layout_host<0, _Extents::rank()>(
    __mdspan.mapping(), _InitOp<__view_type, _ElementType>{__mdspan}, __index);
}
} // namespace cuda::experimental

#include <cuda/std/__cccl/epilogue.h>

#endif //__CUDAX__CONTAINER_MDARRAY_UTILS_CUH
