//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_DLPACK_TO_MDSPAN_H
#define _CUDA___MDSPAN_DLPACK_TO_MDSPAN_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC) && _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)

#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/mdspan_to_dlpack.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/array>
#  include <cuda/std/mdspan>

#  include <stdexcept>

#  include <dlpack/dlpack.h>
//
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

static_assert(DLPACK_MAJOR_VERSION == 1, "DLPACK_MAJOR_VERSION must be 1");

template <typename _ElementType>
[[nodiscard]] _CCCL_HOST_API inline bool __validate_dlpack_data_type(const ::DLDataType& __dtype) noexcept
{
  const auto __expected = ::cuda::__data_type_to_dlpack<_ElementType>();
  return __dtype.code == __expected.code && __dtype.bits == __expected.bits && __dtype.lanes == __expected.lanes;
}

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy>
[[nodiscard]]
_CCCL_HOST_API ::cuda::std::mdspan<_ElementType, ::cuda::std::dextents<::cuda::std::int64_t, _Rank>, _LayoutPolicy>
__to_mdspan(const ::DLTensor& __tensor)
{
  using __extents_type = ::cuda::std::dextents<::cuda::std::int64_t, _Rank>;
  using __mdspan_type  = ::cuda::std::mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  using __mapping_type = typename _LayoutPolicy::template mapping<__extents_type>;
  using __element_type = typename __mdspan_type::element_type;
  if (__tensor.ndim != int{_Rank})
  {
    _CCCL_THROW(::std::invalid_argument{"DLTensor rank does not match expected rank"});
  }
  if (!::cuda::__validate_dlpack_data_type<__element_type>(__tensor.dtype))
  {
    _CCCL_THROW(::std::invalid_argument{"DLTensor data type does not match expected type"});
  }
  auto __base_data = static_cast<char*>(__tensor.data) + __tensor.byte_offset;
  auto __data      = reinterpret_cast<__element_type*>(__base_data);
  if constexpr (_Rank == 0)
  {
    return __mdspan_type{__data, __mapping_type{}};
  }
  else if constexpr (::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_stride>)
  {
    using ::cuda::std::int64_t;
    using ::cuda::std::size_t;
    ::cuda::std::array<int64_t, _Rank> __extents_arr{};
    ::cuda::std::array<int64_t, _Rank> __strides_arr{};
    for (size_t __i = 0; __i < _Rank; ++__i)
    {
      __extents_arr[__i] = __tensor.shape[__i];
      // strides == nullptr means row-major (C-contiguous) layout
      if (__tensor.strides != nullptr)
      {
        __strides_arr[__i] = __tensor.strides[__i];
      }
      else
      {
        __strides_arr[__i] = 1;
        for (size_t __j = __i + 1; __j < _Rank; ++__j)
        {
          __strides_arr[__i] *= __tensor.shape[__j];
        }
      }
    }
    __extents_type __extents{__extents_arr};
    __mapping_type __mapping{__extents, __strides_arr};
    return __mdspan_type{__data, __mapping};
  }
  else
  {
    static_assert(::cuda::std::__always_false_v<_LayoutPolicy>, "Unsupported layout policy");
  }
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy = ::cuda::std::layout_stride>
[[nodiscard]]
_CCCL_HOST_API ::cuda::host_mdspan<_ElementType, ::cuda::std::dextents<::cuda::std::int64_t, _Rank>, _LayoutPolicy>
to_host_mdspan(const ::DLTensor& __tensor)
{
  if (__tensor.device.device_type != ::kDLCPU)
  {
    _CCCL_THROW(::std::invalid_argument{"DLTensor device type must be kDLCPU for host_mdspan"});
  }
  using __extents_type = ::cuda::std::dextents<::cuda::std::int64_t, _Rank>;
  using __mdspan_type  = ::cuda::host_mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  return __mdspan_type{::cuda::__to_mdspan<_ElementType, _Rank, _LayoutPolicy>(__tensor)};
}

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy = ::cuda::std::layout_stride>
[[nodiscard]]
_CCCL_HOST_API ::cuda::device_mdspan<_ElementType, ::cuda::std::dextents<::cuda::std::int64_t, _Rank>, _LayoutPolicy>
to_device_mdspan(const ::DLTensor& __tensor)
{
  if (__tensor.device.device_type != ::kDLCUDA)
  {
    _CCCL_THROW(::std::invalid_argument{"DLTensor device type must be kDLCUDA for device_mdspan"});
  }
  using __extents_type = ::cuda::std::dextents<::cuda::std::int64_t, _Rank>;
  using __mdspan_type  = ::cuda::device_mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  return __mdspan_type{::cuda::__to_mdspan<_ElementType, _Rank, _LayoutPolicy>(__tensor)};
}

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy = ::cuda::std::layout_stride>
[[nodiscard]]
_CCCL_HOST_API ::cuda::managed_mdspan<_ElementType, ::cuda::std::dextents<::cuda::std::int64_t, _Rank>, _LayoutPolicy>
to_managed_mdspan(const ::DLTensor& __tensor)
{
  if (__tensor.device.device_type != ::kDLCUDAManaged)
  {
    _CCCL_THROW(::std::invalid_argument{"DLTensor device type must be kDLCUDAManaged for managed_mdspan"});
  }
  using __extents_type = ::cuda::std::dextents<::cuda::std::int64_t, _Rank>;
  using __mdspan_type  = ::cuda::managed_mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  return __mdspan_type{::cuda::__to_mdspan<_ElementType, _Rank, _LayoutPolicy>(__tensor)};
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC) && _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)
#endif // _CUDA___MDSPAN_DLPACK_TO_MDSPAN_H
