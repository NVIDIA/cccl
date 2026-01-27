//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
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

#if _CCCL_HAS_DLPACK()

#  include <cuda/__cmath/mul_hi.h>
#  include <cuda/__internal/dlpack.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__mdspan/mdspan_to_dlpack.h>
#  include <cuda/__memory/is_aligned.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__utility/cmp.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>
#  include <cuda/std/mdspan>

#  if !_CCCL_COMPILER(NVRTC)
#    include <stdexcept>
#  endif // !_CCCL_COMPILER(NVRTC)

#  include <dlpack/dlpack.h>
//
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _ElementType>
[[nodiscard]] _CCCL_HOST_API inline bool __validate_dlpack_data_type(const ::DLDataType& __dtype) noexcept
{
  const auto __expected = ::cuda::__data_type_to_dlpack<_ElementType>();
  return __dtype.code == __expected.code && __dtype.bits == __expected.bits && __dtype.lanes == __expected.lanes;
}

[[nodiscard]]
_CCCL_HOST_API inline ::cuda::std::int64_t
__get_layout_right_stride(const ::cuda::std::int64_t* __shapes, ::cuda::std::size_t __pos, ::cuda::std::size_t __rank)
{
  ::cuda::std::int64_t __stride = 1;
  for (auto __i = __pos + 1; __i < __rank; ++__i)
  {
    // TODO: replace with mul_overflow
    if (const auto __hi = ::cuda::mul_hi(__stride, __shapes[__i]); __hi != 0 && __hi != -1)
    {
      _CCCL_THROW(std::invalid_argument, "shape overflow");
    }
    __stride *= __shapes[__i]; // TODO: check for overflow
  }
  return __stride;
}

[[nodiscard]]
_CCCL_HOST_API inline ::cuda::std::int64_t
__get_layout_left_stride(const ::cuda::std::int64_t* __shapes, ::cuda::std::size_t __pos)
{
  ::cuda::std::int64_t __stride = 1;
  for (::cuda::std::size_t __i = 0; __i < __pos; ++__i)
  {
    // TODO: replace with mul_overflow
    if (const auto __hi = ::cuda::mul_hi(__stride, __shapes[__i]); __hi != 0 && __hi != -1)
    {
      _CCCL_THROW(std::invalid_argument, "shape overflow");
    }
    __stride *= __shapes[__i];
  }
  return __stride;
}

template <typename _LayoutPolicy>
_CCCL_HOST_API void __validate_dlpack_strides(const ::DLTensor& __tensor, [[maybe_unused]] ::cuda::std::size_t __rank)
{
  [[maybe_unused]] constexpr bool __is_layout_right = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_right>;
  [[maybe_unused]] constexpr bool __is_layout_left  = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_left>;
  [[maybe_unused]] constexpr bool __is_layout_stride =
    ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_stride>;
  const auto __strides_ptr = __tensor.strides;
  if (__strides_ptr == nullptr)
  {
#  if _CCCL_DLPACK_AT_LEAST(1, 2)
    _CCCL_THROW(std::invalid_argument, "strides=nullptr is not supported for DLPack v1.2 and later");
#  else
    // strides == nullptr means row-major (C-contiguous) layout
    if (__is_layout_left && __rank > 1)
    {
      _CCCL_THROW(std::invalid_argument, "strides must be non-null for layout_left");
    }
    else
    {
      return;
    }
#  endif // _CCCL_DLPACK_AT_LEAST(1, 2)
  }
  for (::cuda::std::size_t __pos = 0; __pos < __rank; ++__pos)
  {
    if constexpr (__is_layout_right)
    {
      if (__strides_ptr[__pos] != ::cuda::__get_layout_right_stride(__tensor.shape, __pos, __rank))
      {
        _CCCL_THROW(std::invalid_argument, "DLTensor strides are not compatible with layout_right");
      }
    }
    else if constexpr (__is_layout_left)
    {
      if (__strides_ptr[__pos] != ::cuda::__get_layout_left_stride(__tensor.shape, __pos))
      {
        _CCCL_THROW(std::invalid_argument, "DLTensor strides are not compatible with layout_left");
      }
    }
    else if constexpr (__is_layout_stride)
    {
      if (__strides_ptr[__pos] <= 0)
      {
        _CCCL_THROW(std::invalid_argument, "layout_stride requires strictly positive strides");
      }
    }
  }
}

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy>
[[nodiscard]]
_CCCL_HOST_API ::cuda::std::mdspan<_ElementType, ::cuda::std::dims<_Rank, ::cuda::std::int64_t>, _LayoutPolicy>
__to_mdspan(const ::DLTensor& __tensor)
{
  using __extents_type              = ::cuda::std::dims<_Rank, ::cuda::std::int64_t>;
  using __mdspan_type               = ::cuda::std::mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  using __mapping_type              = typename _LayoutPolicy::template mapping<__extents_type>;
  using __element_type              = typename __mdspan_type::element_type;
  constexpr bool __is_layout_right  = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_right>;
  constexpr bool __is_layout_left   = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_left>;
  constexpr bool __is_layout_stride = ::cuda::std::is_same_v<_LayoutPolicy, ::cuda::std::layout_stride>;
  // TODO: add support for layout_stride_relaxed, layout_right_padded, layout_left_padded
  if constexpr (!__is_layout_right && !__is_layout_left && !__is_layout_stride)
  {
    static_assert(::cuda::std::__always_false_v<_LayoutPolicy>, "Unsupported layout policy");
    return __mdspan_type{};
  }
  else
  {
    if (cuda::std::cmp_not_equal(__tensor.ndim, _Rank))
    {
      _CCCL_THROW(std::invalid_argument, "DLTensor rank does not match expected rank");
    }
    if (!::cuda::__validate_dlpack_data_type<__element_type>(__tensor.dtype))
    {
      _CCCL_THROW(std::invalid_argument, "DLTensor data type does not match expected type");
    }
    if (__tensor.data == nullptr)
    {
      _CCCL_THROW(std::invalid_argument, "DLTensor data must be non-null");
    }
    auto __base_data           = static_cast<char*>(__tensor.data) + __tensor.byte_offset;
    auto __data                = reinterpret_cast<__element_type*>(__base_data);
    const auto __datatype_size = __tensor.dtype.bits * __tensor.dtype.lanes / 8;
    // this is not the exact solution because data type size != data type alignment.
    // However, it always works for the supported data types.
    if (__datatype_size > 0 && !::cuda::is_aligned(__data, __datatype_size))
    {
      _CCCL_THROW(std::invalid_argument, "DLTensor data must be aligned to the data type");
    }
    if constexpr (_Rank == 0)
    {
      return __mdspan_type{__data, __mapping_type{}};
    }
    else // Rank > 0
    {
      if (__tensor.shape == nullptr)
      {
        _CCCL_THROW(std::invalid_argument, "DLTensor shape must be non-null");
      }
      ::cuda::std::array<::cuda::std::int64_t, _Rank> __extents_array{};
      for (::cuda::std::size_t __i = 0; __i < _Rank; ++__i)
      {
        if (__tensor.shape[__i] < 0)
        {
          _CCCL_THROW(std::invalid_argument, "DLTensor shapes must be positive");
        }
        __extents_array[__i] = __tensor.shape[__i];
      }
      ::cuda::__validate_dlpack_strides<_LayoutPolicy>(__tensor, _Rank);
      if constexpr (__is_layout_stride)
      {
        ::cuda::std::array<::cuda::std::int64_t, _Rank> __strides_array{};
        for (::cuda::std::size_t __i = 0; __i < _Rank; ++__i)
        {
          const bool __has_strides = __tensor.strides != nullptr;
          __strides_array[__i] =
            __has_strides ? __tensor.strides[__i] : ::cuda::__get_layout_right_stride(__tensor.shape, __i, _Rank);
        }
        return __mdspan_type{__data, __mapping_type{__extents_array, __strides_array}};
      }
      else
      {
        return __mdspan_type{__data, __extents_type{__extents_array}};
      }
    }
  }
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy = ::cuda::std::layout_stride>
[[nodiscard]]
_CCCL_HOST_API ::cuda::host_mdspan<_ElementType, ::cuda::std::dims<_Rank, ::cuda::std::int64_t>, _LayoutPolicy>
to_host_mdspan(const ::DLTensor& __tensor)
{
  if (__tensor.device.device_type != ::kDLCPU)
  {
    _CCCL_THROW(std::invalid_argument, "DLTensor device type must be kDLCPU for host_mdspan");
  }
  using __extents_type = ::cuda::std::dims<_Rank, ::cuda::std::int64_t>;
  using __mdspan_type  = ::cuda::host_mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  return __mdspan_type{::cuda::__to_mdspan<_ElementType, _Rank, _LayoutPolicy>(__tensor)};
}

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy = ::cuda::std::layout_stride>
[[nodiscard]]
_CCCL_HOST_API ::cuda::device_mdspan<_ElementType, ::cuda::std::dims<_Rank, ::cuda::std::int64_t>, _LayoutPolicy>
to_device_mdspan(const ::DLTensor& __tensor)
{
  if (__tensor.device.device_type != ::kDLCUDA)
  {
    _CCCL_THROW(std::invalid_argument, "DLTensor device type must be kDLCUDA for device_mdspan");
  }
  using __extents_type = ::cuda::std::dims<_Rank, ::cuda::std::int64_t>;
  using __mdspan_type  = ::cuda::device_mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  return __mdspan_type{::cuda::__to_mdspan<_ElementType, _Rank, _LayoutPolicy>(__tensor)};
}

template <typename _ElementType, ::cuda::std::size_t _Rank, typename _LayoutPolicy = ::cuda::std::layout_stride>
[[nodiscard]]
_CCCL_HOST_API ::cuda::managed_mdspan<_ElementType, ::cuda::std::dims<_Rank, ::cuda::std::int64_t>, _LayoutPolicy>
to_managed_mdspan(const ::DLTensor& __tensor)
{
  if (__tensor.device.device_type != ::kDLCUDAManaged)
  {
    _CCCL_THROW(std::invalid_argument, "DLTensor device type must be kDLCUDAManaged for managed_mdspan");
  }
  using __extents_type = ::cuda::std::dims<_Rank, ::cuda::std::int64_t>;
  using __mdspan_type  = ::cuda::managed_mdspan<_ElementType, __extents_type, _LayoutPolicy>;
  return __mdspan_type{::cuda::__to_mdspan<_ElementType, _Rank, _LayoutPolicy>(__tensor)};
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // __CCCL_HAS_DLPACK()
#endif // _CUDA___MDSPAN_DLPACK_TO_MDSPAN_H
