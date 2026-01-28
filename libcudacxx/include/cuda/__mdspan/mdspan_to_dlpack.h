//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_MDSPAN_TO_DLPACK_H
#define _CUDA___MDSPAN_MDSPAN_TO_DLPACK_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_DLPACK()

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__internal/dlpack.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/__type_traits/is_floating_point.h>
#  include <cuda/__type_traits/is_vector_type.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__fwd/complex.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/num_bits.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/__utility/cmp.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>
#  include <cuda/std/mdspan>

#  include <stdexcept>

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

template <typename _ElementType>
[[nodiscard]] _CCCL_HOST_API inline ::DLDataType __data_type_to_dlpack() noexcept
{
  if constexpr (::cuda::std::is_same_v<_ElementType, bool>)
  {
    return ::DLDataType{::kDLBool, 8, 1};
  }
  //--------------------------------------------------------------------------------------------------------------------
  // Signed integer types
  else if constexpr (::cuda::std::__cccl_is_integer_v<_ElementType>)
  {
    return ::DLDataType{
      (::cuda::std::is_signed_v<_ElementType>) ? ::kDLInt : ::kDLUInt, ::cuda::std::__num_bits_v<_ElementType>, 1};
  }
  //--------------------------------------------------------------------------------------------------------------------
  // bfloat16 (must come before general floating-point)
#  if _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_bfloat16>)
  {
    return ::DLDataType{::kDLBfloat, 16, 1};
  }
#  endif // _CCCL_HAS_NVBF16()
  //--------------------------------------------------------------------------------------------------------------------
  // Low-precision Floating-point types (must come before general floating-point)
#  if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp8_e4m3>)
  {
    return ::DLDataType{::kDLFloat8_e4m3fn, 8, 1};
  }
#  endif // _CCCL_HAS_NVFP8_E4M3()
#  if _CCCL_HAS_NVFP8_E5M2()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp8_e5m2>)
  {
    return ::DLDataType{::kDLFloat8_e5m2, 8, 1};
  }
#  endif // _CCCL_HAS_NVFP8_E5M2()
#  if _CCCL_HAS_NVFP8_E8M0()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp8_e8m0>)
  {
    return ::DLDataType{::kDLFloat8_e8m0fnu, 8, 1};
  }
#  endif // _CCCL_HAS_NVFP8_E8M0()
#  if _CCCL_HAS_NVFP6_E2M3()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp6_e2m3>)
  {
    return ::DLDataType{::kDLFloat6_e2m3fn, 6, 1};
  }
#  endif // _CCCL_HAS_NVFP6_E2M3()
#  if _CCCL_HAS_NVFP6_E3M2()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp6_e3m2>)
  {
    return ::DLDataType{::kDLFloat6_e3m2fn, 6, 1};
  }
#  endif // _CCCL_HAS_NVFP6_E3M2()
#  if _CCCL_HAS_NVFP4_E2M1()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp4_e2m1>)
  {
    return ::DLDataType{::kDLFloat4_e2m1fn, 4, 1};
  }
#  endif // _CCCL_HAS_NVFP4_E2M1()
  //--------------------------------------------------------------------------------------------------------------------
  // Floating-point types (after specific types)
  else if constexpr (::cuda::is_floating_point_v<_ElementType>)
  {
    return ::DLDataType{::kDLFloat, ::cuda::std::__num_bits_v<_ElementType>, 1};
  }
  //--------------------------------------------------------------------------------------------------------------------
  // Complex types
  // 256-bit data types are not supported in DLPack, e.g. cuda::std::complex<__float128>
  else if constexpr (::cuda::std::__is_cuda_std_complex_v<_ElementType> && sizeof(_ElementType) <= sizeof(double) * 2)
  {
    // DLPack encodes complex numbers as a compact struct of two scalar values, and `bits` stores
    // the size of the full complex number (e.g. std::complex<float> => bits=64).
    return ::DLDataType{::kDLComplex, sizeof(_ElementType) * CHAR_BIT, 1};
  }
  //--------------------------------------------------------------------------------------------------------------------
  // CUDA built-in vector types
#  if _CCCL_HAS_CTK()
  else if constexpr (::cuda::is_vector_type_v<_ElementType> || ::cuda::is_extended_fp_vector_type_v<_ElementType>)
  {
    constexpr ::cuda::std::uint16_t __lanes = ::cuda::std::tuple_size_v<_ElementType>;
    if constexpr (__lanes == 2 || __lanes == 4)
    {
      using __scalar_t = ::cuda::std::remove_cv_t<::cuda::std::tuple_element_t<0, _ElementType>>;
      auto __scalar    = ::cuda::__data_type_to_dlpack<__scalar_t>();
      __scalar.lanes   = __lanes;
      return __scalar;
    }
    else
    {
      static_assert(::cuda::std::__always_false_v<_ElementType>, "Unsupported vector type");
      return ::DLDataType{};
    }
  }
#  endif // _CCCL_HAS_CTK()
  //--------------------------------------------------------------------------------------------------------------------
  // Unsupported types
  else
  {
    static_assert(::cuda::std::__always_false_v<_ElementType>, "Unsupported type");
    return ::DLDataType{};
  }
}

template <::cuda::std::size_t _Rank>
struct __dlpack_tensor
{
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __shape{};
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __strides{};
  ::DLTensor __tensor{};

  [[nodiscard]] _CCCL_HOST_API ::DLTensor get() const& noexcept _CCCL_LIFETIMEBOUND
  {
    auto __tensor1    = __tensor;
    __tensor1.shape   = _Rank > 0 ? const_cast<::cuda::std::int64_t*>(__shape.data()) : nullptr;
    __tensor1.strides = _Rank > 0 ? const_cast<::cuda::std::int64_t*>(__strides.data()) : nullptr;
    return __tensor1;
  }

  ::DLTensor get() const&& = delete;
};

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API __dlpack_tensor<_Extents::rank()>
__to_dlpack(const ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan,
            ::DLDeviceType __device_type,
            int __device_id)
{
  static_assert(::cuda::std::is_pointer_v<typename _Accessor::data_handle_type>, "data_handle_type must be a pointer");
  using __element_type = ::cuda::std::remove_cv_t<_ElementType>;
  __dlpack_tensor<_Extents::rank()> __wrapper{};
  auto& __tensor  = __wrapper.__tensor;
  __tensor.data   = __mdspan.size() > 0 ? const_cast<__element_type*>(__mdspan.data_handle()) : nullptr;
  __tensor.device = ::DLDevice{__device_type, __device_id};
  __tensor.ndim   = static_cast<int>(__mdspan.rank());
  __tensor.dtype  = ::cuda::__data_type_to_dlpack<::cuda::std::remove_cv_t<_ElementType>>();
  if constexpr (_Extents::rank() > 0)
  {
    constexpr auto __max_extent = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max();
    for (::cuda::std::size_t __i = 0; __i < __mdspan.rank(); ++__i)
    {
      if (::cuda::std::cmp_greater(__mdspan.extent(__i), __max_extent))
      {
        _CCCL_THROW(std::invalid_argument, "Extent is too large");
      }
      if (::cuda::std::cmp_greater(__mdspan.stride(__i), __max_extent))
      {
        _CCCL_THROW(std::invalid_argument, "Stride is too large");
      }
      __wrapper.__shape[__i]   = static_cast<::cuda::std::int64_t>(__mdspan.extent(__i));
      __wrapper.__strides[__i] = static_cast<::cuda::std::int64_t>(__mdspan.stride(__i));
    }
  }
  __tensor.byte_offset = 0;
  return __wrapper;
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API __dlpack_tensor<_Extents::rank()>
to_dlpack_tensor(const ::cuda::host_mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan)
{
  using __mdspan_type = ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>;
  return ::cuda::__to_dlpack(__mdspan_type{__mdspan}, ::kDLCPU, 0);
}

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API __dlpack_tensor<_Extents::rank()>
to_dlpack_tensor(const ::cuda::device_mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan)
{
  using __mdspan_type              = ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>;
  ::CUpointer_attribute __attrs[1] = {::CU_POINTER_ATTRIBUTE_DEVICE_ORDINAL};
  int __ptr_dev_id                 = 0;
  void* __results[1]               = {&__ptr_dev_id};
  const auto __status = ::cuda::__driver::__pointerGetAttributesNoThrow(__attrs, __results, __mdspan.data_handle());
  if (__status != ::cudaSuccess)
  {
    ::cuda::__throw_cuda_error(__status, "Failed to get device ordinal of a pointer");
  }
  return ::cuda::__to_dlpack(__mdspan_type{__mdspan}, ::kDLCUDA, __ptr_dev_id);
}

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API __dlpack_tensor<_Extents::rank()>
to_dlpack_tensor(const ::cuda::managed_mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan)
{
  using __mdspan_type = ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>;
  return ::cuda::__to_dlpack(__mdspan_type{__mdspan}, ::kDLCUDAManaged, 0);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_DLPACK()
#endif // _CUDA___MDSPAN_MDSPAN_TO_DLPACK_H
