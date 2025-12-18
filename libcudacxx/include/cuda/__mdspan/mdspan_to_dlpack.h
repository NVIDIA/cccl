//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___MDSPAN_MDSPAN_TO_DLPACK_H
#define _CUDA___MDSPAN_MDSPAN_TO_DLPACK_H

#include <cuda/std/detail/__config>

#include <cuda/std/__type_traits/num_bits.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if !_CCCL_COMPILER(NVRTC) && _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)

#  include <cuda/__device/device_ref.h>
#  include <cuda/__mdspan/host_device_mdspan.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/__limits/numeric_limits.h>
#  include <cuda/std/__type_traits/always_false.h>
#  include <cuda/std/__type_traits/is_pointer.h>
#  include <cuda/std/__type_traits/is_same.h>
#  include <cuda/std/__type_traits/num_bits.h>
#  include <cuda/std/__type_traits/remove_cv.h>
#  include <cuda/std/__utility/cmp.h>
#  include <cuda/std/__utility/move.h>
#  include <cuda/std/array>
#  include <cuda/std/complex>
#  include <cuda/std/cstdint>
#  include <cuda/std/mdspan>

#  include <dlpack/dlpack.h>

#  if _CCCL_HAS_EXCEPTIONS()
#    include <stdexcept>
#  endif // _CCCL_HAS_EXCEPTIONS()

#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

static_assert(DLPACK_MAJOR_VERSION == 1, "DLPACK_MAJOR_VERSION must be 1");

template <typename _ElementType>
[[nodiscard]] _CCCL_HOST_API inline ::DLDataType __data_type_to_dlpack() noexcept
{
  if constexpr (::cuda::std::is_same_v<_ElementType, bool>)
  {
    return ::DLDataType{::kDLBool, 8, 1};
  }
  //--------------------------------------------------------------------------------------------------------------------
  // Signed integer types
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::int8_t>)
  {
    return ::DLDataType{::kDLInt, 8, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::int16_t>)
  {
    return ::DLDataType{::kDLInt, 16, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::int32_t>)
  {
    return ::DLDataType{::kDLInt, 32, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, long>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, long long>)
  {
    return ::DLDataType{::kDLInt, 64, 1};
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (::cuda::std::is_same_v<_ElementType, __int128_t>)
  {
    return ::DLDataType{::kDLInt, 128, 1};
  }
#  endif // _CCCL_HAS_INT128()
  //--------------------------------------------------------------------------------------------------------------------
  // Unsigned integer types
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::uint8_t>)
  {
    return ::DLDataType{::kDLUInt, 8, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::uint16_t>)
  {
    return ::DLDataType{::kDLUInt, 16, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::uint32_t>)
  {
    return ::DLDataType{::kDLUInt, 32, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, unsigned long>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, unsigned long long>)
  {
    return ::DLDataType{::kDLUInt, 64, 1};
  }
#  if _CCCL_HAS_INT128()
  else if constexpr (::cuda::std::is_same_v<_ElementType, __uint128_t>)
  {
    return ::DLDataType{::kDLUInt, 128, 1};
  }
#  endif // _CCCL_HAS_INT128()
  //--------------------------------------------------------------------------------------------------------------------
  // Floating-point types
#  if _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__half>)
  {
    return ::DLDataType{::kDLFloat, 16, 1};
  }
#  endif // _CCCL_HAS_NVFP16()
#  if _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_bfloat16>)
  {
    return ::DLDataType{::kDLBfloat, 16, 1};
  }
#  endif // _CCCL_HAS_NVBF16()
  else if constexpr (::cuda::std::is_same_v<_ElementType, float>)
  {
    return ::DLDataType{::kDLFloat, 32, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, double>)
  {
    return ::DLDataType{::kDLFloat, 64, 1};
  }
#  if _CCCL_HAS_FLOAT128()
  else if constexpr (::cuda::std::is_same_v<_ElementType, __float128>)
  {
    return ::DLDataType{::kDLFloat, 128, 1};
  }
#  endif // _CCCL_HAS_FLOAT128()
  //--------------------------------------------------------------------------------------------------------------------
  // Low-precision Floating-point types
#  if _CCCL_HAS_NVFP8_E4M3()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::__nv_fp8_e4m3>)
  {
    return ::DLDataType{::kDLFloat8_e4m3, 8, 1};
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
  // Complex types
#  if _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::complex<::__half>>)
  {
    return ::DLDataType{::kDLComplex, 32, 1};
  }
#  endif // _CCCL_HAS_NVFP16()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::complex<float>>)
  {
    return ::DLDataType{::kDLComplex, 64, 1};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::complex<double>>)
  {
    return ::DLDataType{::kDLComplex, 128, 1};
  }
#  if _CCCL_HAS_FLOAT128()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::cuda::std::complex<__float128>>)
  {
    return ::DLDataType{::kDLComplex, 256, 1};
  }
#  endif // _CCCL_HAS_FLOAT128()
  //--------------------------------------------------------------------------------------------------------------------
  // Vector types (CUDA built-in vector types)
#  if _CCCL_HAS_CTK()
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::char2>)
  {
    return ::DLDataType{::kDLInt, 8, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::char4>)
  {
    return ::DLDataType{::kDLInt, 8, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::uchar2>)
  {
    return ::DLDataType{::kDLUInt, 8, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::uchar4>)
  {
    return ::DLDataType{::kDLUInt, 8, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::short2>)
  {
    return ::DLDataType{::kDLInt, 16, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::short4>)
  {
    return ::DLDataType{::kDLInt, 16, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ushort2>)
  {
    return ::DLDataType{::kDLUInt, 16, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ushort4>)
  {
    return ::DLDataType{::kDLUInt, 16, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::int2>)
  {
    return ::DLDataType{::kDLInt, 32, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::int4>)
  {
    return ::DLDataType{::kDLInt, 32, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::uint2>)
  {
    return ::DLDataType{::kDLUInt, 32, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::uint4>)
  {
    return ::DLDataType{::kDLUInt, 32, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::long2>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::long4_32a>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::long4>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulong2>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulong4_32a>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulong4>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::long2>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::long4_32a>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::long4>)
  {
    return ::DLDataType{::kDLInt, ::cuda::std::__num_bits_v<long>, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulong2>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulong4_32a>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulong4>)
  {
    return ::DLDataType{::kDLUInt, ::cuda::std::__num_bits_v<unsigned long>, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::longlong2>)
  {
    return ::DLDataType{::kDLInt, 64, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::longlong4_32a>)
  {
    return ::DLDataType{::kDLInt, 64, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::longlong4>)
  {
    return ::DLDataType{::kDLInt, 64, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulonglong2>)
  {
    return ::DLDataType{::kDLUInt, 64, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulonglong4_32a>)
  {
    return ::DLDataType{::kDLUInt, 64, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::ulonglong4>)
  {
    return ::DLDataType{::kDLUInt, 64, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::float2>)
  {
    return ::DLDataType{::kDLFloat, 32, 2};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::float4>)
  {
    return ::DLDataType{::kDLFloat, 32, 4};
  }
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::double2>)
  {
    return ::DLDataType{::kDLFloat, 64, 2};
  }
#    if _CCCL_CTK_AT_LEAST(13, 0)
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::double4_32a>)
  {
    return ::DLDataType{::kDLFloat, 64, 4};
  }
#    else // ^^^ _CCCL_CTK_AT_LEAST(13, 0) ^^^ / vvv _CCCL_CTK_BELOW(13, 0) vvv
  else if constexpr (::cuda::std::is_same_v<_ElementType, ::double4>)
  {
    return ::DLDataType{::kDLFloat, 64, 4};
  }
#    endif // _CCCL_CTK_BELOW(13, 0)
#  endif // _CCCL_HAS_CTK()
  //--------------------------------------------------------------------------------------------------------------------
  // Unsupported types
  else
  {
    static_assert(::cuda::std::__always_false_v<_ElementType>, "Unsupported type");
  }
}

template <::cuda::std::size_t _Rank>
class DLPackWrapper
{
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __shape{};
  ::cuda::std::array<::cuda::std::int64_t, _Rank> __strides{};
  ::DLTensor __tensor{};

  _CCCL_HOST_API void __update_tensor() noexcept
  {
    __tensor.shape   = _Rank > 0 ? __shape.data() : nullptr;
    __tensor.strides = _Rank > 0 ? __strides.data() : nullptr;
  }

public:
  _CCCL_HOST_API explicit DLPackWrapper() noexcept
  {
    __update_tensor();
  }

  _CCCL_HOST_API DLPackWrapper(const DLPackWrapper& __other) noexcept
      : __shape{__other.__shape}
      , __strides{__other.__strides}
      , __tensor{__other.__tensor}
  {
    __update_tensor();
  }

  _CCCL_HOST_API DLPackWrapper(DLPackWrapper&& __other) noexcept
      : __shape{::cuda::std::move(__other.__shape)}
      , __strides{::cuda::std::move(__other.__strides)}
      , __tensor{__other.__tensor}
  {
    __other.__tensor = ::DLTensor{};
    __update_tensor();
  }

  _CCCL_HOST_API DLPackWrapper& operator=(const DLPackWrapper& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }
    __shape   = __other.__shape;
    __strides = __other.__strides;
    __tensor  = __other.__tensor;
    __update_tensor();
    return *this;
  }

  _CCCL_HOST_API DLPackWrapper& operator=(DLPackWrapper&& __other) noexcept
  {
    if (this == &__other)
    {
      return *this;
    }
    __shape          = ::cuda::std::move(__other.__shape);
    __strides        = ::cuda::std::move(__other.__strides);
    __tensor         = __other.__tensor;
    __other.__tensor = ::DLTensor{};
    __update_tensor();
    return *this;
  }

  _CCCL_HIDE_FROM_ABI ~DLPackWrapper() noexcept = default;

  _CCCL_HOST_API ::DLTensor* operator->() noexcept
  {
    return &__tensor;
  }

  _CCCL_HOST_API const ::DLTensor* operator->() const noexcept
  {
    return &__tensor;
  }

  _CCCL_HOST_API ::DLTensor& get() noexcept
  {
    return __tensor;
  }

  _CCCL_HOST_API const ::DLTensor& get() const noexcept
  {
    return __tensor;
  }
};

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API DLPackWrapper<_Extents::rank()> __mdspan_to_dlpack(
  const ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan,
  ::DLDeviceType __device_type,
  int __device_id)
{
  static_assert(::cuda::std::is_pointer_v<typename _Accessor::data_handle_type>, "data_handle_type must be a pointer");
  using __element_type = ::cuda::std::remove_cv_t<_ElementType>;
  DLPackWrapper<_Extents::rank()> __wrapper{};
  auto& __tensor  = __wrapper.get();
  __tensor.data   = __mdspan.size() > 0 ? const_cast<__element_type*>(__mdspan.data_handle()) : nullptr;
  __tensor.device = ::DLDevice{__device_type, __device_id};
  __tensor.ndim   = __mdspan.rank();
  __tensor.dtype  = ::cuda::__data_type_to_dlpack<::cuda::std::remove_cv_t<_ElementType>>();
  if constexpr (_Extents::rank() > 0)
  {
    constexpr auto __max_extent = ::cuda::std::numeric_limits<::cuda::std::int64_t>::max();
    for (::cuda::std::size_t __i = 0; __i < __mdspan.rank(); ++__i)
    {
      if (::cuda::std::cmp_greater(__mdspan.extent(__i), __max_extent))
      {
        _CCCL_THROW(::std::invalid_argument{"Extent is too large"});
      }
      if (::cuda::std::cmp_greater(__mdspan.stride(__i), __max_extent))
      {
        _CCCL_THROW(::std::invalid_argument{"Stride is too large"});
      }
      __tensor.shape[__i]   = static_cast<::cuda::std::int64_t>(__mdspan.extent(__i));
      __tensor.strides[__i] = static_cast<::cuda::std::int64_t>(__mdspan.stride(__i));
    }
  }
  __tensor.byte_offset = 0;
  return __wrapper;
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API DLPackWrapper<_Extents::rank()>
mdspan_to_dlpack(const ::cuda::host_mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan)
{
  using __mdspan_type = ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>;
  return ::cuda::__mdspan_to_dlpack(__mdspan_type{__mdspan}, ::kDLCPU, 0);
}

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API DLPackWrapper<_Extents::rank()>
mdspan_to_dlpack(const ::cuda::device_mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan,
                 ::cuda::device_ref __device = ::cuda::device_ref{0})
{
  using __mdspan_type = ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>;
  return ::cuda::__mdspan_to_dlpack(__mdspan_type{__mdspan}, ::kDLCUDA, __device.get());
}

template <typename _ElementType, typename _Extents, typename _Layout, typename _Accessor>
[[nodiscard]] _CCCL_HOST_API DLPackWrapper<_Extents::rank()>
mdspan_to_dlpack(const ::cuda::managed_mdspan<_ElementType, _Extents, _Layout, _Accessor>& __mdspan)
{
  using __mdspan_type = ::cuda::std::mdspan<_ElementType, _Extents, _Layout, _Accessor>;
  return ::cuda::__mdspan_to_dlpack(__mdspan_type{__mdspan}, ::kDLCUDAManaged, 0);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // !_CCCL_COMPILER(NVRTC) && _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)
#endif // _CUDA___MDSPAN_MDSPAN_TO_DLPACK_H
