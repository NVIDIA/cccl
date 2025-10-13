//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___TMA_DESCR_TMA_DESCR_H
#define _CUDA___TMA_DESCR_TMA_DESCR_H

#include <cuda/std/detail/__config>

#include "cuda/std/__memory/is_sufficiently_aligned.h"

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

// #if _CCCL_HAS_INCLUDE(<dlpack.h>)
#include <cuda/__tma_descr/dlpack.h>
#include <cuda/std/__memory/is_sufficiently_aligned.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>

#include <cudaTypedefs.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// #if _CCCL_HOST_COMPILATION()

enum class OOBfill
{
  none,
  nan
};

enum class L2FetchSize
{
  none,
  Bytes64,
  Bytes128,
  Bytes256,
};

/***********************************************************************************************************************
 * Internal API
 ***********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t __to_cutensor(OOBfill __oobfill) noexcept
{
  switch (__oobfill)
  {
    case OOBfill::none:
      return ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case OOBfill::nan:
      return ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  }
  _CCCL_UNREACHABLE();
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t __to_cutensor(L2FetchSize __l2_fetch_size) noexcept
{
  switch (__l2_fetch_size)
  {
    case L2FetchSize::none:
      return ::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case L2FetchSize::Bytes64:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case L2FetchSize::Bytes128:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case L2FetchSize::Bytes256:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
  }
  _CCCL_UNREACHABLE();
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t
__cutensor_data_type_to_size(::CUtensorMapDataType __data_type) noexcept
{
  switch (__data_type)
  {
    case ::CU_TENSOR_MAP_DATA_TYPE_INT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
      return 4;
    case ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
      return 2;
  }
  _CCCL_UNREACHABLE();
}

/***********************************************************************************************************************
 * DLTensor API
 ***********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapDataType __get_tensor_map_data_type(const ::DLTensor* __tensor) noexcept
{
  const auto __type = __tensor->dtype.code;
  switch (__type)
  {
    case ::kDLInt:
      return ::CU_TENSOR_MAP_DATA_TYPE_INT32;
    case ::kDLUInt:
      return ::CU_TENSOR_MAP_DATA_TYPE_UINT32;
    case ::kDLFloat:
      return ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32;
    case ::kDLBfloat:
      return ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    default:
      _CCCL_VERIFY(false, "Unsupported data type");
  }
  _CCCL_UNREACHABLE();
}

[[nodiscard]] _CCCL_HOST_API inline int __get_tensor_map_rank(const ::DLTensor* __tensor) noexcept
{
  const auto __rank = __tensor->ndim;
  _CCCL_VERIFY(__rank > 0 && __rank <= 5, "Unsupported rank");
  // TODO (fbusato): check interleave layout
  return __rank;
}

[[nodiscard]] _CCCL_HOST_API inline void* __get_tensor_address(const ::DLTensor* __tensor) noexcept
{
  using ::cuda::std::size_t;
  const auto __address = __tensor->data;
  _CCCL_VERIFY(__address != nullptr, "Address is null");
  constexpr size_t __tma_required_alignment = 16;
  _CCCL_VERIFY(::cuda::std::is_sufficiently_aligned<__tma_required_alignment>(__address),
               "Address is not sufficiently aligned");
  // TODO (fbusato): check that the address is a valid device address
  return __address;
}

[[nodiscard]] _CCCL_HOST_API inline void* __get_tensor_sizes(const ::DLTensor* __tensor, int __rank) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::uint64_t;
  const auto __sizes = __tensor->shape;
  _CCCL_VERIFY(__sizes != nullptr, "Sizes is null");
  for (int i = 0; i < __rank; ++i)
  {
    [[maybe_unused]] constexpr auto __max_allowd_size = int64_t{1} << 32; // 2^32
    _CCCL_VERIFY(__sizes[i] > 0 && __sizes[i] <= __max_allowd_size, "Size is zero or too large");
  }
  return reinterpret_cast<uint64_t*>(__sizes);
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::array<::cuda::std::uint64_t, 5>
__get_tensor_strides(const ::DLTensor* __tensor, int __rank, ::CUtensorMapDataType __data_type) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::uint64_t;
  ::cuda::std::array<::cuda::std::uint64_t, 5> __strides_array{};
  const auto __strides = __tensor->strides;
  _CCCL_VERIFY(__strides != nullptr, "Strides is null");
  [[maybe_unused]] const auto __data_type_size = ::cuda::__get_data_type_size(__data_type);
  for (int i = 1; i < __rank; ++i)
  {
    [[maybe_unused]] constexpr auto __max_allowd_stride  = int64_t{1} << 40; // 2^40
    [[maybe_unused]] constexpr auto __max_data_type_size = 4;
    _CCCL_VERIFY(__strides[i] <= __max_allowd_stride / __max_data_type_size, "Stride is too large (overflow)");
    [[maybe_unused]] const auto __stride_bytes = __strides[i] * __data_type_size;
    _CCCL_VERIFY(__stride_bytes >= 0 && __stride_bytes <= __max_allowd_stride, "Stride is zero or too large");
    _CCCL_VERIFY(__stride_bytes == 0 || __stride_bytes % 16 == 0, "Stride is not a multiple of 16");
    __strides_array[i] = __stride_bytes;
  }
  return __strides_array;
}

/***********************************************************************************************************************
 * Public API
 ***********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap
create_tma_descr(const ::DLTensor* __tensor, OOBfill __oobfill, L2FetchSize __l2_fetch_size) noexcept
{
  static_assert(DLPACK_MAJOR_VERSION == 1, "DLPACK_MAJOR_VERSION must be 1");
  _CCCL_ASSERT(__tensor != nullptr, "__tensor is null");
  // check compute capability 9.0 or higher.
  ::CUtensorMap __tensor_map{};
  const auto __data_type = ::cuda::__get_tensor_map_data_type(__tensor);
  _CCCL_VERIFY(
    __oobfill == OOBfill::none
      || (__data_type != ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32 && __data_type != ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16),
    "OOBfill::nan is only supported for float32 and bfloat16 data types");
  const auto __rank    = ::cuda::__get_tensor_map_rank(__tensor);
  const auto __address = ::cuda::__get_tensor_address(__tensor);
  const auto __sizes   = ::cuda::__get_tensor_sizes(__tensor, __rank);
  ::cuTensorMapEncodeTiled(&__tensor_map, __tensor); // replace with ::cuda::__driver::__tensorMapEncodeTiled
  return __tensor_map;
}

// #endif // _CCCL_HOST_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

// #endif // _CCCL_HAS_INCLUDE(<dlpack.h>)

#endif // _CUDA___TMA_DESCR_TMA_DESCR_H
