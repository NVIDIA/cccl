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
#include <cuda/std/__algorithm/min.h>
#include <cuda/std/__memory/is_aligned.h>
#include <cuda/std/array>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/limits>
#include <cuda/std/span>

#include <cudaTypedefs.h>

#include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

// #if _CCCL_HOST_COMPILATION()

enum class TmaOOBfill
{
  none,
  nan
};

enum class TmaL2FetchSize
{
  none,
  bytes64,
  bytes128,
  bytes256
};

enum class TmaInterleaveLayout
{
  none,
  bytes16,
  bytes32
};

enum class TmaSwizzle
{
  none,
  bytes32,
  bytes64,
  bytes128,
  bytes128_atom_32B,
  bytes128_atom_32B_flip_8B,
  bytes128_atom_64B
};

/***********************************************************************************************************************
 * Internal Conversion APIs
 ***********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapFloatOOBfill __to_cutensor_map(TmaOOBfill __oobfill) noexcept
{
  switch (__oobfill)
  {
    case TmaOOBfill::none:
      return ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case TmaOOBfill::nan:
      return ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
  }
  _CCCL_UNREACHABLE();
}

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapL2promotion __to_cutensor_map(TmaL2FetchSize __l2_fetch_size) noexcept
{
  switch (__l2_fetch_size)
  {
    case TmaL2FetchSize::none:
      return ::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case TmaL2FetchSize::bytes64:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case TmaL2FetchSize::bytes128:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case TmaL2FetchSize::bytes256:
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
 * DLTensor Internal API
 ***********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapDataType
__get_tensor_map_data_type(const ::DLTensor* __tensor, TmaOOBfill __oobfill) noexcept
{
  const auto __type = __tensor->dtype.code;
  switch (__type)
  {
    case ::kDLInt:
      _CCCL_VERIFY(__oobfill == TmaOOBfill::none,
                   "TmaOOBfill::nan is only supported for float32 and bfloat16 data types");
      return ::CU_TENSOR_MAP_DATA_TYPE_INT32;
    case ::kDLUInt:
      _CCCL_VERIFY(__oobfill == TmaOOBfill::none,
                   "TmaOOBfill::nan is only supported for float32 and bfloat16 data types");
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

[[nodiscard]] _CCCL_HOST_API inline int
__get_tensor_map_rank(const ::DLTensor* __tensor, TmaInterleaveLayout __interleave_layout) noexcept
{
  const auto __rank     = __tensor->ndim;
  const auto __max_rank = (__interleave_layout == TmaInterleaveLayout::none) ? 5 : 3;
  _CCCL_VERIFY(__rank > 0 && __rank <= __max_rank, "Unsupported rank");
  return __rank;
}

[[nodiscard]] _CCCL_HOST_API inline void*
__get_tensor_address(const ::DLTensor* __tensor, TmaInterleaveLayout __interleave_layout) noexcept
{
  using ::cuda::std::size_t;
  _CCCL_VERIFY(__tensor->data != nullptr, "Address is null");
  // note: byte_offset is 0 for most cases.
  const auto __address                      = reinterpret_cast<char*>(__tensor->data) + __tensor->byte_offset;
  const auto __alignment                    = (__interleave_layout == TmaInterleaveLayout::bytes32) ? 32 : 16;
  constexpr size_t __tma_required_alignment = 16;
  _CCCL_VERIFY(::cuda::is_aligned(__address, __alignment), "Address is not sufficiently aligned");
  // TODO (fbusato): check that the address is a valid GPU global address
  return static_cast<void*>(__address);
}

using __tma_sizes_span_t = ::cuda::std::span<::cuda::std::uint64_t>;

[[nodiscard]] _CCCL_HOST_API inline __tma_sizes_span_t
__get_tensor_sizes(const ::DLTensor* __tensor, int __rank) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::uint64_t;
  const auto __tensor_sizes = __tensor->shape;
  _CCCL_VERIFY(__tensor_sizes != nullptr, "Sizes is null");
  for (int i = 0; i < __rank; ++i)
  {
    [[maybe_unused]] constexpr auto __max_allowed_size = int64_t{1} << 32; // 2^32
    _CCCL_VERIFY(__tensor_sizes[i] > 0 && __tensor_sizes[i] <= __max_allowed_size, "Size is zero or too large");
  }
  const auto __tensor_sizes_ptr = reinterpret_cast<::cuda::std::uint64_t*>(__tensor_sizes);
  return __tma_sizes_span_t(__tensor_sizes_ptr, __rank);
}

// 5 dimensions max -> 4 strides
[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::array<::cuda::std::uint64_t, 4> __get_tensor_strides(
  const ::DLTensor* __tensor,
  int __rank,
  ::CUtensorMapDataType __data_type,
  __tma_sizes_span_t __tensor_sizes,
  TmaInterleaveLayout __interleave_layout) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::uint64_t;
  ::cuda::std::array<::cuda::std::uint64_t, 4> __strides_array{};
  const auto __strides = __tensor->strides;
  _CCCL_VERIFY(__strides != nullptr, "Strides is null");
  const auto __data_type_size = ::cuda::__cutensor_data_type_to_size(__data_type);
  _CCCL_VERIFY(__strides[0] == 1, "stride[0] != 1");
  for (int i = 1; i < __rank; ++i)
  {
    [[maybe_unused]] constexpr auto __max_allowed_stride_bytes = int64_t{1} << 40; // 2^40
    _CCCL_VERIFY(__strides[i] >= __tensor_sizes[i - 1] && __strides[i] <= __max_allowed_stride_bytes / __data_type_size,
                 "Stride is too large (overflow)");
    [[maybe_unused]] const auto __stride_bytes = __strides[i] * __data_type_size;
    const auto __alignment                     = (__interleave_layout == TmaInterleaveLayout::bytes32) ? 32 : 16;
    _CCCL_VERIFY(__stride_bytes % __alignment == 0, "Stride is not a multiple of alignment (32 or 16)");
    __strides_array[i - 1] = __stride_bytes;
  }
  return __strides_array;
}

template <::cuda::std::size_t _SpanSize>
_CCCL_HOST_API inline void __check_box_sizes(
  ::cuda::std::span<int, _SpanSize> __box_sizes,
  __tma_sizes_span_t __tensor_sizes,
  TmaInterleaveLayout __interleave_layout,
  TmaSwizzle __swizzle,
  ::CUtensorMapDataType __data_type) noexcept
{
  using ::cuda::std::size_t;
  using ::cuda::std::uint64_t;
  _CCCL_VERIFY(__box_sizes.size() == __tensor_sizes.size(), "Box sizes size mismatch");
  const auto __data_type_size = ::cuda::__cutensor_data_type_to_size(__data_type);
  size_t __total_size         = 1;
  for (int i = 0; i < __tensor_sizes.size(); ++i)
  {
    const auto __max_box_size = ::cuda::std::min(__tensor_sizes[i], uint64_t{256});
    _CCCL_VERIFY(__box_sizes[i] > 0 && __box_sizes[i] <= __max_box_size, "Box size is zero or too large");
    __total_size *= __box_sizes[i] * __data_type_size;
  }
  const auto __inner_dimension_bytes = __box_sizes[0] * __data_type_size;
  if (__interleave_layout == TmaInterleaveLayout::none)
  {
    _CCCL_VERIFY(__inner_dimension_bytes % 16 == 0, "Interleave layout requires 16B alignment");
  }
  if (__swizzle == TmaSwizzle::bytes32)
  {
    _CCCL_VERIFY(__inner_dimension_bytes <= 32, "Swizzle requires a box size less than or equal to 32");
  }
  if (__swizzle == TmaSwizzle::bytes64)
  {
    _CCCL_VERIFY(__inner_dimension_bytes <= 64, "Swizzle requires a box size less than or equal to 64");
  }
  if (__swizzle == TmaSwizzle::bytes128)
  {
    _CCCL_VERIFY(__inner_dimension_bytes <= 128, "Swizzle requires a box size less than or equal to 128");
  }
  //_CCCL_VERIFY(__total_size /*fits in shared memory*/, "Box sizes do not fit in shared memory");
}

template <::cuda::std::size_t _SpanSize>
_CCCL_HOST_API inline void
__check_elem_strides(::cuda::std::span<int, _SpanSize> __elem_strides, __tma_sizes_span_t __tensor_sizes) noexcept
{
  _CCCL_VERIFY(__elem_strides.size() == __tensor_sizes.size(), "Elem strides size mismatch");
  for (int i = 0; i < __tensor_sizes.size(); ++i)
  {
    [[maybe_unused]] const auto __max_elem_stride = ::cuda::std::min(__tensor_sizes[i], uint64_t{8});
    _CCCL_VERIFY(__elem_strides[i] > 0 && __elem_strides[i] <= __max_elem_stride, "Elem stride is out of range");
  }
}

/***************************************************************q********************************************************
 * Public API
 ***********************************************************************************************************************/

template <::cuda::std::size_t _SpanSize>
[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLTensor& __tensor,
  ::cuda::std::span<int, _SpanSize> __box_sizes,
  ::cuda::std::span<int, _SpanSize> __elem_strides,
  TmaInterleaveLayout __interleave_layout = TmaInterleaveLayout::none,
  TmaOOBfill __oobfill                    = TmaOOBfill::none,
  TmaL2FetchSize __l2_fetch_size          = TmaL2FetchSize::none) noexcept
{
  static_assert(DLPACK_MAJOR_VERSION == 1, "DLPACK_MAJOR_VERSION must be 1");
  // check compute capability 9.0 or higher.
  ::CUtensorMap __tensor_map{};
  const auto __data_type    = ::cuda::__get_tensor_map_data_type(__tensor, __oobfill);
  const auto __rank         = ::cuda::__get_tensor_map_rank(__tensor);
  const auto __address      = ::cuda::__get_tensor_address(__tensor);
  const auto __tensor_sizes = ::cuda::__get_tensor_sizes(__tensor, __rank);
  ::cuda::__check_box_sizes(__box_sizes, __tensor_sizes, __interleave_layout, __data_type);

  ::cuTensorMapEncodeTiled(&__tensor_map, __tensor); // replace with ::cuda::__driver::__tensorMapEncodeTiled
  return __tensor_map;
}

// #endif // _CCCL_HOST_COMPILATION()

_CCCL_END_NAMESPACE_CUDA

#include <cuda/std/__cccl/epilogue.h>

// #endif // _CCCL_HAS_INCLUDE(<dlpack.h>)

#endif // _CUDA___TMA_DESCR_TMA_DESCR_H
