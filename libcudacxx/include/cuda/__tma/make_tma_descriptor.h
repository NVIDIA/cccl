//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___TMA_MAKE_TMA_DESCRIPTOR_H
#define _CUDA___TMA_MAKE_TMA_DESCRIPTOR_H

#include <cuda/std/detail/__config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#if _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__memory/is_aligned.h>
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__memory/is_sufficiently_aligned.h>
#  include <cuda/std/array>
#  include <cuda/std/cstddef>
#  include <cuda/std/cstdint>
#  include <cuda/std/span>

#  include <dlpack/dlpack.h>
//
#  include <cuda/std/__cccl/prologue.h>

_CCCL_BEGIN_NAMESPACE_CUDA

static_assert(DLPACK_MAJOR_VERSION == 1, "DLPACK_MAJOR_VERSION must be 1");

/***********************************************************************************************************************
 * Public Enums
 ***********************************************************************************************************************/

enum class tma_oob_fill
{
  none,
  nan
};

enum class tma_l2_fetch_size
{
  none,
  bytes64,
  bytes128,
  bytes256
};

enum class tma_interleave_layout
{
  none,
  bytes16,
  bytes32
};

enum class tma_swizzle
{
  none,
  bytes32,
  bytes64,
  bytes128,
#  if _CCCL_CTK_AT_LEAST(12, 8)
  bytes128_atom_32B,
  bytes128_atom_32B_flip_8B,
  bytes128_atom_64B
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
};

/***********************************************************************************************************************
 * Internal Conversion APIs
 ***********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapFloatOOBfill __to_cutensor_map(tma_oob_fill __oobfill) noexcept
{
  switch (__oobfill)
  {
    case tma_oob_fill::none:
      return ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE;
    case tma_oob_fill::nan:
      return ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA;
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapL2promotion
__to_cutensor_map(tma_l2_fetch_size __l2_fetch_size) noexcept
{
  switch (__l2_fetch_size)
  {
    case tma_l2_fetch_size::none:
      return ::CU_TENSOR_MAP_L2_PROMOTION_NONE;
    case tma_l2_fetch_size::bytes64:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_64B;
    case tma_l2_fetch_size::bytes128:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_128B;
    case tma_l2_fetch_size::bytes256:
      return ::CU_TENSOR_MAP_L2_PROMOTION_L2_256B;
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapInterleave
__to_cutensor_map(tma_interleave_layout __interleave_layout) noexcept
{
  switch (__interleave_layout)
  {
    case tma_interleave_layout::none:
      return ::CU_TENSOR_MAP_INTERLEAVE_NONE;
    case tma_interleave_layout::bytes16:
      return ::CU_TENSOR_MAP_INTERLEAVE_16B;
    case tma_interleave_layout::bytes32:
      return ::CU_TENSOR_MAP_INTERLEAVE_32B;
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapSwizzle __to_cutensor_map(tma_swizzle __swizzle) noexcept
{
  switch (__swizzle)
  {
    case tma_swizzle::none:
      return ::CU_TENSOR_MAP_SWIZZLE_NONE;
    case tma_swizzle::bytes32:
      return ::CU_TENSOR_MAP_SWIZZLE_32B;
    case tma_swizzle::bytes64:
      return ::CU_TENSOR_MAP_SWIZZLE_64B;
    case tma_swizzle::bytes128:
      return ::CU_TENSOR_MAP_SWIZZLE_128B;
#  if _CCCL_CTK_AT_LEAST(12, 8)
    case tma_swizzle::bytes128_atom_32B:
      return ::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B;
    case tma_swizzle::bytes128_atom_32B_flip_8B:
      return ::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B;
    case tma_swizzle::bytes128_atom_64B:
      return ::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B;
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
    default:
      _CCCL_UNREACHABLE();
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::cuda::std::size_t
__to_cutensor_map_size(::CUtensorMapDataType __data_type) noexcept
{
  switch (__data_type)
  {
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT8:
      return 1;
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT16:
    case ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT16:
      return 2;
    case ::CU_TENSOR_MAP_DATA_TYPE_INT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT32:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32:
      return 4;
    case ::CU_TENSOR_MAP_DATA_TYPE_INT64:
    case ::CU_TENSOR_MAP_DATA_TYPE_UINT64:
    case ::CU_TENSOR_MAP_DATA_TYPE_FLOAT64:
#  if _CCCL_CTK_AT_LEAST(12, 8)
    case ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B:
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
      return 8;
    default:
      _CCCL_UNREACHABLE();
  }
}

/***********************************************************************************************************************
 * DLTensor Internal API
 ***********************************************************************************************************************/

template <::cuda::std::size_t _BoxDimSize>
[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapDataType __get_tensor_map_data_type(
  const ::DLTensor& __tensor,
  tma_swizzle __swizzle,
  tma_oob_fill __oobfill,
  const void* __address,
  ::cuda::std::span<const int, _BoxDimSize> __box_sizes) noexcept
{
  const auto __type = __tensor.dtype.code;
  switch (__type)
  {
    case ::kDLInt:
      _CCCL_VERIFY(__tensor.dtype.lanes == 1, "Int data type must be 1 lane");
      _CCCL_VERIFY(__tensor.dtype.bits == 32 || __tensor.dtype.bits == 64, "Int data type must be 32 or 64 bits");
      _CCCL_VERIFY(__oobfill == tma_oob_fill::none,
                   "tma_oob_fill::nan is only supported for floating-point data types");
      return (__tensor.dtype.bits == 32) ? ::CU_TENSOR_MAP_DATA_TYPE_INT32 : ::CU_TENSOR_MAP_DATA_TYPE_INT64;
    case ::kDLUInt: {
      _CCCL_VERIFY(__oobfill == tma_oob_fill::none,
                   "tma_oob_fill::nan is only supported for floating-point data types");
      switch (__tensor.dtype.bits)
      {
#  if _CCCL_CTK_AT_LEAST(12, 8)
        case 4: {
          _CCCL_VERIFY(__tensor.dtype.lanes == 16, "uint4 data type must be 16 lanes");
          const auto __is_aligned_16B                    = ::cuda::std::is_sufficiently_aligned<16>(__address);
          const auto __is_inner_dim_multiple_of_128      = __tensor.shape[__tensor.ndim - 1] % 128 == 0;
          const auto __is_inner_box_size_multiple_of_128 = __box_sizes[__tensor.ndim - 1] % 128 == 0;
          auto __is_swizzle_bytes128                     = __swizzle == tma_swizzle::bytes128;
#    if _CCCL_CTK_AT_LEAST(12, 8)
          __is_swizzle_bytes128 = __is_swizzle_bytes128 || __swizzle == tma_swizzle::bytes128_atom_32B;
#    endif // _CCCL_CTK_AT_LEAST(12, 8)
          bool __are_strides_multiple_of_32 = true;
          for (int __i = 0; __i < __tensor.ndim; ++__i)
          {
            if (__tensor.strides[__i] % 16 != 0) // stride (in bytes) * sizeof(U4) % 16 != 0
            {
              __are_strides_multiple_of_32 = false;
              break;
            }
          }
          return __is_aligned_16B && __is_inner_dim_multiple_of_128 && __is_inner_box_size_multiple_of_128
                  && __is_swizzle_bytes128 && __are_strides_multiple_of_32
                 ? ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B
                 : ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
        }
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
        case 8:
          _CCCL_VERIFY(__tensor.dtype.lanes == 1, "uint8 data type must be 1 lane");
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case 16:
          _CCCL_VERIFY(__tensor.dtype.lanes == 1, "uint16 data type must be 1 lane");
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT16;
        case 32:
          _CCCL_VERIFY(__tensor.dtype.lanes == 1, "uint32 data type must be 1 lane");
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT32;
        case 64:
          _CCCL_VERIFY(__tensor.dtype.lanes == 1, "uint64 data type must be 1 lane");
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT64;
        default:
          _CCCL_VERIFY(false, "UInt data type must be 4, 8, 16, 32, or 64 bits");
          _CCCL_UNREACHABLE();
      }
    }
    case ::kDLFloat:
      _CCCL_VERIFY(__tensor.dtype.lanes == 1, "Float data type must be 1 lane");
      _CCCL_VERIFY(__tensor.dtype.bits == 16 || __tensor.dtype.bits == 32 || __tensor.dtype.bits == 64,
                   "Float data type must be 16, 32, or 64 bits");
      return (__tensor.dtype.bits == 16) ? ::CU_TENSOR_MAP_DATA_TYPE_FLOAT16
           : (__tensor.dtype.bits == 32)
             ? ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32
             : ::CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    case ::kDLBfloat:
      _CCCL_VERIFY(__tensor.dtype.lanes == 1, "Bfloat16 data type must be 1 lane");
      _CCCL_VERIFY(__tensor.dtype.bits == 16, "Bfloat16 data type must be 16 bits");
      return ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    default:
      _CCCL_VERIFY(false, "Unsupported data type");
  }
  _CCCL_UNREACHABLE();
}

[[nodiscard]] _CCCL_HOST_API inline int
__get_tensor_map_rank(const ::DLTensor& __tensor, tma_interleave_layout __interleave_layout) noexcept
{
  const auto __rank         = __tensor.ndim;
  constexpr auto __max_rank = 5;
  _CCCL_VERIFY(__rank > 0 && __rank <= __max_rank, "tensor.ndim (rank) must be between 1 and 5");
  if (__interleave_layout != tma_interleave_layout::none)
  {
    _CCCL_VERIFY(__rank >= 3, "tensor.ndim (rank) must be greater than or equal to 3 for interleaved layout");
  }
  return __rank;
}

[[nodiscard]] _CCCL_HOST_API inline void*
__get_tensor_address(const ::DLTensor& __tensor, tma_interleave_layout __interleave_layout) noexcept
{
  using ::cuda::std::size_t;
  _CCCL_VERIFY(__tensor.data != nullptr, "Address is null");
  // note: byte_offset is 0 for most cases.
  const auto __address   = reinterpret_cast<char*>(__tensor.data) + __tensor.byte_offset;
  const auto __alignment = (__interleave_layout == tma_interleave_layout::bytes32) ? 32 : 16;
  _CCCL_VERIFY(::cuda::is_aligned(__address, __alignment), "tensor.data (address) is not sufficiently aligned");
  // TODO (fbusato): check that the address is a valid GPU global address
  return static_cast<void*>(__address);
}

using __tma_sizes_array_t        = ::cuda::std::array<::cuda::std::uint64_t, 5>;
using __tma_strides_array_t      = ::cuda::std::array<::cuda::std::uint64_t, 4>; // first stride is implicit stride = 1
using __tma_box_sizes_array_t    = ::cuda::std::array<::cuda::std::uint32_t, 5>;
using __tma_elem_strides_array_t = ::cuda::std::array<::cuda::std::uint32_t, 5>;

[[nodiscard]] _CCCL_HOST_API inline __tma_sizes_array_t
__get_tensor_sizes(const ::DLTensor& __tensor, int __rank, ::CUtensorMapDataType __data_type) noexcept
{
  using ::cuda::std::int64_t;
  using ::cuda::std::uint64_t;
  __tma_sizes_array_t __tensor_sizes_array{};
  const auto __tensor_sizes = __tensor.shape;
  _CCCL_VERIFY(__tensor_sizes != nullptr, "Sizes is null");
#  if _CCCL_CTK_AT_LEAST(12, 8)
  if (__data_type == ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN16B)
  {
    _CCCL_VERIFY(__tensor_sizes[__rank - 1] % 16 == 0,
                 "The innermost dimension size must be a multiple of 16 for 16U4_ALIGN16B");
  }
  if (__data_type == ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B)
  {
    _CCCL_VERIFY(__tensor_sizes[__rank - 1] % 2 == 0,
                 "The innermost dimension size must be a multiple of 2 for 16U4_ALIGN8B");
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
  for (int __i = 0; __i < __rank; ++__i)
  {
    constexpr auto __max_allowed_size = int64_t{1} << 32; // 2^32
    _CCCL_VERIFY(__tensor_sizes[__i] > 0 && __tensor_sizes[__i] <= __max_allowed_size, "Size is zero or too large");
    __tensor_sizes_array[__rank - 1 - __i] = __tensor_sizes[__i];
  }
  return __tensor_sizes_array;
}

// DLPack assumes row-major convention for sizes and strides, where the fastest changing dimension is the last one.
// cuTensorMap assumes column-major convention for sizes and strides, where the fastest changing dimension is the first
// one.
[[nodiscard]] _CCCL_HOST_API inline __tma_strides_array_t __get_tensor_strides(
  const ::DLTensor& __tensor,
  int __rank,
  ::CUtensorMapDataType __data_type,
  const __tma_sizes_array_t& __tensor_sizes,
  tma_interleave_layout __interleave_layout) noexcept
{
  using ::cuda::std::int64_t;
  const auto __input_strides = __tensor.strides;
  const auto __input_sizes   = __tensor.shape;
  __tma_strides_array_t __output_strides{}; // 4: max rank - 1
  const auto __data_type_size               = ::cuda::__to_cutensor_map_size(__data_type);
  const auto __alignment                    = (__interleave_layout == tma_interleave_layout::bytes32) ? 32 : 16;
  constexpr auto __max_allowed_stride_bytes = int64_t{1} << 40; // 2^40
  const auto __max_allowed_stride_count     = __max_allowed_stride_bytes / static_cast<int64_t>(__data_type_size);
  int64_t __cumulative_size                 = 1;
  if (__input_strides == nullptr)
  {
    for (int __i = 0; __i < __rank - 1; ++__i)
    {
      __cumulative_size *= __tensor_sizes[__i];
      // TODO(fbusato): check mul overflow
      const auto __stride_bytes = __cumulative_size * __data_type_size;
      _CCCL_VERIFY(__stride_bytes % __alignment == 0, "Stride in bytes is not a multiple of the alignment (32 or 16)");
      _CCCL_VERIFY(__cumulative_size < __max_allowed_stride_count, "Stride in bytes is greater than 2^40");
      __output_strides[__i] = __stride_bytes;
    }
    return __output_strides;
  }
  // TMA ignores the innermost stride (always 1).
  for (int __i = __rank - 2; __i >= 0; --__i)
  {
    _CCCL_VERIFY(__input_strides[__i] < __max_allowed_stride_count, "Stride is greater than 2^40 / sizeof(data_type)");
    const auto __next_stride = (__i == __rank - 2) ? 1 : __input_strides[__i + 1];
    // TODO(fbusato): check mul overflow
    _CCCL_VERIFY(__input_strides[__i] == 0 || (__input_strides[__i] >= __input_sizes[__i + 1] * __next_stride),
                 "Stride must be 0 or greater than or equal to the product of the next stride and the size of the next "
                 "dimension");
    const auto __input_stride_bytes = __input_strides[__i] * __data_type_size;
    _CCCL_VERIFY(__input_stride_bytes % __alignment == 0,
                 "Stride in bytes is not a multiple of the alignment (32 or 16)");
    __output_strides[__rank - 2 - __i] = __input_stride_bytes;
  }
  return __output_strides;
}

_CCCL_HOST_API inline void __check_swizzle(tma_interleave_layout __interleave_layout, tma_swizzle __swizzle) noexcept
{
  if (__interleave_layout == tma_interleave_layout::bytes32)
  {
    _CCCL_VERIFY(__swizzle == tma_swizzle::bytes32, "ma_interleave_layout::bytes32 requires tma_swizzle::bytes32");
  }
}

template <::cuda::std::size_t _BoxDimSize>
_CCCL_HOST_API inline __tma_box_sizes_array_t __get_box_sizes(
  ::cuda::std::span<const int, _BoxDimSize> __box_sizes,
  const __tma_sizes_array_t& __tensor_sizes,
  int __rank,
  tma_interleave_layout __interleave_layout,
  tma_swizzle __swizzle,
  ::CUtensorMapDataType __data_type) noexcept
{
  using ::cuda::std::size_t;
  using ::cuda::std::uint64_t;
  __tma_box_sizes_array_t __box_sizes_array{};
  _CCCL_VERIFY(__box_sizes.size() == __rank, "Box sizes size mismatch");
  const auto __data_type_size          = ::cuda::__to_cutensor_map_size(__data_type);
  [[maybe_unused]] size_t __total_size = 1;
  for (int __i = 0; __i < __rank; ++__i)
  {
    const auto __max_box_size = static_cast<int>(::cuda::std::min(__tensor_sizes[__i], uint64_t{256}));
    const auto __box_size     = __box_sizes[__rank - 1 - __i];
    _CCCL_VERIFY(__box_size > 0 && __box_size <= __max_box_size,
                 "box_sizes[i] must be between 1 and min(tensor.shape[rank - 1 - i], 256)");
    __total_size *= __box_size * __data_type_size;
    __box_sizes_array[__i] = __box_size;
  }
  const auto __inner_dimension_bytes = __box_sizes[__rank - 1] * __data_type_size;
  if (__interleave_layout == tma_interleave_layout::none)
  {
    _CCCL_VERIFY(__inner_dimension_bytes % 16 == 0,
                 "tma_interleave_layout::none requires innermost dimension in bytes to be a multiple of 16");
    if (__swizzle == tma_swizzle::bytes32)
    {
      _CCCL_VERIFY(__inner_dimension_bytes <= 32, "tma_swizzle::bytes32 requires a box size less than or equal to 32");
    }
    if (__swizzle == tma_swizzle::bytes64)
    {
      _CCCL_VERIFY(__inner_dimension_bytes <= 64, "tma_swizzle::bytes64 requires a box size less than or equal to 64");
    }
    if (__swizzle == tma_swizzle::bytes128)
    {
      _CCCL_VERIFY(__inner_dimension_bytes <= 128,
                   "tma_swizzle::bytes128 requires a box size less than or equal to 128");
    }
  }
  // TODO(fbusato) _CCCL_VERIFY(__total_size /*fits in shared memory*/, "Box sizes do not fit in shared memory");
  return __box_sizes_array;
}

template <::cuda::std::size_t _ElemStrideSize>
_CCCL_HOST_API inline __tma_elem_strides_array_t __get_elem_strides(
  ::cuda::std::span<const int, _ElemStrideSize> __elem_strides,
  const __tma_sizes_array_t& __tensor_sizes,
  int __rank,
  tma_interleave_layout __interleave_layout) noexcept
{
  using ::cuda::std::size_t;
  using ::cuda::std::uint64_t;
  _CCCL_VERIFY(__elem_strides.size() == static_cast<size_t>(__rank), "Elem strides size mismatch");
  __tma_elem_strides_array_t __elem_strides_array{1};
  const int __init_index = (__interleave_layout == tma_interleave_layout::none) ? 1 : 0;
  for (int __i = __init_index; __i < __rank; ++__i)
  {
    const auto __max_elem_stride = static_cast<int>(::cuda::std::min(__tensor_sizes[__i], uint64_t{8}));
    const auto __elem_stride     = __elem_strides[__rank - 1 - __i];
    _CCCL_VERIFY(__elem_stride > 0 && __elem_stride <= __max_elem_stride, "Elem stride is out of range");
    __elem_strides_array[__i] = __elem_stride;
  }
  return __elem_strides_array;
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

template <::cuda::std::size_t _BoxDimSize, ::cuda::std::size_t _ElemStrideSize>
[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLTensor& __tensor,
  ::cuda::std::span<const int, _BoxDimSize> __box_sizes,
  ::cuda::std::span<const int, _ElemStrideSize> __elem_strides,
  tma_interleave_layout __interleave_layout = tma_interleave_layout::none,
  tma_swizzle __swizzle                     = tma_swizzle::none,
  tma_l2_fetch_size __l2_fetch_size         = tma_l2_fetch_size::none,
  tma_oob_fill __oobfill                    = tma_oob_fill::none) noexcept
{
  _CCCL_VERIFY(__box_sizes.size() == __elem_strides.size(), "Box sizes and elem strides size mismatch");
  _CCCL_VERIFY(__tensor.device.device_type == ::kDLCUDA, "Device type must be kDLCUDA");
  _CCCL_VERIFY(__tensor.device.device_id >= 0, "Device ID must be a valid GPU device ordinal");
  auto __current_device = ::cuda::__driver::__deviceGet(static_cast<int>(__tensor.device.device_id));
  auto __compute_capability =
    ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR, __current_device);
  _CCCL_VERIFY(__compute_capability >= 9, "Compute capability 9.0 or higher is required");
  const auto __address   = ::cuda::__get_tensor_address(__tensor, __interleave_layout);
  const auto __data_type = ::cuda::__get_tensor_map_data_type(__tensor, __swizzle, __oobfill, __address, __box_sizes);
  const auto __rank      = ::cuda::__get_tensor_map_rank(__tensor, __interleave_layout);
  const auto __tensor_sizes = ::cuda::__get_tensor_sizes(__tensor, __rank, __data_type);
  const auto __input_strides =
    ::cuda::__get_tensor_strides(__tensor, __rank, __data_type, __tensor_sizes, __interleave_layout);
  const auto __raw_interleave_layout = ::cuda::__to_cutensor_map(__interleave_layout);
  const auto __raw_swizzle           = ::cuda::__to_cutensor_map(__swizzle);
  const auto __raw_l2_fetch_size     = ::cuda::__to_cutensor_map(__l2_fetch_size);
  const auto __raw_oobfill           = ::cuda::__to_cutensor_map(__oobfill);
  ::cuda::__check_swizzle(__interleave_layout, __swizzle);
  const auto __raw_elem_strides =
    ::cuda::__get_elem_strides(__elem_strides, __tensor_sizes, __rank, __interleave_layout);
  const auto __raw_box_sizes =
    ::cuda::__get_box_sizes(__box_sizes, __tensor_sizes, __rank, __interleave_layout, __swizzle, __data_type);
  return ::cuda::__driver::__tensorMapEncodeTiled(
    __data_type,
    __rank,
    __address,
    __tensor_sizes.data(),
    __input_strides.data(),
    __raw_box_sizes.data(),
    __raw_elem_strides.data(),
    __raw_interleave_layout,
    __raw_swizzle,
    __raw_l2_fetch_size,
    __raw_oobfill);
}

template <::cuda::std::size_t _BoxDimSize>
[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLTensor& __tensor,
  ::cuda::std::span<const int, _BoxDimSize> __box_sizes,
  tma_interleave_layout __interleave_layout = tma_interleave_layout::none,
  tma_swizzle __swizzle                     = tma_swizzle::none,
  tma_l2_fetch_size __l2_fetch_size         = tma_l2_fetch_size::none,
  tma_oob_fill __oobfill                    = tma_oob_fill::none) noexcept
{
  using ::cuda::std::size_t;
  const auto __rank                       = ::cuda::__get_tensor_map_rank(__tensor, __interleave_layout);
  constexpr int __elem_strides_storage[5] = {1, 1, 1, 1, 1};
  cuda::std::span<const int> __elem_strides{__elem_strides_storage, static_cast<size_t>(__rank)};
  return ::cuda::make_tma_descriptor(
    __tensor, __box_sizes, __elem_strides, __interleave_layout, __swizzle, __l2_fetch_size, __oobfill);
}

template <::cuda::std::size_t _BoxDimSize, ::cuda::std::size_t _ElemStrideSize>
[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLManagedTensor& __tensor,
  ::cuda::std::span<const int, _BoxDimSize> __box_sizes,
  ::cuda::std::span<const int, _ElemStrideSize> __elem_strides,
  tma_interleave_layout __interleave_layout = tma_interleave_layout::none,
  tma_swizzle __swizzle                     = tma_swizzle::none,
  tma_l2_fetch_size __l2_fetch_size         = tma_l2_fetch_size::none,
  tma_oob_fill __oobfill                    = tma_oob_fill::none) noexcept
{
  return ::cuda::make_tma_descriptor(
    __tensor.dl_tensor, __box_sizes, __elem_strides, __interleave_layout, __swizzle, __l2_fetch_size, __oobfill);
}

template <::cuda::std::size_t _BoxDimSize>
[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLManagedTensor& __tensor,
  ::cuda::std::span<const int, _BoxDimSize> __box_sizes,
  tma_interleave_layout __interleave_layout = tma_interleave_layout::none,
  tma_swizzle __swizzle                     = tma_swizzle::none,
  tma_l2_fetch_size __l2_fetch_size         = tma_l2_fetch_size::none,
  tma_oob_fill __oobfill                    = tma_oob_fill::none) noexcept
{
  return ::cuda::make_tma_descriptor(
    __tensor.dl_tensor, __box_sizes, __interleave_layout, __swizzle, __l2_fetch_size, __oobfill);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_INCLUDE("dlpack/dlpack.h")
#endif // _CUDA___TMA_MAKE_TMA_DESCRIPTOR_H
