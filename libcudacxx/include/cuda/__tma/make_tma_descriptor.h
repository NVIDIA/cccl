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

#if _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)

#  include <cuda/__driver/driver_api.h>
#  include <cuda/__memory/is_aligned.h>
#  include <cuda/__memory/is_pointer_accessible.h>
#  include <cuda/devices> // sub headers cause circular dependency
#  include <cuda/std/__algorithm/min.h>
#  include <cuda/std/__cstddef/types.h>
#  include <cuda/std/__exception/exception_macros.h>
#  include <cuda/std/array>
#  include <cuda/std/cstdint>
#  include <cuda/std/span>

#  include <stdexcept>

#  include <driver_types.h>

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
  none = ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE,
  nan  = ::CU_TENSOR_MAP_FLOAT_OOB_FILL_NAN_REQUEST_ZERO_FMA
};

enum class tma_l2_fetch_size
{
  none     = ::CU_TENSOR_MAP_L2_PROMOTION_NONE,
  bytes64  = ::CU_TENSOR_MAP_L2_PROMOTION_L2_64B,
  bytes128 = ::CU_TENSOR_MAP_L2_PROMOTION_L2_128B,
  bytes256 = ::CU_TENSOR_MAP_L2_PROMOTION_L2_256B
};

enum class tma_interleave_layout
{
  none    = ::CU_TENSOR_MAP_INTERLEAVE_NONE,
  bytes16 = ::CU_TENSOR_MAP_INTERLEAVE_16B,
  bytes32 = ::CU_TENSOR_MAP_INTERLEAVE_32B
};

enum class tma_swizzle
{
  none     = ::CU_TENSOR_MAP_SWIZZLE_NONE,
  bytes32  = ::CU_TENSOR_MAP_SWIZZLE_32B,
  bytes64  = ::CU_TENSOR_MAP_SWIZZLE_64B,
  bytes128 = ::CU_TENSOR_MAP_SWIZZLE_128B,
#  if _CCCL_CTK_AT_LEAST(12, 8)
  bytes128_atom_32B         = ::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B,
  bytes128_atom_32B_flip_8B = ::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_32B_FLIP_8B,
  bytes128_atom_64B         = ::CU_TENSOR_MAP_SWIZZLE_128B_ATOM_64B,
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

/***********************************************************************************************************************
 * DLTensor Internal API
 ***********************************************************************************************************************/

_CCCL_HOST_API inline void __check_device(const ::DLTensor& __tensor, tma_swizzle __swizzle)
{
  if (__tensor.device.device_type != ::kDLCUDA && __tensor.device.device_type != ::kDLCUDAManaged)
  {
    _CCCL_THROW(::std::invalid_argument{"Device type must be kDLCUDA or kDLCUDAManaged"});
  }
  const auto __current_device     = ::cuda::device_ref(__tensor.device.device_id);
  const auto __compute_capability = __current_device.attribute<::cudaDevAttrComputeCapabilityMajor>();
  if (__compute_capability < 9)
  {
    _CCCL_THROW(::std::runtime_error{"Compute capability 9.0 or higher is required"});
  }
  if (__compute_capability == 9)
  {
#  if _CCCL_CTK_AT_LEAST(12, 8)
    if (__swizzle == tma_swizzle::bytes128_atom_32B || __swizzle == tma_swizzle::bytes128_atom_32B_flip_8B
        || __swizzle == tma_swizzle::bytes128_atom_64B)
    {
      _CCCL_THROW(::std::invalid_argument{"tma_swizzle::bytes128_atom* are not supported with compute capability 9"});
    }
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
    if (__tensor.dtype.code == ::kDLUInt && __tensor.dtype.bits == 4 && __tensor.dtype.lanes == 16)
    {
      _CCCL_THROW(::std::invalid_argument{"U4x16 is not supported with compute capability 9"});
    }
  }
}

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMapDataType
__get_tensor_map_data_type(const ::DLTensor& __tensor, tma_oob_fill __oobfill)
{
  const auto __type = __tensor.dtype.code;
  switch (__type)
  {
    case ::kDLInt:
      if (__tensor.dtype.lanes != 1)
      {
        _CCCL_THROW(::std::invalid_argument{"Int data type must be 1 lane"});
      }
      if (__tensor.dtype.bits != 32 && __tensor.dtype.bits != 64)
      {
        _CCCL_THROW(::std::invalid_argument{"Int data type must be 32 or 64 bits"});
      }
      if (__oobfill != tma_oob_fill::none)
      {
        _CCCL_THROW(::std::invalid_argument{"tma_oob_fill::nan is only supported for floating-point data types"});
      }
      return (__tensor.dtype.bits == 32) ? ::CU_TENSOR_MAP_DATA_TYPE_INT32 : ::CU_TENSOR_MAP_DATA_TYPE_INT64;
    case ::kDLUInt: {
      if (__oobfill != tma_oob_fill::none)
      {
        _CCCL_THROW(::std::invalid_argument{"tma_oob_fill::nan is only supported for floating-point data types"});
      }
      switch (__tensor.dtype.bits)
      {
        case 4: {
          if (__tensor.dtype.lanes != 16)
          {
            _CCCL_THROW(::std::invalid_argument{"uint4 data type must be 16 lanes"});
          }
#  if _CCCL_CTK_AT_LEAST(12, 8)
          return ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
#  else
          _CCCL_THROW(::std::invalid_argument{"U4x16 is not supported for compute capability 9"});
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
        }
        case 8:
          if (__tensor.dtype.lanes != 1)
          {
            _CCCL_THROW(::std::invalid_argument{"uint8 data type must be 1 lane"});
          }
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT8;
        case 16:
          if (__tensor.dtype.lanes != 1)
          {
            _CCCL_THROW(::std::invalid_argument{"uint16 data type must be 1 lane"});
          }
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT16;
        case 32:
          if (__tensor.dtype.lanes != 1)
          {
            _CCCL_THROW(::std::invalid_argument{"uint32 data type must be 1 lane"});
          }
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT32;
        case 64:
          if (__tensor.dtype.lanes != 1)
          {
            _CCCL_THROW(::std::invalid_argument{"uint64 data type must be 1 lane"});
          }
          return ::CU_TENSOR_MAP_DATA_TYPE_UINT64;
        default:
          _CCCL_THROW(::std::invalid_argument{"UInt data type must be 4, 8, 16, 32, or 64 bits"});
      }
    }
    case ::kDLFloat:
      if (__tensor.dtype.lanes != 1)
      {
        _CCCL_THROW(::std::invalid_argument{"Float data type must be 1 lane"});
      }
      if (__tensor.dtype.bits != 16 && __tensor.dtype.bits != 32 && __tensor.dtype.bits != 64)
      {
        _CCCL_THROW(::std::invalid_argument{"Float data type must be 16, 32, or 64 bits"});
      }
      return (__tensor.dtype.bits == 16) ? ::CU_TENSOR_MAP_DATA_TYPE_FLOAT16
           : (__tensor.dtype.bits == 32)
             ? ::CU_TENSOR_MAP_DATA_TYPE_FLOAT32
             : ::CU_TENSOR_MAP_DATA_TYPE_FLOAT64;
    case ::kDLBfloat:
      if (__tensor.dtype.lanes != 1)
      {
        _CCCL_THROW(::std::invalid_argument{"Bfloat16 data type must be 1 lane"});
      }
      if (__tensor.dtype.bits != 16)
      {
        _CCCL_THROW(::std::invalid_argument{"Bfloat16 data type must be 16 bits"});
      }
      return ::CU_TENSOR_MAP_DATA_TYPE_BFLOAT16;
    case ::kDLBool:
      if (__oobfill != tma_oob_fill::none)
      {
        _CCCL_THROW(::std::invalid_argument{"tma_oob_fill::nan is only supported for floating-point data types"});
      }
      [[fallthrough]];
    case ::kDLFloat8_e3m4:
    case ::kDLFloat8_e4m3:
    case ::kDLFloat8_e4m3b11fnuz:
    case ::kDLFloat8_e4m3fn:
    case ::kDLFloat8_e4m3fnuz:
    case ::kDLFloat8_e5m2:
    case ::kDLFloat8_e5m2fnuz:
    case ::kDLFloat8_e8m0fnu:
      return ::CU_TENSOR_MAP_DATA_TYPE_UINT8;
    case ::kDLFloat4_e2m1fn:
      if (__tensor.dtype.lanes != 16)
      {
        _CCCL_THROW(::std::invalid_argument{"Float4_e2m1fn data type must be 16 lanes"});
      }
      if (__tensor.dtype.bits != 4)
      {
        _CCCL_THROW(::std::invalid_argument{"Float4_e2m1fn data type must be 4 bits"});
      }
#  if _CCCL_CTK_AT_LEAST(12, 8)
      return ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B;
#  else
      _CCCL_THROW(::std::invalid_argument{"U4x16 (Float4_e2m1fn) is not supported for compute capability 9"});
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
    default:
      _CCCL_THROW(::std::invalid_argument{"Unsupported data type"});
  }
  _CCCL_UNREACHABLE();
}

[[nodiscard]] _CCCL_HOST_API inline int
__get_tensor_map_rank(const ::DLTensor& __tensor, tma_interleave_layout __interleave_layout)
{
  const auto __rank         = __tensor.ndim;
  constexpr auto __max_rank = 5;
  if (__rank <= 0 || __rank > __max_rank)
  {
    _CCCL_THROW(::std::invalid_argument{"tensor.ndim (rank) must be between 1 and 5"});
  }
  if (__interleave_layout != tma_interleave_layout::none)
  {
    if (__rank <= 3)
    {
      _CCCL_THROW(::std::invalid_argument{"tensor.ndim (rank) must be greater than or equal to 3 for interleaved "
                                          "layout"});
    }
  }
  return __rank;
}

[[nodiscard]] _CCCL_HOST_API inline void*
__get_tensor_address(const ::DLTensor& __tensor, tma_interleave_layout __interleave_layout)
{
  using ::cuda::std::size_t;
  if (__tensor.data == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument{"Address is null"});
  }
  // note: byte_offset is 0 for most cases.
  const auto __address   = reinterpret_cast<char*>(__tensor.data) + __tensor.byte_offset;
  const auto __alignment = (__interleave_layout == tma_interleave_layout::bytes32) ? 32 : 16;
  if (!::cuda::is_aligned(__address, __alignment))
  {
    _CCCL_THROW(::std::invalid_argument{"tensor.data (address) is not sufficiently aligned"});
  }
  if (!::cuda::is_device_accessible(__address, __tensor.device.device_id))
  {
    _CCCL_THROW(::std::invalid_argument{"Address is not a valid GPU global address"});
  }
  return static_cast<void*>(__address);
}

using __tma_sizes_array_t        = ::cuda::std::array<::cuda::std::uint64_t, 5>;
using __tma_strides_array_t      = ::cuda::std::array<::cuda::std::uint64_t, 4>; // inner stride is implicit = 1
using __tma_box_sizes_array_t    = ::cuda::std::array<::cuda::std::uint32_t, 5>;
using __tma_elem_strides_array_t = ::cuda::std::array<::cuda::std::uint32_t, 5>;

[[nodiscard]] _CCCL_HOST_API inline __tma_sizes_array_t
__get_tensor_sizes(const ::DLTensor& __tensor, int __rank, ::CUtensorMapDataType __data_type)
{
  using ::cuda::std::int64_t;
  using ::cuda::std::uint64_t;
  __tma_sizes_array_t __tensor_sizes_array{};
  const auto __tensor_sizes = __tensor.shape;
  if (__tensor.shape == nullptr)
  {
    _CCCL_THROW(::std::invalid_argument{"__tensor.shape is null"});
  }
#  if _CCCL_CTK_AT_LEAST(12, 8)
  if (__data_type == ::CU_TENSOR_MAP_DATA_TYPE_16U4_ALIGN8B)
  {
    if (__tensor_sizes[__rank - 1] % 2 != 0)
    {
      _CCCL_THROW(::std::invalid_argument{"The innermost tensor dimension size must be a multiple of 2 for U4x16 or "
                                          "Float4_e2m1fn"});
    }
  }
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
  for (int __i = 0; __i < __rank; ++__i)
  {
    constexpr auto __max_allowed_size = int64_t{1} << 32; // 2^32
    if (__tensor_sizes[__i] <= 0 || __tensor_sizes[__i] > __max_allowed_size)
    {
      _CCCL_THROW(::std::invalid_argument{"tensor.shape[i] must be greater than 0 and less than or equal to 2^32"});
    }
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
  tma_interleave_layout __interleave_layout)
{
  using ::cuda::std::int64_t;
  __tma_strides_array_t __output_strides{1}; // inner stride is implicit = 1
  const auto __input_strides                = __tensor.strides;
  const auto __input_sizes                  = __tensor.shape;
  const auto __alignment                    = (__interleave_layout == tma_interleave_layout::bytes32) ? 32 : 16;
  constexpr auto __max_allowed_stride_bytes = int64_t{1} << 40; // 2^40
  int64_t __cumulative_size                 = 1;
  if (__input_strides == nullptr)
  {
    for (int __i = 0; __i < __rank - 1; ++__i)
    {
      // TODO(fbusato): check mul overflow
      __cumulative_size *= __tensor_sizes[__i];
      const auto __stride_bytes = ::cuda::__driver::__cutensormap_size_bytes(__cumulative_size, __data_type);
      if (__stride_bytes % __alignment != 0)
      {
        _CCCL_THROW(::std::invalid_argument{"Stride in bytes is not a multiple of the alignment (32 or 16)"});
      }
      if (__stride_bytes >= __max_allowed_stride_bytes)
      {
        _CCCL_THROW(::std::invalid_argument{"Stride in bytes is greater than or equal to 2^40"});
      }
      __output_strides[__i] = __stride_bytes;
    }
    return __output_strides;
  }
  // TMA ignores the innermost stride (always 1).
  for (int __i = __rank - 2; __i >= 0; --__i)
  {
    const auto __next_stride = (__i == __rank - 2) ? int64_t{1} : __input_strides[__i + 1];
    // TODO(fbusato): check mul overflow
    if (__input_strides[__i] != 0 && (__input_strides[__i] < __input_sizes[__i + 1] * __next_stride))
    {
      _CCCL_THROW(::std::invalid_argument{
        "Stride must be 0 or greater than or equal to the product of the next "
        "stride and the size of the next "
        "dimension"});
    }
    const auto __input_stride_bytes = ::cuda::__driver::__cutensormap_size_bytes(__input_strides[__i], __data_type);
    if (__input_stride_bytes % __alignment != 0)
    {
      _CCCL_THROW(::std::invalid_argument{"Stride in bytes is not a multiple of the alignment (32 or 16)"});
    }
    if (__input_stride_bytes >= __max_allowed_stride_bytes)
    {
      _CCCL_THROW(::std::invalid_argument{"Stride in bytes is greater than or equal to 2^40"});
    }
    __output_strides[__rank - 2 - __i] = __input_stride_bytes;
  }
  return __output_strides;
}

[[nodiscard]]
_CCCL_HOST_API inline __tma_box_sizes_array_t __get_box_sizes(
  ::cuda::std::span<const int> __box_sizes,
  const __tma_sizes_array_t& __tensor_sizes,
  int __rank,
  tma_interleave_layout __interleave_layout,
  tma_swizzle __swizzle,
  ::CUtensorMapDataType __data_type,
  int __device_id)
{
  using ::cuda::std::size_t;
  using ::cuda::std::uint64_t;
  __tma_box_sizes_array_t __box_sizes_output{};
  if (__box_sizes.size() != static_cast<size_t>(__rank))
  {
    _CCCL_THROW(::std::invalid_argument{"Box sizes size mismatch"});
  }
  size_t __total_size = 1;
  for (int __i = 0; __i < __rank; ++__i)
  {
    const auto __max_box_size = static_cast<int>(::cuda::std::min(__tensor_sizes[__i], uint64_t{256}));
    const auto __box_size     = __box_sizes[__rank - 1 - __i];
    if (__box_size <= 0 || __box_size > __max_box_size)
    {
      _CCCL_THROW(::std::invalid_argument{"box_sizes[i] must be between 1 and min(tensor.shape[rank - 1 - i], 256)"});
    }
    __total_size *= __box_size;
    __box_sizes_output[__i] = __box_size;
  }
  const auto __inner_dimension_bytes =
    ::cuda::__driver::__cutensormap_size_bytes(__box_sizes_output[__rank - 1], __data_type);
  if (__interleave_layout == tma_interleave_layout::none)
  {
    if (__inner_dimension_bytes % 16 != 0)
    {
      _CCCL_THROW(::std::invalid_argument{
        "tma_interleave_layout::none requires box_sizes innermost dimension "
        "(box_sizes[__rank - 1]) in bytes to be a "
        "multiple of 16"});
    }
    if (__swizzle == tma_swizzle::bytes32)
    {
      if (__inner_dimension_bytes > 32)
      {
        _CCCL_THROW(::std::invalid_argument{
          "tma_swizzle::bytes32 requires box_sizes innermost dimension "
          "(box_sizes[__rank - 1]) in bytes to be "
          "less than or equal to 32"});
      }
    }
    if (__swizzle == tma_swizzle::bytes64)
    {
      if (__inner_dimension_bytes > 64)
      {
        _CCCL_THROW(::std::invalid_argument{
          "tma_swizzle::bytes64 requires box_sizes innermost dimension "
          "(box_sizes[__rank - 1]) in bytes to be "
          "less than or equal to 64"});
      }
    }
    if (__swizzle == tma_swizzle::bytes128)
    {
      if (__inner_dimension_bytes > 128)
      {
        _CCCL_THROW(::std::invalid_argument{
          "tma_swizzle::bytes128 requires box_sizes innermost dimension "
          "(box_sizes[__rank - 1]) in bytes to be "
          "less than or equal to 128"});
      }
    }
#  if _CCCL_CTK_AT_LEAST(12, 8)
    if (__swizzle == tma_swizzle::bytes128_atom_32B || __swizzle == tma_swizzle::bytes128_atom_32B_flip_8B
        || __swizzle == tma_swizzle::bytes128_atom_64B)
    {
      if (__inner_dimension_bytes > 128)
      {
        _CCCL_THROW(::std::invalid_argument{
          "tma_swizzle::bytes128_atom* requires box_sizes innermost dimension (box_sizes[__rank - "
          "1]) in bytes to be less than or equal to 128"});
      }
    }
#  endif // _CCCL_CTK_AT_LEAST(12, 8)
  }
  const auto __max_shmem =
    ::cuda::__driver::__deviceGetAttribute(::CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK_OPTIN, __device_id);
  if (::cuda::__driver::__cutensormap_size_bytes(__total_size, __data_type) > static_cast<size_t>(__max_shmem))
  {
    _CCCL_THROW(::std::invalid_argument{"Box sizes do not fit in shared memory"});
  }
  return __box_sizes_output;
}

[[nodiscard]]
_CCCL_HOST_API inline __tma_elem_strides_array_t __get_elem_strides(
  ::cuda::std::span<const int> __elem_strides,
  const __tma_sizes_array_t& __tensor_sizes,
  int __rank,
  tma_interleave_layout __interleave_layout)
{
  using ::cuda::std::size_t;
  using ::cuda::std::uint64_t;
  if (__elem_strides.size() != static_cast<size_t>(__rank))
  {
    _CCCL_THROW(::std::invalid_argument{"Elem strides size mismatch"});
  }
  __tma_elem_strides_array_t __elem_strides_array{1};
  // tma_interleave_layout::none ignores the innermost elem stride (implicitly 1).
  const int __init_index = (__interleave_layout == tma_interleave_layout::none) ? 1 : 0;
  for (int __i = __init_index; __i < __rank; ++__i)
  {
    const auto __max_elem_stride = static_cast<int>(::cuda::std::min(__tensor_sizes[__i], uint64_t{8}));
    const auto __elem_stride     = __elem_strides[__rank - 1 - __i];
    if (__elem_stride <= 0 || __elem_stride > __max_elem_stride)
    {
      _CCCL_THROW(::std::invalid_argument{
        "elem_strides[i] must be greater than 0 and less than or equal to min(tensor.shape[rank - 1 - i], "
        "8)"});
    }
    __elem_strides_array[__i] = __elem_stride;
  }
  return __elem_strides_array;
}

_CCCL_HOST_API inline void __check_swizzle(tma_interleave_layout __interleave_layout, tma_swizzle __swizzle)
{
  if (__interleave_layout == tma_interleave_layout::bytes32)
  {
    if (__swizzle != tma_swizzle::bytes32)
    {
      _CCCL_THROW(::std::invalid_argument{"tma_interleave_layout::bytes32 requires tma_swizzle::bytes32"});
    }
  }
}

/***********************************************************************************************************************
 * Public API
 **********************************************************************************************************************/

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLTensor& __tensor,
  ::cuda::std::span<const int> __box_sizes,
  ::cuda::std::span<const int> __elem_strides,
  tma_interleave_layout __interleave_layout = tma_interleave_layout::none,
  tma_swizzle __swizzle                     = tma_swizzle::none,
  tma_l2_fetch_size __l2_fetch_size         = tma_l2_fetch_size::none,
  tma_oob_fill __oobfill                    = tma_oob_fill::none)
{
  using ::cuda::std::size_t;
  ::cuda::__check_device(__tensor, __swizzle);
  const auto __rank         = ::cuda::__get_tensor_map_rank(__tensor, __interleave_layout);
  const auto __address      = ::cuda::__get_tensor_address(__tensor, __interleave_layout);
  const auto __data_type    = ::cuda::__get_tensor_map_data_type(__tensor, __oobfill);
  const auto __tensor_sizes = ::cuda::__get_tensor_sizes(__tensor, __rank, __data_type);
  const auto __input_strides =
    ::cuda::__get_tensor_strides(__tensor, __rank, __data_type, __tensor_sizes, __interleave_layout);
  const auto __raw_interleave_layout = ::cuda::__to_cutensor_map(__interleave_layout);
  const auto __raw_swizzle           = ::cuda::__to_cutensor_map(__swizzle);
  const auto __raw_l2_fetch_size     = ::cuda::__to_cutensor_map(__l2_fetch_size);
  const auto __raw_oobfill           = ::cuda::__to_cutensor_map(__oobfill);
  ::cuda::__check_swizzle(__interleave_layout, __swizzle);
  const auto __raw_box_sizes = ::cuda::__get_box_sizes(
    __box_sizes, __tensor_sizes, __rank, __interleave_layout, __swizzle, __data_type, __tensor.device.device_id);
  const auto __raw_elem_strides =
    ::cuda::__get_elem_strides(__elem_strides, __tensor_sizes, __rank, __interleave_layout);
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

[[nodiscard]] _CCCL_HOST_API inline ::CUtensorMap make_tma_descriptor(
  const ::DLTensor& __tensor,
  ::cuda::std::span<const int> __box_sizes,
  tma_interleave_layout __interleave_layout = tma_interleave_layout::none,
  tma_swizzle __swizzle                     = tma_swizzle::none,
  tma_l2_fetch_size __l2_fetch_size         = tma_l2_fetch_size::none,
  tma_oob_fill __oobfill                    = tma_oob_fill::none)
{
  using ::cuda::std::size_t;
  const auto __rank                       = ::cuda::__get_tensor_map_rank(__tensor, __interleave_layout);
  constexpr int __elem_strides_storage[5] = {1, 1, 1, 1, 1};
  cuda::std::span<const int> __elem_strides{__elem_strides_storage, static_cast<size_t>(__rank)};
  return ::cuda::make_tma_descriptor(
    __tensor, __box_sizes, __elem_strides, __interleave_layout, __swizzle, __l2_fetch_size, __oobfill);
}

_CCCL_END_NAMESPACE_CUDA

#  include <cuda/std/__cccl/epilogue.h>

#endif // _CCCL_HAS_CTK() && !_CCCL_COMPILER(NVRTC) && _CCCL_HAS_INCLUDE(<dlpack/dlpack.h>)
#endif // _CUDA___TMA_MAKE_TMA_DESCRIPTOR_H
