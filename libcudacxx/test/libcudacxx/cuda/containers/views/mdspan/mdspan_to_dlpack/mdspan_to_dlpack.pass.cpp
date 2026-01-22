//===----------------------------------------------------------------------===//
//
// Part of the libcu++ Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
// UNSUPPORTED: nvrtc

#include <cuda/mdspan>
#include <cuda/std/cassert>
#include <cuda/std/complex>
#include <cuda/std/cstdint>
#include <cuda/std/type_traits>
#include <cuda/utility>

#include "test_macros.h"

void check_datatype(const DLDataType& dt, uint8_t code, uint8_t bits, uint16_t lanes)
{
  assert(dt.code == code);
  assert(dt.bits == bits);
  assert(dt.lanes == lanes);
}

bool test_mdspan_to_dlpack_host_layout_right()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data[6]     = {0, 1, 2, 3, 4, 5};
  cuda::host_mdspan<int, extents_t> md{data, extents_t{}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCPU);
  assert(dltensor.device.device_id == 0);
  assert(dltensor.ndim == 2);
  check_datatype(dltensor.dtype, kDLInt, 32, 1);
  assert(dltensor.shape != nullptr);
  assert(dltensor.strides != nullptr);
  assert(dltensor.shape[0] == 2);
  assert(dltensor.shape[1] == 3);
  assert(dltensor.strides[0] == 3);
  assert(dltensor.strides[1] == 1);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == data);
  return true;
}

bool test_mdspan_to_dlpack_host_layout_left()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  int data[6]     = {0, 1, 2, 3, 4, 5};
  cuda::host_mdspan<int, extents_t, cuda::std::layout_left> md{data, extents_t{}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCPU);
  assert(dltensor.device.device_id == 0);
  check_datatype(dltensor.dtype, kDLInt, 32, 1);
  assert(dltensor.ndim == 2);
  assert(dltensor.shape != nullptr);
  assert(dltensor.strides != nullptr);
  assert(dltensor.shape[0] == 2);
  assert(dltensor.shape[1] == 3);
  assert(dltensor.strides[0] == 1);
  assert(dltensor.strides[1] == 2);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == data);
  return true;
}

bool test_mdspan_to_dlpack_empty_size()
{
  using extents_t = cuda::std::dims<2>;
  int data[1]     = {42};
  cuda::host_mdspan<int, extents_t> m{data, extents_t{0, 3}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(m);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCPU);
  assert(dltensor.device.device_id == 0);
  check_datatype(dltensor.dtype, kDLInt, 32, 1);
  assert(dltensor.ndim == 2);
  assert(dltensor.shape[0] == 0);
  assert(dltensor.shape[1] == 3);
  assert(dltensor.strides[0] == 3);
  assert(dltensor.strides[1] == 1);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == nullptr); // size() == 0 => nullptr
  return true;
}

bool test_mdspan_to_dlpack_rank_0()
{
  using extents_t = cuda::std::extents<size_t>;
  int data[1]     = {7};
  cuda::host_mdspan<int, extents_t> md{data, extents_t{}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCPU);
  assert(dltensor.device.device_id == 0);
  check_datatype(dltensor.dtype, kDLInt, 32, 1);
  assert(dltensor.ndim == 0);
  assert(dltensor.shape == nullptr);
  assert(dltensor.strides == nullptr);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == data); // rank-0 mdspan has size() == 1
  return true;
}

bool test_mdspan_to_dlpack_const_pointer()
{
  using extents_t   = cuda::std::dims<3>;
  const int data[6] = {0, 1, 2, 3, 4, 5};
  cuda::host_mdspan<const int, extents_t> md{data, extents_t{2, 3, 4}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCPU);
  assert(dltensor.device.device_id == 0);
  check_datatype(dltensor.dtype, kDLInt, 32, 1);
  assert(dltensor.ndim == 3);
  assert(dltensor.shape[0] == 2);
  assert(dltensor.shape[1] == 3);
  assert(dltensor.shape[2] == 4);
  assert(dltensor.strides[0] == 12);
  assert(dltensor.strides[1] == 4);
  assert(dltensor.strides[2] == 1);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == data); // rank-0 mdspan has size() == 1
  return true;
}

bool test_mdspan_to_dlpack_device()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  float* data     = nullptr;
  assert(cudaMalloc(&data, 6 * sizeof(float)) == cudaSuccess);
  cuda::device_mdspan<float, extents_t> md{data, extents_t{}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCUDA);
  assert(dltensor.device.device_id == 0);
  assert(dltensor.ndim == 2);
  check_datatype(dltensor.dtype, kDLFloat, 32, 1);
  assert(dltensor.shape[0] == 2);
  assert(dltensor.shape[1] == 3);
  assert(dltensor.strides[0] == 3);
  assert(dltensor.strides[1] == 1);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == data);
  return true;
}

bool test_mdspan_to_dlpack_managed()
{
  using extents_t = cuda::std::extents<size_t, 2, 3>;
  float* data     = nullptr;
  assert(cudaMallocManaged(&data, 6 * sizeof(float)) == cudaSuccess);
  cuda::managed_mdspan<float, extents_t> md{data, extents_t{}};
  auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
  auto dltensor       = dlpack_wrapper.get();

  assert(dltensor.device.device_type == kDLCUDAManaged);
  assert(dltensor.device.device_id == 0);
  assert(dltensor.ndim == 2);
  check_datatype(dltensor.dtype, kDLFloat, 32, 1);
  assert(dltensor.shape[0] == 2);
  assert(dltensor.shape[1] == 3);
  assert(dltensor.strides[0] == 3);
  assert(dltensor.strides[1] == 1);
  assert(dltensor.byte_offset == 0);
  assert(dltensor.data == data);
  return true;
}

template <typename ListT>
struct test_mdspan_to_dlpack_types_fn
{
  using list_t = ListT;

  cuda::std::array<DLDataType, list_t::__size> expected_types;

  template <size_t index>
  void call_impl() const
  {
    using T         = cuda::std::__type_at_c<index, list_t>;
    using extents_t = cuda::std::extents<size_t, 2, 3>;
    T data[6]       = {};
    cuda::host_mdspan<T, extents_t> md{data, extents_t{}};
    auto dlpack_wrapper = cuda::to_dlpack_tensor(md);
    auto dltensor       = dlpack_wrapper.get();

    auto type = expected_types[index];
    check_datatype(dltensor.dtype, type.code, type.bits, type.lanes);
  }

  template <size_t... Indices>
  void call(cuda::std::index_sequence<Indices...>) const
  {
    (call_impl<Indices>(), ...);
  }
};

bool test_mdspan_to_dlpack_basic_types()
{
  using list_t = cuda::std::__type_list<
    bool,
    signed char,
    short,
    int,
    long,
    long long,
#if _CCCL_HAS_INT128()
    __int128_t,
#endif
    // Unsigned integer types
    unsigned char,
    unsigned short,
    unsigned int,
    unsigned long,
    unsigned long long,
#if _CCCL_HAS_INT128()
    __uint128_t,
#endif
    // Floating-point types
    float,
    double
#if _CCCL_HAS_FLOAT128()
    ,
    __float128
#endif
    >;
  cuda::std::array<DLDataType, list_t::__size> expected_types = {
    DLDataType{kDLBool, 8, 1},
    // Signed integer types
    DLDataType{kDLInt, 8, 1},
    DLDataType{kDLInt, 16, 1},
    DLDataType{kDLInt, 32, 1},
    DLDataType{kDLInt, sizeof(long) * 8, 1},
    DLDataType{kDLInt, 64, 1},
#if _CCCL_HAS_INT128()
    DLDataType{kDLInt, 128, 1},
#endif
    // Unsigned integer types
    DLDataType{kDLUInt, 8, 1},
    DLDataType{kDLUInt, 16, 1},
    DLDataType{kDLUInt, 32, 1},
    DLDataType{kDLUInt, sizeof(unsigned long) * 8, 1},
    DLDataType{kDLUInt, 64, 1},
#if _CCCL_HAS_INT128()
    DLDataType{kDLUInt, 128, 1},
#endif
    // Floating-point types
    DLDataType{kDLFloat, 32, 1},
    DLDataType{kDLFloat, 64, 1},
#if _CCCL_HAS_FLOAT128()
    DLDataType{kDLFloat, 128, 1},
#endif
  };
  test_mdspan_to_dlpack_types_fn<list_t> test_fn{expected_types};
  test_fn.call(cuda::std::make_index_sequence<list_t::__size - 1>{});
  return true;
}

bool test_mdspan_to_dlpack_extended_fp_and_complex_types()
{
  using list_t = cuda::std::__type_list<
#if _CCCL_HAS_NVFP16()
    ::__half,
#endif
#if _CCCL_HAS_NVBF16()
    ::__nv_bfloat16,
#endif
#if _CCCL_HAS_FLOAT128()
    __float128,
#endif
  // Low-precision floating-point types
#if _CCCL_HAS_NVFP8_E4M3()
    ::__nv_fp8_e4m3,
#endif
#if _CCCL_HAS_NVFP8_E5M2()
    ::__nv_fp8_e5m2,
#endif
#if _CCCL_HAS_NVFP8_E8M0()
    ::__nv_fp8_e8m0,
#endif
#if _CCCL_HAS_NVFP6_E2M3()
    ::__nv_fp6_e2m3,
#endif
#if _CCCL_HAS_NVFP6_E3M2()
    ::__nv_fp6_e3m2,
#endif
#if _CCCL_HAS_NVFP4_E2M1()
    ::__nv_fp4_e2m1,
#endif
  // Complex types
#if _CCCL_HAS_NVFP16()
    cuda::std::complex<::__half>,
#endif
    cuda::std::complex<float>,
    cuda::std::complex<double>>;
  cuda::std::array<DLDataType, list_t::__size> expected_types = {
#if _CCCL_HAS_NVFP16()
    DLDataType{kDLFloat, 16, 1},
#endif
#if _CCCL_HAS_NVBF16()
    DLDataType{kDLBfloat, 16, 1},
#endif
#if _CCCL_HAS_FLOAT128()
    DLDataType{kDLFloat, 128, 1},
#endif
  // Low-precision floating-point types
#if _CCCL_HAS_NVFP8_E4M3()
    DLDataType{kDLFloat8_e4m3fn, 8, 1},
#endif
#if _CCCL_HAS_NVFP8_E5M2()
    DLDataType{kDLFloat8_e5m2, 8, 1},
#endif
#if _CCCL_HAS_NVFP8_E8M0()
    DLDataType{kDLFloat8_e8m0fnu, 8, 1},
#endif
#if _CCCL_HAS_NVFP6_E2M3()
    DLDataType{kDLFloat6_e2m3fn, 6, 1},
#endif
#if _CCCL_HAS_NVFP6_E3M2()
    DLDataType{kDLFloat6_e3m2fn, 6, 1},
#endif
#if _CCCL_HAS_NVFP4_E2M1()
    DLDataType{kDLFloat4_e2m1fn, 4, 1},
#endif
  // Complex types
#if _CCCL_HAS_NVFP16()
    DLDataType{kDLComplex, 32, 1},
#endif
    DLDataType{kDLComplex, 64, 1},
    DLDataType{kDLComplex, 128, 1}};
  test_mdspan_to_dlpack_types_fn<list_t> test_fn{expected_types};
  test_fn.call(cuda::std::make_index_sequence<list_t::__size>{});
  return true;
}

#if _CCCL_HAS_CTK()
bool test_mdspan_to_dlpack_vector_types()
{
  using list_t = cuda::std::__type_list<
    ::char2,
    ::char4,
    ::uchar2,
    ::uchar4,
    ::short2,
    ::short4,
    ::ushort2,
    ::ushort4,
    ::int2,
    ::int4,
    ::uint2,
    ::uint4,
    ::long2
#  if _CCCL_CTK_AT_LEAST(13, 0)
    ,
    ::long4_32a
#  else
    ,
    ::long4
#  endif
    ,
    ::ulong2
#  if _CCCL_CTK_AT_LEAST(13, 0)
    ,
    ::ulong4_32a
#  else
    ,
    ::ulong4
#  endif
    ,
    ::longlong2
#  if _CCCL_CTK_AT_LEAST(13, 0)
    ,
    ::longlong4_32a
#  else
    ,
    ::longlong4
#  endif
    ,
    ::ulonglong2
#  if _CCCL_CTK_AT_LEAST(13, 0)
    ,
    ::ulonglong4_32a
#  else
    ,
    ::ulonglong4
#  endif
    ,
    ::float2,
    ::float4,
    ::double2
#  if _CCCL_CTK_AT_LEAST(13, 0)
    ,
    ::double4_32a
#  else
    ,
    ::double4
#  endif
    >;
  cuda::std::array<DLDataType, list_t::__size> expected_types = {
    DLDataType{kDLInt, 8, 2},
    DLDataType{kDLInt, 8, 4},
    DLDataType{kDLUInt, 8, 2},
    DLDataType{kDLUInt, 8, 4},
    DLDataType{kDLInt, 16, 2},
    DLDataType{kDLInt, 16, 4},
    DLDataType{kDLUInt, 16, 2},
    DLDataType{kDLUInt, 16, 4},
    DLDataType{kDLInt, 32, 2},
    DLDataType{kDLInt, 32, 4},
    DLDataType{kDLUInt, 32, 2},
    DLDataType{kDLUInt, 32, 4},
    DLDataType{kDLInt, sizeof(long) * 8, 2},
    DLDataType{kDLInt, sizeof(long) * 8, 4},
    DLDataType{kDLUInt, sizeof(unsigned long) * 8, 2},
    DLDataType{kDLUInt, sizeof(unsigned long) * 8, 4},
    DLDataType{kDLInt, 64, 2},
    DLDataType{kDLInt, 64, 4},
    DLDataType{kDLUInt, 64, 2},
    DLDataType{kDLUInt, 64, 4},
    DLDataType{kDLFloat, 32, 2},
    DLDataType{kDLFloat, 32, 4},
    DLDataType{kDLFloat, 64, 2},
    DLDataType{kDLFloat, 64, 4}};
  test_mdspan_to_dlpack_types_fn<list_t> test_fn{expected_types};
  test_fn.call(cuda::std::make_index_sequence<list_t::__size - 1>{});
  return true;
}
#endif // _CCCL_HAS_CTK()

bool test_mdspan_to_dlpack_extended_fp_vector_types()
{
  using list_t = cuda::std::__type_list<
#if _CCCL_HAS_NVFP16()
    ::__half2,
#endif
#if _CCCL_HAS_NVBF16()
    ::__nv_bfloat162,
#endif
#if _CCCL_HAS_NVFP8_E4M3()
    ::__nv_fp8x2_e4m3,
    ::__nv_fp8x4_e4m3,
#endif
#if _CCCL_HAS_NVFP8_E5M2()
    ::__nv_fp8x2_e5m2,
    ::__nv_fp8x4_e5m2,
#endif
#if _CCCL_HAS_NVFP8_E8M0()
    ::__nv_fp8x2_e8m0,
    ::__nv_fp8x4_e8m0,
#endif
#if _CCCL_HAS_NVFP6_E2M3()
    ::__nv_fp6x2_e2m3,
    ::__nv_fp6x4_e2m3,
#endif
#if _CCCL_HAS_NVFP6_E3M2()
    ::__nv_fp6x2_e3m2,
    ::__nv_fp6x4_e3m2,
#endif
#if _CCCL_HAS_NVFP4_E2M1()
    ::__nv_fp4x2_e2m1,
    ::__nv_fp4x4_e2m1,
#endif
    void* /* dummy to allow trailing commas */>;
  cuda::std::array<DLDataType, list_t::__size> expected_types = {
#if _CCCL_HAS_NVFP16()
    DLDataType{kDLFloat, 16, 2},
#endif
#if _CCCL_HAS_NVBF16()
    DLDataType{kDLBfloat, 16, 2},
#endif
#if _CCCL_HAS_NVFP8_E4M3()
    DLDataType{kDLFloat8_e4m3fn, 8, 2},
    DLDataType{kDLFloat8_e4m3fn, 8, 4},
#endif
#if _CCCL_HAS_NVFP8_E5M2()
    DLDataType{kDLFloat8_e5m2, 8, 2},
    DLDataType{kDLFloat8_e5m2, 8, 4},
#endif
#if _CCCL_HAS_NVFP8_E8M0()
    DLDataType{kDLFloat8_e8m0fnu, 8, 2},
    DLDataType{kDLFloat8_e8m0fnu, 8, 4},
#endif
#if _CCCL_HAS_NVFP6_E2M3()
    DLDataType{kDLFloat6_e2m3fn, 6, 2},
    DLDataType{kDLFloat6_e2m3fn, 6, 4},
#endif
#if _CCCL_HAS_NVFP6_E3M2()
    DLDataType{kDLFloat6_e3m2fn, 6, 2},
    DLDataType{kDLFloat6_e3m2fn, 6, 4},
#endif
#if _CCCL_HAS_NVFP4_E2M1()
    DLDataType{kDLFloat4_e2m1fn, 4, 2},
    DLDataType{kDLFloat4_e2m1fn, 4, 4},
#endif
  };
  test_mdspan_to_dlpack_types_fn<list_t> test_fn{expected_types};
  test_fn.call(cuda::std::make_index_sequence<list_t::__size - 1>{});
  return true;
}

int main(int, char**)
{
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_host_layout_right());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_host_layout_left());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_empty_size());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_rank_0());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_const_pointer());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_device());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_managed());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_basic_types());))
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_extended_fp_and_complex_types());))
#if _CCCL_HAS_CTK()
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_vector_types());))
#endif // _CCCL_HAS_CTK()
  NV_IF_TARGET(NV_IS_HOST, (assert(test_mdspan_to_dlpack_extended_fp_vector_types());))
  return 0;
}
