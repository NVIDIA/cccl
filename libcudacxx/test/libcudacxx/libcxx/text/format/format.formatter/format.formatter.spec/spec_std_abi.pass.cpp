//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// cuda::std::__fmt_spec_std

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/utility>

struct TestSpecStdValues
{
  cuda::std::__fmt_spec_alignment alignment;
  cuda::std::__fmt_spec_sign sign;
  bool alternate_form;
  bool locale_specific_form;
  bool padding_0;
  cuda::std::__fmt_spec_type type;
};

__host__ __device__ TestSpecStdValues make_test_spec_std_values() noexcept
{
  TestSpecStdValues value{};
  value.alignment            = cuda::std::__fmt_spec_alignment::__center;
  value.sign                 = cuda::std::__fmt_spec_sign::__space;
  value.alternate_form       = false;
  value.locale_specific_form = true;
  value.padding_0            = false;
  value.type                 = cuda::std::__fmt_spec_type::__fixed_lower_case;
  return value;
}

__host__ __device__ void verify_spec_std(const cuda::std::__fmt_spec_std& value) noexcept
{
  const auto ref = make_test_spec_std_values();
  assert(value.__alignment_ == cuda::std::to_underlying(ref.alignment));
  assert(value.__sign_ == cuda::std::to_underlying(ref.sign));
  assert(value.__alternate_form_ == ref.alternate_form);
  assert(value.__locale_specific_form_ == ref.locale_specific_form);
  assert(value.__padding_0_ == ref.padding_0);
  assert(value.__type_ == ref.type);
}

__host__ __device__ void test()
{
  static_assert(sizeof(cuda::std::__fmt_spec_std) == 2);
  assert(offsetof(cuda::std::__fmt_spec_std, __type_) == 1);

  const auto ref = make_test_spec_std_values();

  cuda::std::__fmt_spec_std value{};
  value.__alignment_            = cuda::std::to_underlying(ref.alignment);
  value.__sign_                 = cuda::std::to_underlying(ref.sign);
  value.__alternate_form_       = ref.alternate_form;
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__padding_0_            = ref.padding_0;
  value.__type_                 = ref.type;

  verify_spec_std(value);
}

#if !_CCCL_COMPILER(NVRTC)
__global__ void test_host_device_abi_compatiblity_kernel(cuda::std::__fmt_spec_std value)
{
  verify_spec_std(value);
}

void test_host_device_abi_compatiblity()
{
  const auto ref = make_test_spec_std_values();

  cuda::std::__fmt_spec_std value{};
  value.__alignment_            = cuda::std::to_underlying(ref.alignment);
  value.__sign_                 = cuda::std::to_underlying(ref.sign);
  value.__alternate_form_       = ref.alternate_form;
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__padding_0_            = ref.padding_0;
  value.__type_                 = ref.type;

  test_host_device_abi_compatiblity_kernel<<<1, 1>>>(value);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  test();
  NV_IF_TARGET(NV_IS_HOST, (test_host_device_abi_compatiblity();))
  return 0;
}
