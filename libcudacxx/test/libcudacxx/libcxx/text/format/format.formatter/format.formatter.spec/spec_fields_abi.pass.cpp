//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// cuda::std::__fmt_spec_fields

#include <cuda/std/__format_>
#include <cuda/std/cassert>

struct TestSpecFieldsValues
{
  bool sign;
  bool alternate_form;
  bool zero_padding;
  bool precision;
  bool locale_specific_form;
  bool type;
  bool use_range_fill;
  bool clear_brackets;
  bool consume_all;
};

__host__ __device__ TestSpecFieldsValues make_test_spec_fields_values() noexcept
{
  TestSpecFieldsValues value{};
  value.sign                 = true;
  value.alternate_form       = false;
  value.zero_padding         = false;
  value.precision            = true;
  value.locale_specific_form = true;
  value.type                 = false;
  value.use_range_fill       = true;
  value.clear_brackets       = false;
  value.consume_all          = true;
  return value;
}

__host__ __device__ void verify_spec_fields(const cuda::std::__fmt_spec_fields& value) noexcept
{
  const auto ref = make_test_spec_fields_values();
  assert(value.__sign_ == ref.sign);
  assert(value.__alternate_form_ == ref.alternate_form);
  assert(value.__zero_padding_ == ref.zero_padding);
  assert(value.__precision_ == ref.precision);
  assert(value.__locale_specific_form_ == ref.locale_specific_form);
  assert(value.__type_ == ref.type);
  assert(value.__use_range_fill_ == ref.use_range_fill);
  assert(value.__clear_brackets_ == ref.clear_brackets);
  assert(value.__consume_all_ == ref.consume_all);
}

__host__ __device__ void test()
{
  static_assert(sizeof(cuda::std::__fmt_spec_fields) == 2);

  const auto ref = make_test_spec_fields_values();

  cuda::std::__fmt_spec_fields value{};
  value.__sign_                 = ref.sign;
  value.__alternate_form_       = ref.alternate_form;
  value.__zero_padding_         = ref.zero_padding;
  value.__precision_            = ref.precision;
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__type_                 = ref.type;
  value.__use_range_fill_       = ref.use_range_fill;
  value.__clear_brackets_       = ref.clear_brackets;
  value.__consume_all_          = ref.consume_all;

  verify_spec_fields(value);
}

#if !_CCCL_COMPILER(NVRTC)
__global__ void test_host_device_abi_compatiblity_kernel(cuda::std::__fmt_spec_fields value)
{
  verify_spec_fields(value);
}

void test_host_device_abi_compatiblity()
{
  const auto ref = make_test_spec_fields_values();

  cuda::std::__fmt_spec_fields value{};
  value.__sign_                 = ref.sign;
  value.__alternate_form_       = ref.alternate_form;
  value.__zero_padding_         = ref.zero_padding;
  value.__precision_            = ref.precision;
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__type_                 = ref.type;
  value.__use_range_fill_       = ref.use_range_fill;
  value.__clear_brackets_       = ref.clear_brackets;
  value.__consume_all_          = ref.consume_all;

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
