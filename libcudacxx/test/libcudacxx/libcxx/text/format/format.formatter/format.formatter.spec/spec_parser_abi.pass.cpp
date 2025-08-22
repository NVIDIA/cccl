//===----------------------------------------------------------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

// <cuda/std/format>

// cuda::std::__fmt_spec_parser

#include <cuda/std/__format_>
#include <cuda/std/cassert>
#include <cuda/std/cstddef>
#include <cuda/std/cstdint>
#include <cuda/std/cstring>
#include <cuda/std/utility>

template <class CharT>
struct TestSpecParserValues
{
  cuda::std::__fmt_spec_alignment alignment;
  cuda::std::__fmt_spec_sign sign;
  bool alternate_form;
  bool locale_specific_form;
  bool padding_0;
  cuda::std::__fmt_spec_type type;
  bool hour;
  bool weekday_name;
  bool weekday;
  bool day_of_year;
  bool week_of_year;
  bool month_name;
  cuda::std::uint8_t reserved_0;
  cuda::std::uint8_t reserved_1;
  bool width_as_arg;
  bool precision_as_arg;
  cuda::std::uint32_t width;
  cuda::std::int32_t precision;
  cuda::std::__fmt_spec_code_point<CharT> fill;
};

template <class CharT>
__host__ __device__ TestSpecParserValues<CharT> make_test_spec_parser_values() noexcept
{
  TestSpecParserValues<CharT> value{};
  value.alignment            = cuda::std::__fmt_spec_alignment::__center;
  value.sign                 = cuda::std::__fmt_spec_sign::__space;
  value.alternate_form       = false;
  value.locale_specific_form = true;
  value.type                 = cuda::std::__fmt_spec_type::__fixed_lower_case;
  value.hour                 = true;
  value.weekday_name         = true;
  value.weekday              = false;
  value.day_of_year          = true;
  value.week_of_year         = false;
  value.month_name           = false;
  value.reserved_0           = 0b10;
  value.reserved_1           = 0b10'1001;
  value.width_as_arg         = false;
  value.precision_as_arg     = true;
  value.width                = 0x1234'abcd;
  value.precision            = 0x5678'ef01;
  // value.fill has own constructor
  return value;
}

template <class CharT>
__host__ __device__ void verify_spec_parser(const cuda::std::__fmt_spec_parser<CharT>& value)
{
  const auto ref = make_test_spec_parser_values<CharT>();
  assert(value.__alignment_ == cuda::std::to_underlying(ref.alignment));
  assert(value.__sign_ == cuda::std::to_underlying(ref.sign));
  assert(value.__alternate_form_ == ref.alternate_form);
  assert(value.__locale_specific_form_ == ref.locale_specific_form);
  assert(value.__type_ == ref.type);
  assert(value.__hour_ == ref.hour);
  assert(value.__weekday_name_ == ref.weekday_name);
  assert(value.__weekday_ == ref.weekday);
  assert(value.__day_of_year_ == ref.day_of_year);
  assert(value.__week_of_year_ == ref.week_of_year);
  assert(value.__month_name_ == ref.month_name);
  assert(value.__reserved_0_ == ref.reserved_0);
  assert(value.__reserved_1_ == ref.reserved_1);
  assert(value.__width_as_arg_ == ref.width_as_arg);
  assert(value.__precision_as_arg_ == ref.precision_as_arg);
  assert(value.__width_ == ref.width);
  assert(value.__precision_ == ref.precision);
  assert(cuda::std::memcmp(&value.__fill_.__data, &ref.fill.__data, 4) == 0);
}

template <class CharT>
__host__ __device__ void test_type()
{
  static_assert(sizeof(cuda::std::__fmt_spec_parser<CharT>) == 16);
  assert(offsetof(cuda::std::__fmt_spec_parser<CharT>, __type_) == 1);
  assert(offsetof(cuda::std::__fmt_spec_parser<CharT>, __width_) == 4);
  assert(offsetof(cuda::std::__fmt_spec_parser<CharT>, __precision_) == 8);
  assert(offsetof(cuda::std::__fmt_spec_parser<CharT>, __fill_) == 12);

  const auto ref = make_test_spec_parser_values<CharT>();

  cuda::std::__fmt_spec_parser<CharT> value{};
  value.__alignment_            = cuda::std::to_underlying(ref.alignment);
  value.__sign_                 = cuda::std::to_underlying(ref.sign);
  value.__alternate_form_       = ref.alternate_form;
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__type_                 = ref.type;
  value.__hour_                 = ref.hour;
  value.__weekday_name_         = ref.weekday_name;
  value.__weekday_              = ref.weekday;
  value.__day_of_year_          = ref.day_of_year;
  value.__week_of_year_         = ref.week_of_year;
  value.__month_name_           = ref.month_name;
  value.__reserved_0_           = ref.reserved_0;
  value.__reserved_1_           = ref.reserved_1;
  value.__width_as_arg_         = ref.width_as_arg;
  value.__precision_as_arg_     = ref.precision_as_arg;
  value.__width_                = ref.width;
  value.__precision_            = ref.precision;
  value.__fill_                 = ref.fill;

  verify_spec_parser(value);
}

__host__ __device__ void test()
{
  test_type<char>();
#if _CCCL_HAS_WCHAR_T()
  test_type<wchar_t>();
#endif // _CCCL_HAS_WCHAR_T()
}

#if !_CCCL_COMPILER(NVRTC)
template <class T>
__global__ void test_host_device_abi_compatiblity_kernel(cuda::std::__fmt_spec_parser<T> value)
{
  verify_spec_parser(value);
}

template <class CharT>
void test_host_device_abi_compatiblity()
{
  const auto ref = make_test_spec_parser_values<CharT>();

  cuda::std::__fmt_spec_parser<CharT> value{};
  value.__alignment_            = cuda::std::to_underlying(ref.alignment);
  value.__sign_                 = cuda::std::to_underlying(ref.sign);
  value.__alternate_form_       = ref.alternate_form;
  value.__locale_specific_form_ = ref.locale_specific_form;
  value.__type_                 = ref.type;
  value.__hour_                 = ref.hour;
  value.__weekday_name_         = ref.weekday_name;
  value.__weekday_              = ref.weekday;
  value.__day_of_year_          = ref.day_of_year;
  value.__week_of_year_         = ref.week_of_year;
  value.__month_name_           = ref.month_name;
  value.__reserved_0_           = ref.reserved_0;
  value.__reserved_1_           = ref.reserved_1;
  value.__width_as_arg_         = ref.width_as_arg;
  value.__precision_as_arg_     = ref.precision_as_arg;
  value.__width_                = ref.width;
  value.__precision_            = ref.precision;
  value.__fill_                 = ref.fill;

  test_host_device_abi_compatiblity_kernel<CharT><<<1, 1>>>(value);
  assert(cudaDeviceSynchronize() == cudaSuccess);
}

void test_host_device_abi_compatiblity()
{
  test_host_device_abi_compatiblity<char>();
#  if _CCCL_HAS_WCHAR_T()
  test_host_device_abi_compatiblity<wchar_t>();
#  endif // _CCCL_HAS_WCHAR_T()
}
#endif // !_CCCL_COMPILER(NVRTC)

int main(int, char**)
{
  test();
  NV_IF_TARGET(NV_IS_HOST, (test_host_device_abi_compatiblity();))
  return 0;
}
