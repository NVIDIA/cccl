//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//
#include <cuda/std/type_traits>

#include "test_macros.h"

#ifdef _LIBCUDACXX_HAS_NVFP16
static_assert(cuda::std::is_same<cuda::std::common_type<__half, __half>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, __half&>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, __half>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, __half&&>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, __half>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, __half&&>::type, __half>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, __half&>::type, __half>::value, "");

static_assert(cuda::std::is_same<cuda::std::common_type<__half, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, float&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__half&&, float&>::type, float>::value, "");
#endif // _LIBCUDACXX_HAS_NVFP16

#ifdef _LIBCUDACXX_HAS_NVBF16
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, __nv_bfloat16>::type, __nv_bfloat16>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, __nv_bfloat16&>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, __nv_bfloat16>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, __nv_bfloat16&&>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, __nv_bfloat16>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, __nv_bfloat16&&>::type, __nv_bfloat16>::value,
              "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, __nv_bfloat16&>::type, __nv_bfloat16>::value,
              "");

static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, float&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, float>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&, float&&>::type, float>::value, "");
static_assert(cuda::std::is_same<cuda::std::common_type<__nv_bfloat16&&, float&>::type, float>::value, "");

#  if TEST_STD_VER >= 2014
static_assert(!cuda::std::__has_common_type<__nv_bfloat16, __half>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16, __half&>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&, __half>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16, __half&&>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&&, __half>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&, __half&&>, "");
static_assert(!cuda::std::__has_common_type<__nv_bfloat16&&, __half&>, "");
#  endif // TEST_STD_VER >= 2014
#endif // _LIBCUDACXX_HAS_NVBF16

int main(int argc, char** argv)
{
  return 0;
}
