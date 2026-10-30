//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <cuda/experimental/binutils.cuh>

#include <stdexcept>
#include <string>

#include "testing.cuh"
#include <catch2/matchers/catch_matchers_string.hpp>

C2H_TEST("cudax::demangle", "[binutils]")
{
  // 1. Test signature
  static_assert(cuda::std::is_same_v<std::string, decltype(cudax::demangle(cuda::std::string_view{}))>);
  static_assert(!noexcept(cudax::demangle(cuda::std::string_view{})));

  // 2. Test positive case
  {
    constexpr auto real_mangled_name = "_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
    const auto demangled             = cudax::demangle(real_mangled_name);
    REQUIRE(demangled == "clstmp01<cls01>::clstmp01_mf01()");
  }

// 3. Test error case
#if _CCCL_HAS_EXCEPTIONS()
  {
    constexpr auto fake_mangled_name = "B@d_iDentiFier";
    REQUIRE_THROWS_MATCHES(((void) cudax::demangle(fake_mangled_name)),
                           std::runtime_error,
                           Catch::Matchers::Message("invalid mangled name passed to cuda::demangle function"));
  }
#endif // _CCCL_HAS_EXCEPTIONS()
}
