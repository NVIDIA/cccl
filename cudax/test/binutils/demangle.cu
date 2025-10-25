//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/string_view>
#include <cuda/std/type_traits>

#include <cuda/experimental/binutils.cuh>

#include <string>

#include <testing.cuh>
#include <utility.cuh>

C2H_TEST("Demangle", "[binutils.demangle]")
{
  SECTION("Test signature");
  {
    static_assert(cuda::std::is_same_v<std::string, decltype(cudax::demangle(cuda::std::string_view{}))>);
    static_assert(!noexcept(cudax::demangle(cuda::std::string_view{})));
  }

  SECTION("Test positive case")
  {
    constexpr auto real_mangled_name = "_ZN8clstmp01I5cls01E13clstmp01_mf01Ev";
    const auto demangled             = cudax::demangle(real_mangled_name);
    CUDAX_REQUIRE(demangled == "clstmp01<cls01>::clstmp01_mf01()");
  }

  SECTION("Test error case")
  {
#if _CCCL_HAS_EXCEPTIONS()
    constexpr auto fake_mangled_name = "B@d_iDentiFier";
    try
    {
      auto demangled = cudax::demangle(fake_mangled_name);
      CUDAX_REQUIRE(false);
    }
    catch (const cuda::cuda_error& e)
    {
      CUDAX_REQUIRE(e.status() == cudaErrorInvalidSymbol);
    }
    catch (...)
    {
      CUDAX_REQUIRE(false);
    }
#endif // _CCCL_HAS_EXCEPTIONS()
  }
}
