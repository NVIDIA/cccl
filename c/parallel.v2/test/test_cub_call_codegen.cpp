//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA Core Compute Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION.
//
//===----------------------------------------------------------------------===//

#include <cstddef>
#include <string>
#include <string_view>

#include <catch2/catch_test_macros.hpp>
#include <cccl/c/types.h>
#include <hostjit/codegen/cub_call.hpp>

namespace
{
cccl_iterator_t pointer_iterator(cccl_type_info value_type)
{
  cccl_iterator_t it{};
  it.size       = sizeof(void*);
  it.alignment  = alignof(void*);
  it.type       = CCCL_POINTER;
  it.value_type = value_type;
  return it;
}

hostjit::codegen::CubCall make_device_copy_call(const char* name, cccl_iterator_t it)
{
  using namespace hostjit::codegen;

  CubCall call = CubCall::from("cub/device/device_copy.cuh");
  call.run("cub::DeviceCopy::Copy").name(name).with(temp_storage, temp_bytes, in(it), out(it), num_items, stream);
  return call;
}

std::size_t find_required(std::string_view source, std::string_view needle)
{
  const auto pos = source.find(needle);
  CATCH_INFO(std::string{"missing generated source fragment: "} + std::string{needle});
  CATCH_REQUIRE(pos != std::string_view::npos);
  return pos;
}

std::size_t count_occurrences(std::string_view source, std::string_view needle)
{
  std::size_t count = 0;
  std::size_t pos   = 0;
  while ((pos = source.find(needle, pos)) != std::string_view::npos)
  {
    ++count;
    pos += needle.size();
  }
  return count;
}

void require_absent(std::string_view source, std::string_view needle)
{
  CATCH_INFO(std::string{"unexpected generated source fragment: "} + std::string{needle});
  CATCH_REQUIRE(source.find(needle) == std::string_view::npos);
}

void require_driver_initializer_source(std::string_view source, std::size_t expected_wrapper_count)
{
  find_required(source, "#include <cuda/__driver/driver_api.h>");
  find_required(source, "#if _CCCL_HOSTJIT() && !_CCCL_HOSTED() && !defined(__CUDA_ARCH__) && !defined(_WIN32)");
  find_required(source, "extern \"C\" void* dlopen(const char*, int);");
  find_required(source, "extern \"C\" void* dlsym(void*, const char*);");
  find_required(source, "::dlopen(\"libcuda.so.1\", RTLD_NOW)");
  find_required(source, "::dlsym(driver_library, \"cuGetProcAddress_v2\")");
  find_required(source, "::cuda::__driver::__getProcAddressFn(");
  find_required(source, "reinterpret_cast<decltype(::cuGetProcAddress)*>(get_proc_address)");
  find_required(source, "return static_cast<int>(cudaErrorNotSupported);");
  find_required(source, "return static_cast<int>(cudaErrorInitializationError);");
  require_absent(source, "CCCL_HOSTJIT_DRIVER_LOOKUP_");

  CATCH_INFO("generated source should define one HostJIT CUDA driver initializer");
  CATCH_REQUIRE(count_occurrences(source, "static int __cccl_hostjit_init_cuda_driver()") == 1);

  CATCH_INFO("generated source should call the HostJIT CUDA driver initializer once per wrapper");
  CATCH_REQUIRE(count_occurrences(source, "__cccl_hostjit_init_cuda_driver();") == expected_wrapper_count);
}
} // namespace

CATCH_TEST_CASE("CubCall generated source initializes cuda driver API", "[cub_call][codegen]")
{
  using namespace hostjit::codegen;

  const cccl_type_info int_type{sizeof(int), alignof(int), CCCL_INT32};
  const cccl_iterator_t int_it = pointer_iterator(int_type);

  CATCH_SECTION("single-function source")
  {
    const CubCall single_call = make_device_copy_call("cccl_jit_device_copy", int_it);
    const std::string source  = single_call.source();

    require_driver_initializer_source(source, 1);

    const auto init_pos =
      find_required(source, "int __cccl_hostjit_driver_status = __cccl_hostjit_init_cuda_driver();");
    const auto cub_call_pos = find_required(source, "cudaError_t err = cub::DeviceCopy::Copy(");
    CATCH_INFO("generated wrapper should initialize cuda::__driver before calling CUB");
    CATCH_REQUIRE(init_pos < cub_call_pos);

    find_required(source, "return __cccl_hostjit_driver_status;");
  }

  CATCH_SECTION("multi-function source")
  {
    const std::string source = CubCall::source(
      {make_device_copy_call("cccl_jit_device_copy_0", int_it),
       make_device_copy_call("cccl_jit_device_copy_1", int_it)});

    require_driver_initializer_source(source, 2);

    find_required(source, "namespace fn_0 {");
    find_required(source, "namespace fn_1 {");
    find_required(source, "extern \"C\" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy_0(");
    find_required(source, "extern \"C\" _CCCL_VISIBILITY_EXPORT int cccl_jit_device_copy_1(");
  }
}
