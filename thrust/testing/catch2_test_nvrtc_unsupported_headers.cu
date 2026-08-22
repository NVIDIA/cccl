// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include <thrust/detail/config.h>

#include <array>
#include <cstddef>
#include <string>
#include <vector>

#include <nvrtc.h>
#include <nvrtc_args.h>

#include "catch2_test_helper.h"

TEST_CASE("Unsupported headers emit an NVRTC diagnostic", "[nvrtc]")
{
  constexpr std::array unsupported_headers{
    "thrust/allocate_unique.h",
    "thrust/device_allocator.h",
    "thrust/device_delete.h",
    "thrust/device_free.h",
    "thrust/device_make_unique.h",
    "thrust/device_malloc.h",
    "thrust/device_malloc_allocator.h",
    "thrust/device_new.h",
    "thrust/device_new_allocator.h",
    "thrust/device_vector.h",
    "thrust/host_vector.h",
    "thrust/mr/allocator.h",
    "thrust/mr/device_memory_resource.h",
    "thrust/mr/disjoint_pool.h",
    "thrust/mr/disjoint_sync_pool.h",
    "thrust/mr/disjoint_tls_pool.h",
    "thrust/mr/fancy_pointer_resource.h",
    "thrust/mr/host_memory_resource.h",
    "thrust/mr/memory_resource.h",
    "thrust/mr/new.h",
    "thrust/mr/polymorphic_adaptor.h",
    "thrust/mr/pool.h",
    "thrust/mr/pool_options.h",
    "thrust/mr/sync_pool.h",
    "thrust/mr/tls_pool.h",
    "thrust/mr/universal_memory_resource.h",
    "thrust/mr/validator.h",
    "thrust/per_device_resource.h",
    "thrust/system/cpp/memory.h",
    "thrust/system/cpp/memory_resource.h",
    "thrust/system/cpp/vector.h",
    "thrust/system/cuda/error.h",
    "thrust/system/cuda/memory.h",
    "thrust/system/cuda/memory_resource.h",
    "thrust/system/cuda/vector.h",
    "thrust/system/error_code.h",
    "thrust/system/system_error.h",
    "thrust/system_error.h",
    "thrust/universal_allocator.h",
    "thrust/universal_vector.h"};

  const std::string standard = std::string{"-std=c++"} + std::to_string(_CCCL_STD_VER - 2000);
  std::vector<const char*> options{nvrtc_cub_path, nvrtc_thrust_path, nvrtc_libcudacxx_path};
  for (const char* const nvrtc_ctk_path : nvrtc_ctk_paths)
  {
    options.push_back(nvrtc_ctk_path);
  }
  options.push_back(standard.c_str());

  for (const char* const header : unsupported_headers)
  {
    INFO("header = " << header);

    const std::string source = std::string{"#include <"} + header + ">\n";
    const std::string expected_diagnostic =
      std::string{"Including <"} + header + "> is not supported when compiling with NVRTC";

    nvrtcProgram program{};
    REQUIRE(NVRTC_SUCCESS
            == nvrtcCreateProgram(&program, source.c_str(), "nvrtc_unsupported_header_test.cu", 0, nullptr, nullptr));

    const nvrtcResult compile_result = nvrtcCompileProgram(program, static_cast<int>(options.size()), options.data());

    std::size_t log_size{};
    const nvrtcResult get_log_size_result = nvrtcGetProgramLogSize(program, &log_size);

    std::string log;
    nvrtcResult get_log_result = get_log_size_result;
    if (NVRTC_SUCCESS == get_log_size_result)
    {
      log.resize(log_size);
      get_log_result = nvrtcGetProgramLog(program, log.data());
    }

    const nvrtcResult destroy_result = nvrtcDestroyProgram(&program);

    REQUIRE(NVRTC_SUCCESS == get_log_size_result);
    REQUIRE(NVRTC_SUCCESS == get_log_result);
    REQUIRE(NVRTC_SUCCESS == destroy_result);

    INFO("NVRTC log = " << log);
    CHECK(NVRTC_ERROR_COMPILATION == compile_result);
    CHECK(log.find(expected_diagnostic) != std::string::npos);
  }
}
