//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda.h>

#include <iostream>
#include <optional>
#include <string>

#include "test_util.h"
#include <c2h/catch2_test_helper.h>
#include <cccl/c/types.h>

template <int device_id_ = 0>
class BuildInformation
{
  int cc_major;
  int cc_minor;
  const char* cub_path;
  const char* thrust_path;
  const char* libcudacxx_path;
  const char* ctk_path;

  BuildInformation() = default;
  BuildInformation(int major, int minor, const char* cub, const char* thrust, const char* libcudacxx, const char* ctk)
      : cc_major(major)
      , cc_minor(minor)
      , cub_path(cub)
      , thrust_path(thrust)
      , libcudacxx_path(libcudacxx)
      , ctk_path(ctk)
  {}

public:
  static constexpr int device_id = device_id_;

  static const auto& init()
  {
    cudaDeviceProp deviceProp;
    cudaGetDeviceProperties(&deviceProp, device_id);

    static BuildInformation singleton{
      deviceProp.major, deviceProp.minor, TEST_CUB_PATH, TEST_THRUST_PATH, TEST_LIBCUDACXX_PATH, TEST_CTK_PATH};
    return singleton;
  }

  int get_cc_major() const
  {
    return cc_major;
  }
  int get_cc_minor() const
  {
    return cc_minor;
  }
  const char* get_cub_path() const
  {
    return cub_path;
  }
  const char* get_thrust_path() const
  {
    return thrust_path;
  }
  const char* get_libcudacxx_path() const
  {
    return libcudacxx_path;
  }
  const char* get_ctk_path() const
  {
    return ctk_path;
  }
};

template <typename BuildResultT,
          typename Build,
          typename Cleanup,
          typename Run,
          typename BuildCache,
          typename KeyT,
          typename... Tx>
void AlgorithmExecute(std::optional<BuildCache>& cache, const std::optional<KeyT>& lookup_key, Tx&&... args)
{
  constexpr int device_id = 0;
  const auto& build_info  = BuildInformation<device_id>::init();

  BuildResultT build;

  bool found               = false;
  const bool cache_and_key = bool(cache) && bool(lookup_key);

  if (cache_and_key)
  {
    auto& cache_v     = cache.value();
    const auto& key_v = lookup_key.value();
    if (cache_v.contains(key_v))
    {
      build = cache_v.get(key_v).get();
      found = true;
    }
  }

  if (!found)
  {
    REQUIRE(
      CUDA_SUCCESS
      == Build{}(&build,
                 args...,
                 build_info.get_cc_major(),
                 build_info.get_cc_minor(),
                 build_info.get_cub_path(),
                 build_info.get_thrust_path(),
                 build_info.get_libcudacxx_path(),
                 build_info.get_ctk_path()));

    if (cache_and_key)
    {
      auto& cache_v     = cache.value();
      const auto& key_v = lookup_key.value();
      cache_v.insert(key_v, build);
    }
  }

  const std::string& sass = inspect_sass(build.cubin, build.cubin_size);

  REQUIRE(sass.find("LDL") == std::string::npos);
  REQUIRE(sass.find("STL") == std::string::npos);

  CUstream null_stream = 0;

  size_t temp_storage_bytes = 0;
  REQUIRE(CUDA_SUCCESS == Run{}(build, nullptr, &temp_storage_bytes, args..., null_stream));

  pointer_t<uint8_t> temp_storage(temp_storage_bytes);

  REQUIRE(CUDA_SUCCESS == Run{}(build, temp_storage.ptr, &temp_storage_bytes, args..., null_stream));

  if (cache_and_key)
  {
    // if cache and lookup_key were provided, the ownership of resources
    // allocated for build is transferred to the cache, hence do nothing
  }
  else
  {
    // release build data resources
    REQUIRE(CUDA_SUCCESS == Cleanup{}(&build));
  }
}

template <typename BuildResultT, typename Cleanup>
struct BuildResultDeleter
{
  static constexpr Cleanup cleanup_{};
  void operator()(BuildResultT* build_data) const noexcept
  {
    BuildResultDeleter::check_success(cleanup_(build_data));
  }

private:
  static void check_success(CUresult status) noexcept
  {
    if (status != CUDA_SUCCESS)
    {
      std::cerr << "Clean-up call returned status " << status << std::endl;
    }
  }
};
