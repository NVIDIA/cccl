// SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: BSD-3

#include "insert_nested_NVTX_range_guard.h"

#include <cub/device/device_select.cuh>

#include <cuda/std/cstdint>

#include <cstddef>

#include "cub_test_macros.h"

CUB_TEST("DeviceSelect::UniqueByKey legacy size-query is unambiguous", "[select][device]", CUB_SMALL)
{
  void* d_temp_storage      = nullptr;
  size_t temp_storage_bytes = 0;
  int* d_keys_in            = nullptr;
  int* d_values_in          = nullptr;
  int* d_keys_out           = nullptr;
  int* d_values_out         = nullptr;
  int* d_num_selected       = nullptr;
  const int n               = 0;

  REQUIRE(cudaSuccess
          == cub::DeviceSelect::UniqueByKey(
            d_temp_storage, temp_storage_bytes, d_keys_in, d_values_in, d_keys_out, d_values_out, d_num_selected, n));
}
