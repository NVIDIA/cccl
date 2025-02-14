//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/std/cstdint>

#include <string>

#include "errors.h"
#include <cccl/c/types.h>

struct storage_t;
struct items_storage_t; // Used in merge_sort

template <typename StorageT = storage_t>
std::string cccl_type_enum_to_name(cccl_type_enum type, bool is_pointer = false)
{
  std::string result;

  switch (type)
  {
    case cccl_type_enum::INT8:
      result = "::cuda::std::int8_t";
      break;
    case cccl_type_enum::INT16:
      result = "::cuda::std::int16_t";
      break;
    case cccl_type_enum::INT32:
      result = "::cuda::std::int32_t";
      break;
    case cccl_type_enum::INT64:
      result = "::cuda::std::int64_t";
      break;
    case cccl_type_enum::UINT8:
      result = "::cuda::std::uint8_t";
      break;
    case cccl_type_enum::UINT16:
      result = "::cuda::std::uint16_t";
      break;
    case cccl_type_enum::UINT32:
      result = "::cuda::std::uint32_t";
      break;
    case cccl_type_enum::UINT64:
      result = "::cuda::std::uint64_t";
      break;
    case cccl_type_enum::FLOAT32:
      result = "float";
      break;
    case cccl_type_enum::FLOAT64:
      result = "double";
      break;
    case cccl_type_enum::STORAGE:
      check(nvrtcGetTypeName<StorageT>(&result));
      break;
  }

  if (is_pointer)
  {
    result += "*";
  }

  return result;
}
