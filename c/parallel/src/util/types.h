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
struct input_storage_t;
struct output_storage_t;
struct items_storage_t; // Used in merge_sort

template <typename StorageT = storage_t>
std::string cccl_type_enum_to_name(cccl_type_enum type, bool is_pointer = false)
{
  std::string result;

  switch (type)
  {
    case cccl_type_enum::CCCL_INT8:
      result = "::cuda::std::int8_t";
      break;
    case cccl_type_enum::CCCL_INT16:
      result = "::cuda::std::int16_t";
      break;
    case cccl_type_enum::CCCL_INT32:
      result = "::cuda::std::int32_t";
      break;
    case cccl_type_enum::CCCL_INT64:
      result = "::cuda::std::int64_t";
      break;
    case cccl_type_enum::CCCL_UINT8:
      result = "::cuda::std::uint8_t";
      break;
    case cccl_type_enum::CCCL_UINT16:
      result = "::cuda::std::uint16_t";
      break;
    case cccl_type_enum::CCCL_UINT32:
      result = "::cuda::std::uint32_t";
      break;
    case cccl_type_enum::CCCL_UINT64:
      result = "::cuda::std::uint64_t";
      break;
    case cccl_type_enum::CCCL_FLOAT16:
#if _CCCL_HAS_NVFP16()
      result = "__half";
      break;
#else
      throw std::runtime_error("float16 is not supported");
#endif
    case cccl_type_enum::CCCL_FLOAT32:
      result = "float";
      break;
    case cccl_type_enum::CCCL_FLOAT64:
      result = "double";
      break;
    case cccl_type_enum::CCCL_STORAGE:
      check(nvrtcGetTypeName<StorageT>(&result));
      break;
    case cccl_type_enum::CCCL_BOOLEAN:
      result = "bool";
      break;
  }

  if (is_pointer)
  {
    result += "*";
  }

  return result;
}
