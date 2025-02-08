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

  if (is_pointer)
  {
    switch (type)
    {
      case cccl_type_enum::INT8:

        check(nvrtcGetTypeName<::cuda::std::int8_t*>(&result));
        break;
      case cccl_type_enum::INT16:
        check(nvrtcGetTypeName<::cuda::std::int16_t*>(&result));
        break;
      case cccl_type_enum::INT32:
        check(nvrtcGetTypeName<::cuda::std::int32_t*>(&result));
        break;
      case cccl_type_enum::INT64:
        check(nvrtcGetTypeName<::cuda::std::int64_t*>(&result));
        break;
      case cccl_type_enum::UINT8:
        check(nvrtcGetTypeName<::cuda::std::uint8_t*>(&result));
        break;
      case cccl_type_enum::UINT16:
        check(nvrtcGetTypeName<::cuda::std::uint16_t*>(&result));
        break;
      case cccl_type_enum::UINT32:
        check(nvrtcGetTypeName<::cuda::std::uint32_t*>(&result));
        break;
      case cccl_type_enum::UINT64:
        check(nvrtcGetTypeName<::cuda::std::uint64_t*>(&result));
        break;
      case cccl_type_enum::FLOAT32:
        check(nvrtcGetTypeName<float*>(&result));
        break;
      case cccl_type_enum::FLOAT64:
        check(nvrtcGetTypeName<double*>(&result));
        break;
      case cccl_type_enum::STORAGE:
        check(nvrtcGetTypeName<StorageT*>(&result));
        break;
    }
  }
  else
  {
    switch (type)
    {
      case cccl_type_enum::INT8:
        check(nvrtcGetTypeName<::cuda::std::int8_t>(&result));
        break;
      case cccl_type_enum::INT16:
        check(nvrtcGetTypeName<::cuda::std::int16_t>(&result));
        break;
      case cccl_type_enum::INT32:
        check(nvrtcGetTypeName<::cuda::std::int32_t>(&result));
        break;
      case cccl_type_enum::INT64:
        check(nvrtcGetTypeName<::cuda::std::int64_t>(&result));
        break;
      case cccl_type_enum::UINT8:
        check(nvrtcGetTypeName<::cuda::std::uint8_t>(&result));
        break;
      case cccl_type_enum::UINT16:
        check(nvrtcGetTypeName<::cuda::std::uint16_t>(&result));
        break;
      case cccl_type_enum::UINT32:
        check(nvrtcGetTypeName<::cuda::std::uint32_t>(&result));
        break;
      case cccl_type_enum::UINT64:
        check(nvrtcGetTypeName<::cuda::std::uint64_t>(&result));
        break;
      case cccl_type_enum::FLOAT32:
        check(nvrtcGetTypeName<float>(&result));
        break;
      case cccl_type_enum::FLOAT64:
        check(nvrtcGetTypeName<double>(&result));
        break;
      case cccl_type_enum::STORAGE:
        check(nvrtcGetTypeName<StorageT>(&result));
        break;
    }
  }

  return result;
}

std::string_view cccl_type_enum_to_string(cccl_type_enum type);
