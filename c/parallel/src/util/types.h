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

// On Windows, nvrtcGetTypeName calls UnDecorateSymbolName from Dbghelp.dll,
// which, for certain input types, returns string representations that nvcc
// balks on (e.g. `long long` becomes `__int64`).  This helper function looks
// for these unsupported types and converts them to nvcc-compatible types.
// The method signature is kept identical to `nvrtcGetTypeName` so that this
// helper can be used as a drop-in replacement.
template <typename T>
nvrtcResult cccl_type_name_from_nvrtc(std::string* result)
{
  if (const nvrtcResult res = nvrtcGetTypeName<T>(result); res != NVRTC_SUCCESS)
  {
    return res;
  }

  if (result->find("unsigned __int64") != std::string::npos)
  {
    *result = "::cuda::std::uint64_t";
  }
  else if (result->find("__int64") != std::string::npos)
  {
    *result = "::cuda::std::int64_t";
  }

  return NVRTC_SUCCESS;
}

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
      check(cccl_type_name_from_nvrtc<StorageT>(&result));
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
