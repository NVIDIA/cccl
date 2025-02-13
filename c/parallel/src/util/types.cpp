//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include "types.h"

#include <cuda/std/cstdint>

#include "errors.h"
#include <nvJitLink.h>
#include <nvrtc.h>

std::string_view cccl_type_enum_to_string(cccl_type_enum type)
{
  switch (type)
  {
    case cccl_type_enum::INT8:
      return "::cuda::std::int8_t";
    case cccl_type_enum::INT16:
      return "::cuda::std::int16_t";
    case cccl_type_enum::INT32:
      return "::cuda::std::int32_t";
    case cccl_type_enum::INT64:
      return "::cuda::std::int64_t";
    case cccl_type_enum::UINT8:
      return "::cuda::std::uint8_t";
    case cccl_type_enum::UINT16:
      return "::cuda::std::uint16_t";
    case cccl_type_enum::UINT32:
      return "::cuda::std::uint32_t";
    case cccl_type_enum::UINT64:
      return "::cuda::std::uint64_t";
    case cccl_type_enum::FLOAT32:
      return "float";
    case cccl_type_enum::FLOAT64:
      return "double";
    case cccl_type_enum::STORAGE:
      return "storage_t";
  }
  return "unknown";
}
