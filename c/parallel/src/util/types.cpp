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
    case cccl_type_enum::CCCL_INT8:
      return "::cuda::std::int8_t";
    case cccl_type_enum::CCCL_INT16:
      return "::cuda::std::int16_t";
    case cccl_type_enum::CCCL_INT32:
      return "::cuda::std::int32_t";
    case cccl_type_enum::CCCL_INT64:
      return "::cuda::std::int64_t";
    case cccl_type_enum::CCCL_UINT8:
      return "::cuda::std::uint8_t";
    case cccl_type_enum::CCCL_UINT16:
      return "::cuda::std::uint16_t";
    case cccl_type_enum::CCCL_UINT32:
      return "::cuda::std::uint32_t";
    case cccl_type_enum::CCCL_UINT64:
      return "::cuda::std::uint64_t";
    case cccl_type_enum::CCCL_FLOAT32:
      return "float";
    case cccl_type_enum::CCCL_FLOAT64:
      return "double";
    case cccl_type_enum::CCCL_STORAGE:
      return "storage_t";
  }
  return "unknown";
}
