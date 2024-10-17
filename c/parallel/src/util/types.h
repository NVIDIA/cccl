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

#include <string>

#include <cccl/c/types.h>

struct storage_t;

std::string cccl_type_enum_to_name(cccl_type_enum type, bool is_pointer = false);
char const* cccl_type_enum_to_string(cccl_type_enum type);
