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

std::string make_kernel_input_iterator(std::string_view offset_t, std::string_view input_value_t, cccl_iterator_t iter);

std::string make_kernel_output_iterator(std::string_view offset_t, std::string_view input_value_t, cccl_iterator_t iter);

std::string make_kernel_inout_iterator(std::string_view offset_t, std::string_view input_value_t, cccl_iterator_t iter);
