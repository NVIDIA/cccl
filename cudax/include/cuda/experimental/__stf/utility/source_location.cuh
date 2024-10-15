//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Source location
 */

#pragma once

#if __has_include(<source_location>)
#include <source_location>
namespace cuda::experimental::stf {
using source_location = ::std::source_location;
}
#elif __has_include(<experimental/source_location>)
#include <experimental/source_location>
namespace cuda::experimental::stf {
using source_location = ::std::experimental::source_location;
}
#else
#error "source_location was not found"
#endif
