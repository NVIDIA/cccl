// SPDX-FileCopyrightText: Copyright (c) 2011, Duane Merrill. All rights reserved.
// SPDX-FileCopyrightText: Copyright (c) 2011-2024, NVIDIA CORPORATION. All rights reserved.
// SPDX-License-Identifier: BSD-3

/**
 * \file
 * Common type manipulation (metaprogramming) utilities
 */

#pragma once

#include <cub/config.cuh>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cub/detail/align_bytes.cuh> // IWYU pragma: export
#include <cub/detail/binary_op_has_idx_param.cuh> // IWYU pragma: export
#include <cub/detail/constant.cuh> // IWYU pragma: export
#include <cub/detail/cub_vector.cuh> // IWYU pragma: export
#include <cub/detail/detect_nested_type.cuh> // IWYU pragma: export
#include <cub/detail/double_buffer.cuh> // IWYU pragma: export
#include <cub/detail/future_value.cuh> // IWYU pragma: export
#include <cub/detail/input_value.cuh> // IWYU pragma: export
#include <cub/detail/it_traits.cuh> // IWYU pragma: export
#include <cub/detail/key_value_pair.cuh> // IWYU pragma: export
#include <cub/detail/lazy_trait.cuh> // IWYU pragma: export
#include <cub/detail/log2.cuh> // IWYU pragma: export
#include <cub/detail/non_void_value.cuh> // IWYU pragma: export
#include <cub/detail/null_type.cuh> // IWYU pragma: export
#include <cub/detail/numeric_traits.cuh> // IWYU pragma: export
#include <cub/detail/power_of_two.cuh> // IWYU pragma: export
#include <cub/detail/type_size.cuh> // IWYU pragma: export
#include <cub/detail/uninitialized.cuh> // IWYU pragma: export
#include <cub/detail/unit_word.cuh> // IWYU pragma: export
