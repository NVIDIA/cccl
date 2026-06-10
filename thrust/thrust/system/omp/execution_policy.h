// SPDX-FileCopyrightText: Copyright (c) 2008-2013, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! \file
//! \brief Execution policies for Thrust's OpenMP system.

// get the execution policies definitions first
#include <thrust/system/omp/detail/execution_policy.h>

// now get all the algorithm definitions
#include <thrust/system/omp/detail/adjacent_difference.h>
#include <thrust/system/omp/detail/assign_value.h>
#include <thrust/system/omp/detail/binary_search.h>
#include <thrust/system/omp/detail/copy.h>
#include <thrust/system/omp/detail/copy_if.h>
#include <thrust/system/omp/detail/count.h>
#include <thrust/system/omp/detail/equal.h>
#include <thrust/system/omp/detail/extrema.h>
#include <thrust/system/omp/detail/fill.h>
#include <thrust/system/omp/detail/find.h>
#include <thrust/system/omp/detail/for_each.h>
#include <thrust/system/omp/detail/gather.h>
#include <thrust/system/omp/detail/generate.h>
#include <thrust/system/omp/detail/get_value.h>
#include <thrust/system/omp/detail/inner_product.h>
#include <thrust/system/omp/detail/iter_swap.h>
#include <thrust/system/omp/detail/logical.h>
#include <thrust/system/omp/detail/malloc_and_free.h>
#include <thrust/system/omp/detail/merge.h>
#include <thrust/system/omp/detail/mismatch.h>
#include <thrust/system/omp/detail/partition.h>
#include <thrust/system/omp/detail/reduce.h>
#include <thrust/system/omp/detail/reduce_by_key.h>
#include <thrust/system/omp/detail/remove.h>
#include <thrust/system/omp/detail/replace.h>
#include <thrust/system/omp/detail/reverse.h>
#include <thrust/system/omp/detail/scan.h>
#include <thrust/system/omp/detail/scan_by_key.h>
#include <thrust/system/omp/detail/scatter.h>
#include <thrust/system/omp/detail/sequence.h>
#include <thrust/system/omp/detail/set_operations.h>
#include <thrust/system/omp/detail/sort.h>
#include <thrust/system/omp/detail/swap_ranges.h>
#include <thrust/system/omp/detail/tabulate.h>
#include <thrust/system/omp/detail/transform.h>
#include <thrust/system/omp/detail/transform_reduce.h>
#include <thrust/system/omp/detail/transform_scan.h>
#include <thrust/system/omp/detail/uninitialized_copy.h>
#include <thrust/system/omp/detail/uninitialized_fill.h>
#include <thrust/system/omp/detail/unique.h>
#include <thrust/system/omp/detail/unique_by_key.h>
