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
//! \brief Execution policies for Thrust's TBB system.

// get the execution policies definitions first
#include <thrust/system/tbb/detail/execution_policy.h>

// now get all the algorithm definitions
#include <thrust/system/tbb/detail/adjacent_difference.h>
#include <thrust/system/tbb/detail/assign_value.h>
#include <thrust/system/tbb/detail/binary_search.h>
#include <thrust/system/tbb/detail/copy.h>
#include <thrust/system/tbb/detail/copy_if.h>
#include <thrust/system/tbb/detail/count.h>
#include <thrust/system/tbb/detail/equal.h>
#include <thrust/system/tbb/detail/extrema.h>
#include <thrust/system/tbb/detail/fill.h>
#include <thrust/system/tbb/detail/find.h>
#include <thrust/system/tbb/detail/for_each.h>
#include <thrust/system/tbb/detail/gather.h>
#include <thrust/system/tbb/detail/generate.h>
#include <thrust/system/tbb/detail/get_value.h>
#include <thrust/system/tbb/detail/inner_product.h>
#include <thrust/system/tbb/detail/iter_swap.h>
#include <thrust/system/tbb/detail/logical.h>
#include <thrust/system/tbb/detail/malloc_and_free.h>
#include <thrust/system/tbb/detail/merge.h>
#include <thrust/system/tbb/detail/mismatch.h>
#include <thrust/system/tbb/detail/partition.h>
#include <thrust/system/tbb/detail/reduce.h>
#include <thrust/system/tbb/detail/reduce_by_key.h>
#include <thrust/system/tbb/detail/remove.h>
#include <thrust/system/tbb/detail/replace.h>
#include <thrust/system/tbb/detail/reverse.h>
#include <thrust/system/tbb/detail/scan.h>
#include <thrust/system/tbb/detail/scan_by_key.h>
#include <thrust/system/tbb/detail/scatter.h>
#include <thrust/system/tbb/detail/sequence.h>
#include <thrust/system/tbb/detail/set_operations.h>
#include <thrust/system/tbb/detail/sort.h>
#include <thrust/system/tbb/detail/swap_ranges.h>
#include <thrust/system/tbb/detail/tabulate.h>
#include <thrust/system/tbb/detail/transform.h>
#include <thrust/system/tbb/detail/transform_reduce.h>
#include <thrust/system/tbb/detail/transform_scan.h>
#include <thrust/system/tbb/detail/uninitialized_copy.h>
#include <thrust/system/tbb/detail/uninitialized_fill.h>
#include <thrust/system/tbb/detail/unique.h>
#include <thrust/system/tbb/detail/unique_by_key.h>
