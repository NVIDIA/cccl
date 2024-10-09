/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_HELPER_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_HELPER_HPP_

#include "cublas_v2.h"

namespace __cublas_std
{

namespace __ex = std::experimental;

/**
 * Get cublasOp for a vector
 * When op = conjugate-only, we'll treat it as a one-column matrix and call the
 * cuBLAS API for a matrix with CUBLAS_OP_C
 * No-op                -> CUBLAS_OP_N
 * conjugated           -> CUBLAS_OP_C
 * transposed           -> CUBLAS_OP_T
 * conjugate_transposed -> CUBLAS_OP_C
 */
template <class in_matrix_t>
constexpr cublasOperation_t __get_cublas_op_vector()
{
  return (__nvhpc_std::__extract_conj<in_matrix_t>()
            ? CUBLAS_OP_C
            : (__nvhpc_std::__extract_trans<in_matrix_t>() ? CUBLAS_OP_T : CUBLAS_OP_N));
}

cublasOperation_t __op_to_cublas_op(char op)
{
  switch (op)
  {
    case 'N':
      return CUBLAS_OP_N;
    case 'T':
      return CUBLAS_OP_T;
    case 'C':
      return CUBLAS_OP_C;
    default:
      return CUBLAS_OP_N;
  }
}

cublasFillMode_t __to_cublas_fill_mode(char fill)
{
  switch (fill)
  {
    case 'U':
      return CUBLAS_FILL_MODE_UPPER;
    case 'L':
      return CUBLAS_FILL_MODE_LOWER;
  }
}

cublasDiagType_t __to_cublas_diag_type(char diag)
{
  switch (diag)
  {
    case 'U':
      return CUBLAS_DIAG_UNIT;
    case 'N':
      return CUBLAS_DIAG_NON_UNIT;
  }
}

cublasSideMode_t __to_cublas_side_mode(char side)
{
  switch (side)
  {
    case 'L':
      return CUBLAS_SIDE_LEFT;
    case 'R':
      return CUBLAS_SIDE_RIGHT;
  }
}

} // namespace __cublas_std

#endif
