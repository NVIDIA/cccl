/*
 * SPDX-FileCopyrightText: Copyright (c) 2022 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_BLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __bl = __blas_std;

/*
 * Going from row-major to column-major
 * - Matrices are effectively transposed
 * - Upper-triangular matrices become lower-triangular matrices, and vice versa
 * Left-__side op:
 * In C with row-major layout        : A   * X   = B
 * In cuBLAS with column-major layout: X^t * A^t = B^t
 * which is what cuBLAS Trsm computes with input A^t, B^t, and SIDE = right
 * Right-side op:
 * In C with row-major layout        : X   * A   = B
 * In cuBLAS with column-major layout: A^t * X^t = B^t
 * which is what cuBLAS Trsm computes with input A^t, B^t, and SIDE = left
 */

template <class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_B,
          class _SizeType_B,
          ::std::size_t _numRows_B,
          ::std::size_t _numCols_B,
          class _Layout_B,
          class _Accessor_B>
void __triangular_matrix_matrix_solve_impl(
  __nvhpc_exec<__blas_exec_space>&& /* __exec */
  ,
  _ElementType_B __alpha,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  char __trans,
  char __uplo,
  char __side,
  char __diag,
  bool __is_transposed,
  __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B)
{
  // TODO dimension checks

#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(triangular_matrix_matrix_solve, BLAS);
#endif

  __bl::__blas_trsm<_ElementType_B>::__trsm(
    &__side,
    &__uplo,
    &__trans,
    &__diag,
    __ex::linalg::__get_row_count(__B, __is_transposed),
    __ex::linalg::__get_col_count(__B, __is_transposed),
    __alpha,
    __A.data_handle(),
    __ex::linalg::__get_leading_dim(__A),
    __B.data_handle(),
    __ex::linalg::__get_leading_dim(__B));
}

} // namespace __nvhpc_std

#endif
