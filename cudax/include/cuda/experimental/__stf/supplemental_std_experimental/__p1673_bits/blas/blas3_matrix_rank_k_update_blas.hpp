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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS3_MATRIX_RANK_K_UPDATE_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS3_MATRIX_RANK_K_UPDATE_BLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __bl = __blas_std;

// TODO implement hermitian_matrix_rank_k_update

/*
 * Going from row-major to column-major
 * - Matrices are effectively transposed
 * - The upper-triangular part of a matrix becomes the lower-triangular matrices,
 *   and vice versa
 * In C with row-major layout        : C   = alpha *  A     *A^t + beta * C
 * In cuBLAS with column-major layout: C^t = alpha * (A^t)^t*A^t + beta * C^t
 * which is what cuBLAS Syrk computes with input A^t, C^t, and OP = trans
 */

template <class _Scalar,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_C,
          class _SizeType_C,
          ::std::size_t _numRows_C,
          ::std::size_t _numCols_C,
          class _Layout_C,
          class _Accessor_C>
void __symmetric_matrix_rank_k_update_impl(
  __nvhpc_exec<__blas_exec_space>&& /* __exec */
  ,
  _Scalar __alpha,
  _Scalar __beta,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C,
  char __trans,
  char __uplo)
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(matrix_rank_k_update, BLAS);
#endif

  __bl::__blas_syrk<_ElementType_C>::__syrk(
    &__uplo,
    &__trans,
    __A.extent(0),
    __A.extent(1),
    __alpha,
    __A.data_handle(),
    __ex::linalg::__get_leading_dim(__A),
    __beta,
    __C.data_handle(),
    __ex::linalg::__get_leading_dim(__C));
}

} // namespace __nvhpc_std

#endif
