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
#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS2_MATRIX_VECTOR_SOLVE_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS2_MATRIX_VECTOR_SOLVE_CUBLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

/*
 * Going from row-major to column-major
 * - Matrices are effectively transposed
 * - Upper-triangular matrices become lower-triangular matrices, and vice versa
 * In C with row-major layout        :  A      * x = b
 * In cuBLAS with column-major layout: (A^t)^t * x = b
 */

template <class _SyncType,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_b,
          class _SizeType_b,
          ::std::size_t _ext_b,
          class _Layout_b,
          class _Accessor_b>
void __triangular_matrix_vector_solve_impl(
  __nvhpc_exec<__cublas_exec_space<_SyncType>>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  char op_A,
  char fillMode,
  char diag,
  __ex::mdspan<_ElementType_b, __ex::extents<_SizeType_b, _ext_b>, _Layout_b, _Accessor_b> b)
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(triangular_matrix_vector_solve, cuBLAS);
#endif

  __cb::__check_cublas_status(
    __cb::__cublas_trsv(
      __cb::__get_cublas_handle(),
      __cb::__to_cublas_fill_mode(fillMode),
      __cb::__op_to_cublas_op(op_A),
      __cb::__to_cublas_diag_type(diag),
      __A.extent(0),
      __A.data_handle(),
      __ex::linalg::__get_leading_dim(__A),
      b.data_handle()),
    "matrix_vector_solve",
    "cublas_trsv");

  __cb::__synchronize(std::forward<__nvhpc_exec<__cublas_exec_space<_SyncType>>>(__exec));
}

} // namespace __nvhpc_std

#endif
