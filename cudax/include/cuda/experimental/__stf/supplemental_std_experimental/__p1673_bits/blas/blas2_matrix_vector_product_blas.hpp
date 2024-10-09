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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS2_MATRIX_VECTOR_PRODUCT_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS2_MATRIX_VECTOR_PRODUCT_BLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __bl = __blas_std;

/*
 * Going from row-major to column-major, matrices are effectively transposed
 * In C with row-major layout        : y =  A      * x
 * In cuBLAS with column-major layout: y = (A^t)^t * x
 * which is what cuBLAS Gemv computes with input A^t and OP = trans
 */

template <class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A, ::std::size_t _numCols_A, class _Layout_A,
        class _Accessor_A, class _ElementType_x, class _SizeType_x, ::std::size_t _ext_x, class _Layout_x,
        class _Accessor_x, class _ElementType_y, class _SizeType_y, ::std::size_t _ext_y, class _Layout_y,
        class _Accessor_y>
void __matrix_vector_product_impl(__nvhpc_exec<__blas_exec_space>&& /* __exec */
        ,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        char __op_A, __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x>, _Layout_x, _Accessor_x> __x,
        __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y>, _Layout_y, _Accessor_y> __y,
        _ElementType_y __beta) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(matrix_vector_product, BLAS);
#endif

    auto __alpha = __extract_scaling_factor(__A) * __extract_scaling_factor(__x);

    __bl::__blas_gemv<_ElementType_y>::__gemv(&__op_A, __ex::linalg::__get_mem_row_count(__A),
            __ex::linalg::__get_mem_col_count(__A), __alpha, __A.data_handle(), __ex::linalg::__get_leading_dim(__A),
            __x.data_handle(), 1, __beta, __y.data_handle(), 1);
}

}  // namespace __nvhpc_std

#endif
