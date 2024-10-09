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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS3_MATRIX_PRODUCT_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS3_MATRIX_PRODUCT_CUBLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

template <class _SyncType, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A, ::std::size_t _numCols_A,
        class _Layout_A, class _Accessor_A, class _ElementType_B, class _SizeType_B, ::std::size_t _numRows_B,
        ::std::size_t _numCols_B, class _Layout_B, class _Accessor_B, class _ElementType_C, class _SizeType_C,
        ::std::size_t _numRows_C, ::std::size_t _numCols_C, class _Layout_C, class _Accessor_C>
void __matrix_product_impl(__nvhpc_exec<__cublas_exec_space<_SyncType>>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        char __op_A,
        __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
        char __op_B,
        __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C,
        _ElementType_C __alpha, _ElementType_C __beta, bool __is_transposed) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(matrix_product, cuBLAS);
#endif

    using __A_t = typename __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A,
            _Accessor_A>;
    using __B_t = typename __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B,
            _Accessor_B>;
    using __C_t = typename __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C,
            _Accessor_C>;

    __cb::__check_cublas_status(
            __cb::__cublas_gemm(__cb::__get_cublas_handle(), __cb::__op_to_cublas_op(__op_A),
                    __cb::__op_to_cublas_op(__op_B), __ex::linalg::__get_row_count(__C, __is_transposed),
                    __ex::linalg::__get_col_count(__C, __is_transposed),
                    __ex::linalg::__get_row_count(__B, __is_transposed), &__alpha, __A.data_handle(),
                    __ex::linalg::__get_leading_dim<__A_t>(__A), __B.data_handle(),
                    __ex::linalg::__get_leading_dim<__B_t>(__B), &__beta, __C.data_handle(),
                    __ex::linalg::__get_leading_dim<__C_t>(__C)),
            "matrix_product", "cublas_gemm");

    __cb::__synchronize(std::forward<__nvhpc_exec<__cublas_exec_space<_SyncType>>>(__exec));
}

}  // namespace __nvhpc_std

#endif
