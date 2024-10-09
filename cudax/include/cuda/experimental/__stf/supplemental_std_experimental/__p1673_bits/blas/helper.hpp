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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_HELPER_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_HELPER_HPP_

namespace __blas_std {

namespace __ex = std::experimental;

template <class triangle_t, class _MDSpan>
constexpr char getBlasFillMode() {
    return (std::is_same<triangle_t, __ex::linalg::lower_triangle_t>::value != __nvhpc_std::__extract_trans<_MDSpan>()
                    ? 'U'
                    : 'L');
}

template <class diag_t>
constexpr char getBlasDiagType() {
    return (std::is_same<diag_t, __ex::linalg::implicit_unit_diagonal_t>::value ? 'U' : 'N');
}

template <class in_matrix_t>
constexpr char getBlasOp(bool __is_operate_on_transposed) {
    return (__nvhpc_std::__extract_trans<in_matrix_t>() != __is_operate_on_transposed
                    ? (__nvhpc_std::__extract_conj<in_matrix_t>() ? 'C' : 'T')
                    : 'N');
}

// TODO expand this to suppor the case there left_side_t corresponds to CUBLAS_SIDE_LEFT
// in column-major layout
template <class side_t>
constexpr char getSide() {
    return (std::is_same_v<side_t, __ex::linalg::left_side_t> ? 'R' : 'L');
}

// Get n_row of matrix w/o OP in column-major layout
template <class in_matrix_t>
constexpr int __get_row_count(const in_matrix_t& __A) {
    return (getBlasOp<in_matrix_t>(false) == 'N' ? __A.extent(1) : __A.extent(0));
}

// Get n_col of matrix w/o OP in column-major layout
template <class in_matrix_t>
constexpr int __get_col_count(const in_matrix_t& __A) {
    return (getBlasOp<in_matrix_t>(false) == 'N' ? __A.extent(0) : __A.extent(1));
}

// Get the leading dimension of matrix storage
// The "leading dimension" for cuBLAS/BLAS calls is the stride along the slow
// dimension. It is always bigger than the stride along the fast dimension, which is 1.
template <class in_matrix_t>
constexpr int __get_leading_dim(const in_matrix_t& __A) {
    return (std::max(__A.stride(0), __A.stride(1)));
}

}  // namespace __blas_std

#endif
