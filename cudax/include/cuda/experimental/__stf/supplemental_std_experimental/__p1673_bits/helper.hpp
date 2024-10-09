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

#ifndef LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_HELPER_HPP_
#define LINALG_INCLUDE_EXPERIMENTAL___P1673_BITS_HELPER_HPP_

namespace std { namespace experimental { inline namespace __p1673_version_0 { namespace linalg {

template <typename T>
struct is_complex : std::false_type {};
template <typename T>
struct is_complex<std::complex<T>> : std::true_type {};

/**
 * Returns if the matrix is stored in the column-major layout in memory
 */
template <class mdspan_t>
bool __is_column_major(mdspan_t A) {
    using layout_t = typename mdspan_t::layout_type;

    assert(A.rank() == 2, "Internal error, '__is_column_major' only works for rank-2 mdspan's");

    if (!A.is_unique() || !A.is_strided()) {
        return false;
    }

    return (A.stride(0) > 1 || A.stride(1) > 1
                    // In most cases, stride(0) == 1 means it's column-major
                    ? A.stride(0) == 1
                    // Special case of stride(0) == stride(1) == 1 for single-row column-major and single-column
                    // row-major matrices Check extent(0) instead of stride(0)
                    : A.extent(0) == 1);
}

template <class mdspan_t, class triangle_t>
char __get_fill_mode(mdspan_t A, triangle_t /* t */) {
    return (__is_column_major(A) == std::is_same_v<triangle_t, lower_triangle_t> ? 'L' : 'U');
}

template <class mdspan_t, class side_t>
constexpr char __get_side_mode(mdspan_t A, side_t /* s */) {
    return (std::is_same_v<side_t, left_side_t> == __is_column_major(A) ? 'L' : 'R');
}

template <class diag_t>
constexpr char __get_diag_type(diag_t /* d */) {
    return (std::is_same<diag_t, implicit_unit_diagonal_t>::value ? 'U' : 'N');
}

/**
 * Get row count of matrix to be passed to cuBLAS/BLAS API
 * isOperateOnTransposed is true when the API operates on the transpose
 * of matrices, which is the case when the output matrix is in row-major
 * memory layout.
 */
template <class mdspan_t>
int __get_row_count(const mdspan_t& A, bool isOperateOnTransposed) {
    if constexpr (A.rank() == 1) {
        return A.extent(0);
    } else {
        return (isOperateOnTransposed ? A.extent(1) : A.extent(0));
    }
}

/**
 * Get column count of matrix to be passed to cuBLAS/BLAS API
 * isOperateOnTransposed is true when the API operates on the transpose
 * of matrices, which is the case when the output matrix is in row-major
 * memory layout.
 */
template <class mdspan_t>
int __get_col_count(const mdspan_t& A, bool isOperateOnTransposed) {
    if constexpr (A.rank() == 1) {
        return 1;
    } else {
        return (isOperateOnTransposed ? A.extent(0) : A.extent(1));
    }
}

template <class mdspan_t>
constexpr int __get_mem_row_count(const mdspan_t& A) {
    return (__is_column_major(A) ? A.extent(0) : A.extent(1));
}

template <class mdspan_t>
constexpr int __get_mem_col_count(const mdspan_t& A) {
    return (__is_column_major(A) ? A.extent(1) : A.extent(0));
}

/**
 * Get the leading dimension of matrix storage
 *
 * The "leading dimension" for cuBLAS/BLAS calls is the stride along the slow
 * dimension. The stride along this dimension is always bigger than that along
 * the fast dimension, which is 1.
 */
template <class mdspan_t>
int __get_leading_dim(const mdspan_t& A) {
    if constexpr (A.rank() == 1) {
        return A.extent(0);
    } else {
        return (__is_column_major(A) ? A.stride(1) : A.stride(0));
    }
}

template <class mdspan_t>
bool __is_contiguous(const mdspan_t& A) {
    return (A.mapping().required_span_size() == A.size() && A.is_strided());
}

template <class mdspan_t>
int __get_conjugate_length(const mdspan_t& A) {
    if (A.rank() == 1) {
        return A.extent(0);
    }

    assert(A.extent(0) > 0 && A.extent(1) > 0, "Extent has to be greater than 0");

    return (__is_column_major(A) ? (A.extent(1) - 1) * A.stride(1) + A.extent(0)
                                 : (A.extent(0) - 1) * A.stride(0) + A.extent(1));
}

}}}}  // namespace std::experimental::__p1673_version_0::linalg

#endif
