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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_SCALE_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_SCALE_BLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __bl = __blas_std;

template <class _Scalar, class _SizeType, class _ElementType, ::std::size_t... _ext, class _Layout, class _Accessor>
void __scale_impl(__nvhpc_exec<__blas_exec_space>&& /* __exec */
        ,
        const _Scalar __alpha, __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __x) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(scale, BLAS);
#endif
    static_assert(__x.rank() <= 2);

    if (__ex::linalg::__is_contiguous(__x)) {
        auto __length(__x.rank() == 1 ? __x.extent(0) : __x.extent(0) * __x.extent(1));

        __bl::__blas_scal<_ElementType>::__scal(__length, __alpha, __x.data_handle(), 1);
    } else {
        __bl::__blas_scal_2d<_ElementType>::__scal(__ex::linalg::__get_mem_row_count(__x),
                __ex::linalg::__get_mem_col_count(__x), __alpha, __x.data_handle(),
                __ex::linalg::__get_leading_dim(__x));
    }
}

}  // namespace __nvhpc_std

#endif
