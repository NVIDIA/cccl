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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_SCALE_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_SCALE_CUBLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

template <class _SyncType, class _Scalar, class _ElementType, class _SizeType, ::std::size_t... _ext, class _Layout,
        class _Accessor>
void __scale_impl(__nvhpc_exec<__cublas_exec_space<_SyncType>>&& __exec, const _Scalar __alpha,
        __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __x) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(scale, cuBLAS);
#endif
    static_assert(__x.rank() <= 2);

    // TODO: cublas only supports certain combinations for _Scalar and _ElementType
    //      could always convert __alpha to be consistent with _ElementType
    //      or could throw an exception if _Scalar and _ElementType don't match up
    if (__x.rank() == 1 || !std::is_same_v<_Layout, __ex::layout_stride>) {
        auto const __length(__x.rank() == 1 ? __x.extent(0) : __x.extent(0) * __x.extent(1));

        __cb::__check_cublas_status(
                __cb::__cublas_scal(__cb::__get_cublas_handle(), __length, __alpha, __x.data_handle()), "scale",
                "cublas_scal");
    } else {
        _ElementType const __my_alpha = static_cast<_ElementType>(__alpha);
        _ElementType const __beta = 0;
        bool const __is_transposed = !__ex::linalg::__is_column_major(__x);
        auto const __ldx = __ex::linalg::__get_leading_dim(__x);

        __cb::__check_cublas_status(
                __cb::__cublas_geam(__cb::__get_cublas_handle(), CUBLAS_OP_N, CUBLAS_OP_N,
                        __ex::linalg::__get_row_count(__x, __is_transposed),
                        __ex::linalg::__get_col_count(__x, __is_transposed), &__my_alpha, __x.data_handle(), __ldx,
                        &__beta, __x.data_handle(), __ldx, __x.data_handle(), __ldx),
                "copy", "cublas_geam");
    }

    __cb::__synchronize(std::forward<__nvhpc_exec<__cublas_exec_space<_SyncType>>>(__exec));
}

}  // namespace __nvhpc_std

#endif
