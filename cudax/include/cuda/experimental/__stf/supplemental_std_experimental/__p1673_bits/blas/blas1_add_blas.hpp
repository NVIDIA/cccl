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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_ADD_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_ADD_BLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __bl = __blas_std;

/* z = x + y */
template <class _ElementType_x, class _SizeType_x, ::std::size_t... _ext_x, class _Layout_x, class _Accessor_x,
        class _ElementType_y, class _SizeType_y, ::std::size_t... _ext_y, class _Layout_y, class _Accessor_y,
        class _ElementType_z, class _SizeType_z, ::std::size_t... _ext_z, class _Layout_z, class _Accessor_z>
void __add_impl(__nvhpc_exec<__blas_exec_space>&& __exec,
        __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x> __x,
        __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y> __y,
        __ex::mdspan<_ElementType_z, __ex::extents<_SizeType_z, _ext_z...>, _Layout_z, _Accessor_z> __z) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(add, BLAS);
#endif

    using __y_t = typename __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y>;
    using __valy_t = typename __y_t::value_type;
    using __ptry_t = typename std::unique_ptr<__valy_t, std::function<void(__valy_t*)>>;

    constexpr bool __complex_y = __la::is_complex<__valy_t>::value;

    auto const __beta = __extract_scaling_factor(__y);

    bool __conj_y = __extract_conj<__y_t>();
    char __op_y = 'N';
    bool __is_use_1d_api = true;
    std::tuple<__ptry_t, __y_t> __work_y { __ptry_t(), __y };

    if constexpr (__y.rank() == 2) {
        __extract_ops(__y, !__ex::linalg::__is_column_major(__z), &__op_y, &__conj_y);

        __is_use_1d_api = (__op_y == 'N') && __ex::linalg::__is_contiguous(__y);
    }

    __copy_impl(std::forward<__nvhpc_exec<__blas_exec_space>>(__exec), __x, __z);

    if constexpr (__complex_y) {
        if (__conj_y) {
            __work_y = __create_conjugate(__exec, __y);
        }
    }

    if (__is_use_1d_api) {
        int const __len = (__z.rank() == 1 ? __z.extent(0) : __z.extent(0) * __z.extent(1));

        __bl::__blas_axpy<_ElementType_z>::__axpy(__len, __beta,
                (__conj_y ? std::get<1>(__work_y).data_handle() : __y.data_handle()), 1, __z.data_handle(), 1);
    } else {
        __bl::__blas_axpy_2d<_ElementType_z>::__axpy(__op_y, __ex::linalg::__get_mem_row_count(__y),
                __ex::linalg::__get_mem_col_count(__y), __beta,
                (__conj_y ? std::get<1>(__work_y).data_handle() : __y.data_handle()),
                __ex::linalg::__get_leading_dim(__y), __z.data_handle(), __ex::linalg::__get_leading_dim(__z));
    }
}

}  // namespace __nvhpc_std

#endif
