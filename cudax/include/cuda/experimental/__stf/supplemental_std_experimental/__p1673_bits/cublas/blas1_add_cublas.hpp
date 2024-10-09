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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_ADD_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_ADD_CUBLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

/* z = x + y */
template <class _SyncType, class _ElementType_x, class _SizeType_x, ::std::size_t... _ext_x, class _Layout_x,
        class _Accessor_x, class _ElementType_y, class _SizeType_y, ::std::size_t... _ext_y, class _Layout_y,
        class _Accessor_y, class _ElementType_z, class _SizeType_z, ::std::size_t... _ext_z, class _Layout_z,
        class _Accessor_z>
void __add_impl(__nvhpc_exec<__cublas_exec_space<_SyncType>>&& __exec,
        __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x> __x,
        __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y> __y,
        __ex::mdspan<_ElementType_z, __ex::extents<_SizeType_z, _ext_z...>, _Layout_z, _Accessor_z> __z) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(add, cuBLAS);
#endif

    // TODO extent and elementtype compatibility checks
    using __x_t = typename __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x>;
    using __valx_t = typename __x_t::value_type;
    using __ptrx_t = typename std::unique_ptr<__valx_t, std::function<void(__valx_t*)>>;
    using __y_t = typename __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y>;
    using __valy_t = typename __y_t::value_type;
    using __ptry_t = typename std::unique_ptr<__valy_t, std::function<void(__valy_t*)>>;
    _ElementType_z __alpha = __extract_scaling_factor(__x);
    _ElementType_z __beta = __extract_scaling_factor(__y);
    cublasHandle_t __handle = __cb::__get_cublas_handle();

    constexpr bool __complex_x = __la::is_complex<__valx_t>::value;
    constexpr bool __complex_y = __la::is_complex<__valy_t>::value;

    // TODO: cublas only supports certain combinations for _Scalar and _ElementType
    //      could always convert __alpha to be consistent with _ElementType
    //      or could throw an exception if _Scalar and _ElementType don't match up

    if constexpr (__x.rank() == 1) {
        int const __len = __z.extent(0);
        constexpr bool __conj_x = __extract_conj<__x_t>();
        constexpr bool __conj_y = __extract_conj<__y_t>();

        __cb::__check_cublas_status(__cb::__cublas_geam(__handle, __cb::__get_cublas_op_vector<__x_t>(),
                                            __cb::__get_cublas_op_vector<__y_t>(), __len, 1, &__alpha,
                                            __x.data_handle(), (__conj_x ? 1 : __len), &__beta, __y.data_handle(),
                                            (__conj_y ? 1 : __len), __z.data_handle(), __len),
                "add", "cublas_geam");
    } else {
        int const __length_conj_z = __ex::linalg::__get_conjugate_length(__z);
        bool const __is_transposed = (!__ex::linalg::__is_column_major(__z));
        char __op_x, __op_y;
        bool __conj_x, __conj_y, __conj_z = false;
        bool const __contiguous = __ex::linalg::__is_contiguous(__z);
        std::tuple<__ptrx_t, __x_t> __work_x { __ptrx_t(), __x };
        std::tuple<__ptry_t, __y_t> __work_y { __ptry_t(), __y };

        __extract_ops(__x, __is_transposed, &__op_x, &__conj_x);
        __extract_ops(__y, __is_transposed, &__op_y, &__conj_y);

        // z = conj(x) + conj(y) -> z = conj(x + y) when z is contiguous in memory
        // Otherwise create work array(s) to hold conj(x) and conj(y)
        if (__conj_x && __conj_y && __contiguous) {
            __conj_x = false;
            __conj_y = false;
            __conj_z = true;
        }

        // If we take the conjugate of the output, note that alpha*conj(x) + beta*conj(y) <- conj(conj(alpha)*x +
        // conj(beta)*y)
        if constexpr (__complex_x) {
            if (__conj_x) {
                __work_x = __create_conjugate(__exec, __x);
            }
            if (__conj_z) {
                __alpha = std::conj(__alpha);
            }
        }

        if constexpr (__complex_y) {
            if (__conj_y) {
                __work_y = __create_conjugate(__exec, __y);
            }
            if (__conj_z) {
                __beta = std::conj(__beta);
            }
        }

        __cb::__check_cublas_status(
                __cb::__cublas_geam(__handle, __cb::__op_to_cublas_op(__op_x), __cb::__op_to_cublas_op(__op_y),
                        __ex::linalg::__get_row_count(__z, __is_transposed),
                        __ex::linalg::__get_col_count(__z, __is_transposed), &__alpha,
                        (__conj_x ? std::get<1>(__work_x).data_handle() : __x.data_handle()),
                        __ex::linalg::__get_leading_dim(__x), &__beta,
                        (__conj_y ? std::get<1>(__work_y).data_handle() : __y.data_handle()),
                        __ex::linalg::__get_leading_dim(__y), __z.data_handle(), __ex::linalg::__get_leading_dim(__z)),
                "add", "cublas_geam");

        if (__conj_z) {
            __cb::__check_cublas_status(
                    __cb::__cublas_conj(__handle, __length_conj_z, __z.data_handle()), "add", "cublas_conj");
        }
    }

    __cb::__synchronize(std::forward<__nvhpc_exec<__cublas_exec_space<_SyncType>>>(__exec));
}

}  // namespace __nvhpc_std

#endif
