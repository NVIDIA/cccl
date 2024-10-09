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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_TRIANGULAR_MATRIX_MATRIX_SOLVE_NVHPC_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;

// TODO Add triangular_matrix_matrix_[left|right]_solve( in __A, inout __B ) when the base impl is available
template <class _exec_space, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A,
        ::std::size_t _numCols_A, class _Layout_A, class _Accessor_A, class Side, class _Triangle,
        class _DiagonalStorage, class _ElementType_B, class _SizeType_B, ::std::size_t _numRows_B,
        ::std::size_t _numCols_B, class _Layout_B, class _Accessor_B, class _ElementType_X, class _SizeType_X,
        ::std::size_t _numRows_X, ::std::size_t _numCols_X, class _Layout_X, class _Accessor_X>
void __triangular_matrix_matrix_solve_impl(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        Side __s, _Triangle __t, _DiagonalStorage __d,
        __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
        __ex::mdspan<_ElementType_X, __ex::extents<_SizeType_X, _numRows_X, _numCols_X>, _Layout_X, _Accessor_X> __X) {
    constexpr bool __types_supported = __data_types_supported(__stdblas_data_type<_ElementType_A>,
            __stdblas_data_type<_ElementType_B>, __stdblas_output_data_type<_ElementType_X>);
    constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
    constexpr bool __B_supported = __input_supported<_Layout_B, _Accessor_B>();
    constexpr bool __X_supported = __output_supported<_Layout_X, _Accessor_X>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
    __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
    __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
    __STDBLAS_STATIC_ASSERT_INPUT(__B_supported, __B);
    __STDBLAS_STATIC_ASSERT_OUTPUT(__X_supported, __X);
#endif

    if constexpr (__types_supported && __A_supported && __B_supported && __X_supported) {
        using __A_t = typename __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>,
                _Layout_A, _Accessor_A>;
        using __valA_t = typename __A_t::value_type;
        using __ptrA_t = typename std::unique_ptr<__valA_t, std::function<void(__valA_t*)>>;

        bool const __is_operate_on_transposed = (!__ex::linalg::__is_column_major(__B));
        char __op_A;
        bool __conj_A;
        std::tuple<__ptrA_t, __A_t> __work_A { __ptrA_t(), __A };

        __extract_ops(__A, __is_operate_on_transposed, &__op_A, &__conj_A);

        copy(std::forward<__nvhpc_exec<_exec_space>>(__exec), __B, __X);

        if constexpr (__la::is_complex<__valA_t>::value) {
            if (__conj_A) {
                __work_A = __create_conjugate(__exec, __A);
            }
        }

        _ElementType_X __alpha(_ElementType_B { 1 } / __extract_scaling_factor(__A));

        __triangular_matrix_matrix_solve_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec), __alpha,
                (__conj_A ? std::get<1>(__work_A) : __A), __op_A, __ex::linalg::__get_fill_mode(__A, __t),
                __ex::linalg::__get_side_mode(__B, __s), __ex::linalg::__get_diag_type(__d), __is_operate_on_transposed,
                __X);
    } else if constexpr (std::is_same_v<Side, __ex::linalg::left_side_t>) {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(triangular_matrix_matrix_left_solve);
#endif
        __ex::linalg::triangular_matrix_matrix_left_solve(std::execution::seq, __A, __t, __d, __B, __X);
    } else {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(triangular_matrix_matrix_right_solve);
#endif
        __ex::linalg::triangular_matrix_matrix_right_solve(std::execution::seq, __A, __t, __d, __B, __X);
    }
}

template <class _exec_space, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A,
        ::std::size_t _numCols_A, class _Layout_A, class _Accessor_A, class _Triangle, class _DiagonalStorage,
        class _ElementType_B, class _SizeType_B, ::std::size_t _numRows_B, ::std::size_t _numCols_B, class _Layout_B,
        class _Accessor_B, class _ElementType_X, class _SizeType_X, ::std::size_t _numRows_X, ::std::size_t _numCols_X,
        class _Layout_X, class _Accessor_X>
void triangular_matrix_matrix_left_solve(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        _Triangle __t, _DiagonalStorage __d,
        __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
        __ex::mdspan<_ElementType_X, __ex::extents<_SizeType_X, _numRows_X, _numCols_X>, _Layout_X, _Accessor_X> __X) {
    __triangular_matrix_matrix_solve_impl(
            std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __ex::linalg::left_side_t {}, __t, __d, __B, __X);
}

template <class _exec_space, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A,
        ::std::size_t _numCols_A, class _Layout_A, class _Accessor_A, class _Triangle, class _DiagonalStorage,
        class _ElementType_B, class _SizeType_B, ::std::size_t _numRows_B, ::std::size_t _numCols_B, class _Layout_B,
        class _Accessor_B, class _ElementType_X, class _SizeType_X, ::std::size_t _numRows_X, ::std::size_t _numCols_X,
        class _Layout_X, class _Accessor_X>
void triangular_matrix_matrix_right_solve(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        _Triangle __t, _DiagonalStorage __d,
        __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
        __ex::mdspan<_ElementType_X, __ex::extents<_SizeType_X, _numRows_X, _numCols_X>, _Layout_X, _Accessor_X> __X) {
    __triangular_matrix_matrix_solve_impl(
            std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __ex::linalg::right_side_t {}, __t, __d, __B, __X);
}

}  // namespace __nvhpc_std

#endif
