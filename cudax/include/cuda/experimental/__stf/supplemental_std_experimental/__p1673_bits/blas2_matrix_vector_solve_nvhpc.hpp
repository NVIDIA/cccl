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
#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS2_MATRIX_VECTOR_SOLVE_NVHPC_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __la = std::experimental::linalg;

/*
 * Going from row-major to column-major
 * - Matrices are effectively transposed
 * - Upper-triangular matrices become lower-triangular matrices, and vice versa
 * In C with row-major layout        :  A      * x = b
 * In cuBLAS with column-major layout: (A^t)^t * x = b
 */

template <class _exec_space, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A,
        ::std::size_t _numCols_A, class _Layout_A, class _Accessor_A, class _Triangle, class _DiagonalStorage,
        class _ElementType_b, class _SizeType_b, ::std::size_t _ext_b, class _Layout_b, class _Accessor_b>
inline void __triangular_matrix_vector_solve_nvhpc(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        _Triangle __t, _DiagonalStorage __d,
        __ex::mdspan<_ElementType_b, __ex::extents<_SizeType_b, _ext_b>, _Layout_b, _Accessor_b> __b) {
    using __A_t = typename __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A,
            _Accessor_A>;
    using __valA_t = typename __A_t::value_type;
    using __ptrA_t = typename std::unique_ptr<__valA_t, std::function<void(__valA_t*)>>;

    char __op_A;
    bool __conj_A;
    std::tuple<__ptrA_t, __A_t> __work_A { __ptrA_t(), __A };

    __extract_ops<__A_t>(__A, false /* __is_transposed */, &__op_A, &__conj_A);

    if constexpr (__la::is_complex<__valA_t>::value) {
        if (__conj_A) {
            __work_A = __create_conjugate(__exec, __A);
        }
    }

    __triangular_matrix_vector_solve_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec),
            (__conj_A ? std::get<1>(__work_A) : __A), __op_A, __la::__get_fill_mode(__A, __t),
            __la::__get_diag_type(__d), __b);

    if constexpr (__is_scaled<__A_t>()) {
        scale(std::forward<__nvhpc_exec<_exec_space>>(__exec),
                static_cast<_ElementType_A>(1) / __extract_scaling_factor(__A), __b);
    }
}

template <class _exec_space, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A,
        ::std::size_t _numCols_A, class _Layout_A, class _Accessor_A, class _Triangle, class _DiagonalStorage,
        class _ElementType_b, class _SizeType_b, ::std::size_t _ext_b, class _Layout_b, class _Accessor_b>
void triangular_matrix_vector_solve(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        _Triangle __t, _DiagonalStorage __d,
        __ex::mdspan<_ElementType_b, __ex::extents<_SizeType_b, _ext_b>, _Layout_b, _Accessor_b> __b) {
    constexpr bool __types_supported =
            __data_types_supported(__stdblas_data_type<_ElementType_A>, __stdblas_output_data_type<_ElementType_b>);
    constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
    constexpr bool __b_supported = __output_supported<_Layout_b, _Accessor_b>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
    __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
    __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
    __STDBLAS_STATIC_ASSERT_OUTPUT(__b_supported, __b);
#endif

    if constexpr (__types_supported && Asupported && __b_supported) {
        __triangular_matrix_vector_solve_nvhpc(std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __t, __d, __b);
    } else {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(triangular_matrix_vector_solve);
#endif
        __la::__triangular_matrix_vector_solve_nvhpc(std::execution::seq, __A, __t, __d, __b);
    }
}

template <class _exec_space, class _ElementType_A, class _SizeType_A, ::std::size_t _numRows_A,
        ::std::size_t _numCols_A, class _Layout_A, class _Accessor_A, class _Triangle, class _DiagonalStorage,
        class _ElementType_b, class _SizeType_b, ::std::size_t _ext_b, class _Layout_b, class _Accessor_b,
        class _ElementType_x, class _SizeType_x, ::std::size_t _ext_x, class _Layout_x, class _Accessor_x>
void triangular_matrix_vector_solve(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
        _Triangle __t, _DiagonalStorage __d,
        __ex::mdspan<_ElementType_b, __ex::extents<_SizeType_b, _ext_b>, _Layout_b, _Accessor_b> __b,
        __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x>, _Layout_x, _Accessor_x> __x) {
    constexpr bool __types_supported = __data_types_supported(__stdblas_data_type<_ElementType_A>,
            __stdblas_data_type<_ElementType_b>, __stdblas_output_data_type<_ElementType_x>);
    constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
    constexpr bool __b_supported = __input_supported<_Layout_b, _Accessor_b>();
    constexpr bool __x_supported = __output_supported<_Layout_x, _Accessor_x>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
    __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
    __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
    __STDBLAS_STATIC_ASSERT_INPUT(__b_supported, __b);
    __STDBLAS_STATIC_ASSERT_OUTPUT(__x_supported, __x);
#endif

    if constexpr (__types_supported && __A_supported && __b_supported && __x_supported) {
        copy(std::forward<__nvhpc_exec<_exec_space>>(__exec), __b, __x);

        __triangular_matrix_vector_solve_nvhpc(std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __t, __d, __x);
    } else {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(triangular_matrix_vector_solve);
#endif
        __la::triangular_matrix_vector_solve(std::execution::seq, __A, __t, __d, __b, __x);
    }
}

}  // namespace __nvhpc_std

#endif
