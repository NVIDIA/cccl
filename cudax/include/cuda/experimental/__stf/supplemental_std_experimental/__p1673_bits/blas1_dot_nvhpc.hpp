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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_DOT_NVHPC_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;

/* returns v1 dot v2 */
template <class _exec_space, class _ElementType1, class _SizeType1, ::std::size_t _ext1, class _Layout1,
        class _Accessor1, class _ElementType2, class _SizeType2, ::std::size_t _ext2, class _Layout2, class _Accessor2>
auto dot(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1>, _Layout1, _Accessor1> __v1,
        __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2>, _Layout2, _Accessor2> __v2)
        -> decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2)) {
    constexpr bool __types_supported =
            __data_types_supported(__stdblas_data_type<_ElementType1>, __stdblas_data_type<_ElementType2>);
    constexpr bool __v1_supported = __input_supported<_Layout1, _Accessor1>();
    constexpr bool __v2_supported = __input_supported<_Layout2, _Accessor2>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
    __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
    __STDBLAS_STATIC_ASSERT_INPUT(__v1_supported, __v1);
    __STDBLAS_STATIC_ASSERT_INPUT(__v2_supported, __v2);
#endif

    if constexpr (__types_supported && __v1_supported && __v2_supported) {
        using __Scalar = decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2));
        using __t1 = __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1>, _Layout1, _Accessor1>;
        using __t2 = __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2>, _Layout2, _Accessor2>;

        constexpr bool __conj1 = __apply_conjugate<__t1>(false);
        constexpr bool __conj2 = __apply_conjugate<__t2>(false);
        constexpr bool __call_dot =
                (!__ex::linalg::impl::is_complex<typename __t1::value_type>::value || __conj1 == __conj2);

        __Scalar __alpha1 = __extract_scaling_factor(__v1);
        __Scalar __alpha2 = __extract_scaling_factor(__v2);

        __Scalar __ret;
        if constexpr (__call_dot) {
            __ret = __dot_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec), __v1, __v2);
        } else {
            __ret = __dotc_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec), __v1, __v2);
        }

        if constexpr (__conj2) {
            return __alpha1 * __alpha2 * std::conj(__ret);
        } else {
            return __alpha1 * __alpha2 * __ret;
        }
    } else {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(dot);
#endif
        return __ex::linalg::dot(std::execution::seq, __v1, __v2);
    }
}

/* returns v1 dot v2 + init */
template <class _exec_space, class _ElementType1, class _SizeType1, ::std::size_t _ext1, class _Layout1,
        class _Accessor1, class _ElementType2, class _SizeType2, ::std::size_t _ext2, class _Layout2, class _Accessor2,
        class _Scalar>
_Scalar dot(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1>, _Layout1, _Accessor1> __v1,
        __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2>, _Layout2, _Accessor2> __v2, _Scalar __init) {
    return __init + dot(std::forward<__nvhpc_exec<_exec_space>>(__exec), __v1, __v2);
}

}  // namespace __nvhpc_std

#endif
