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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_VECTOR_NORM2_NVHPC_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;

template <class _exec_space, class _ElementType, class _SizeType, ::std::size_t _ext0, class _Layout, class _Accessor>
auto vector_norm2(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext0>, _Layout, _Accessor> __x)
        -> decltype(__ex::linalg::vector_norm2_detail::vector_norm2_return_type_deducer(__x)) {
    constexpr bool __type_supported = __data_type_supported(__stdblas_data_type<_ElementType>);
    constexpr bool __x_supported = __input_supported<_Layout, _Accessor>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
    __STDBLAS_STATIC_ASSERT_TYPES(__type_supported);
    __STDBLAS_STATIC_ASSERT_INPUT(__x_supported, __x);
#endif

    if constexpr (__type_supported && __x_supported) {
        using _Scalar = decltype(__ex::linalg::vector_norm2_detail::vector_norm2_return_type_deducer(__x));

        _Scalar __ret = __vector_norm2_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec), __x);

        return std::abs(__extract_scaling_factor(__x)) * __ret;
    } else {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(vector_norm2);
#endif
        return __ex::linalg::vector_norm2(std::execution::seq, __x);
    }
}

template <class _exec_space, class _ElementType, class _SizeType, ::std::size_t _ext0, class _Layout, class _Accessor,
        class _Scalar>
_Scalar vector_norm2(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext0>, _Layout, _Accessor> __x, _Scalar __init) {
    return __init + vector_norm2(std::forward<__nvhpc_exec<_exec_space>>(__exec), __x);
}

}  // namespace __nvhpc_std

#endif
