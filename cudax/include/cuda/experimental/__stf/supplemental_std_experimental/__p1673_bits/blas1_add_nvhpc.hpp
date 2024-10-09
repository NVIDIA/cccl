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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_ADD_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS1_ADD_NVHPC_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;

/* z = x + y */
template <class _exec_space, class _ElementType_x, class _SizeType_x, ::std::size_t... _ext_x, class _Layout_x,
        class _Accessor_x, class _ElementType_y, class _SizeType_y, ::std::size_t... _ext_y, class _Layout_y,
        class _Accessor_y, class _ElementType_z, class _SizeType_z, ::std::size_t... _ext_z, class _Layout_z,
        class _Accessor_z>
void add(__nvhpc_exec<_exec_space>&& __exec,
        __ex::mdspan<_ElementType_x, __ex::extents<_SizeType_x, _ext_x...>, _Layout_x, _Accessor_x> __x,
        __ex::mdspan<_ElementType_y, __ex::extents<_SizeType_y, _ext_y...>, _Layout_y, _Accessor_y> __y,
        __ex::mdspan<_ElementType_z, __ex::extents<_SizeType_z, _ext_z...>, _Layout_z, _Accessor_z> __z) {
    constexpr bool __types_supported = __data_types_supported(__stdblas_data_type<_ElementType_x>,
            __stdblas_data_type<_ElementType_y>, __stdblas_output_data_type<_ElementType_z>);
    constexpr bool __x_supported = __input_supported<_Layout_x, _Accessor_x>();
    constexpr bool __y_supported = __input_supported<_Layout_y, _Accessor_y>();
    constexpr bool __z_supported = __output_supported<_Layout_z, _Accessor_z>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
    __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
    __STDBLAS_STATIC_ASSERT_INPUT(__x_supported, __x);
    __STDBLAS_STATIC_ASSERT_INPUT(__y_supported, __y);
    __STDBLAS_STATIC_ASSERT_OUTPUT(__z_supported, __z);
#endif

    if constexpr (__types_supported && __x_supported && __y_supported && __z_supported) {
        __add_impl(std::forward<__nvhpc_exec<_exec_space>>(__exec), __x, __y, __z);
    } else {
#ifdef STDBLAS_VERBOSE
        __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(add);
#endif
        __ex::linalg::add(std::execution::seq, __x, __y, __z);
    }
}

}  // namespace __nvhpc_std

#endif
