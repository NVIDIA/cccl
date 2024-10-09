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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_DOT_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_DOT_BLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __bl = __blas_std;

template <class _ElementType1, class _SizeType1, ::std::size_t... _ext1, class _Layout1, class _Accessor1,
        class _ElementType2, class _SizeType2, ::std::size_t... _ext2, class _Layout2, class _Accessor2>
auto __dot_impl(__nvhpc_exec<__blas_exec_space>&& /* __exec */
        ,
        __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1...>, _Layout1, _Accessor1> __v1,
        __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2...>, _Layout2, _Accessor2> __v2)
        -> decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2)) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(dot, BLAS);
#endif

    using __Scalar = decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2));

    __Scalar __ret =
            __bl::__blas_dot<_ElementType1>::__dot(__v1.extent(0), __v1.data_handle(), 1, __v2.data_handle(), 1);

    return __ret;
}

template <class _ElementType1, class _SizeType1, ::std::size_t... _ext1, class _Layout1, class _Accessor1,
        class _ElementType2, class _SizeType2, ::std::size_t... _ext2, class _Layout2, class _Accessor2>
auto __dotc_impl(__nvhpc_exec<__blas_exec_space>&& /* __exec */
        ,
        __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1...>, _Layout1, _Accessor1> __v1,
        __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2...>, _Layout2, _Accessor2> __v2)
        -> decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2)) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(dotc, BLAS);
#endif

    using __Scalar = decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2));

    __Scalar __ret =
            __bl::__blas_dotc<_ElementType1>::__dotc(__v1.extent(0), __v1.data_handle(), 1, __v2.data_handle(), 1);

    return __ret;
}

}  // namespace __nvhpc_std

#endif
