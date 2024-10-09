/*
 * SPDX-FileCopyrightText: Copyright (c) 2021 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: LicenseRef-NvidiaProprietary
 *
 * NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
 * property and proprietary rights in and to this material, related
 * documentation and any modifications thereto. Any use, reproduction,
 * disclosure or distribution of this material and related documentation
 * without an express license agreement from NVIDIA CORPORATION or
 * its affiliates is strictly prohibited.
 */

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_VECTOR_NORM2_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_VECTOR_NORM2_CUBLAS_HPP_

namespace __nvhpc_std {

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

template <class _ElementType, class _SizeType, ::std::size_t... _ext, class _Layout, class _Accessor>
auto __vector_norm2_impl(__nvhpc_exec<__cublas_exec_space<__nvhpc_sync>>&& /* __exec */
        ,
        __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __x)
        -> decltype(__ex::linalg::vector_norm2_detail::vector_norm2_return_type_deducer(__x)) {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_BACKEND_MESSAGE(vector_norm2, cuBLAS);
#endif

    using __Scalar = decltype(__ex::linalg::vector_norm2_detail::vector_norm2_return_type_deducer(__x));

    __Scalar __ret = 0;

    // TODO: cublas only supports certain combinations for _Scalar and _ElementType
    //      could always convert __alpha to be consistent with _ElementType
    //      or could throw an exception if _Scalar and _ElementType don't match up
    __cb::__check_cublas_status(
            __cb::__cublas_nrm2(__cb::__get_cublas_handle(), __x.extent(0), __x.data_handle(), &__ret), "vector_norm2",
            "cublas_nrm2");

    // No sync is needed since __cublas_nrm2 is blocking by default

    return __ret;
}
}  // namespace __nvhpc_std

#endif
