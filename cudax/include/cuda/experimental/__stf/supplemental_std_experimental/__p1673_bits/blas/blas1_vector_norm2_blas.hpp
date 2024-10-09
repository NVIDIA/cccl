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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_VECTOR_NORM2_BLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS_BLAS1_VECTOR_NORM2_BLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __bl = __blas_std;

template <class _ElementType, class _SizeType, ::std::size_t... _ext, class _Layout, class _Accessor>
auto __vector_norm2_impl(__nvhpc_exec<__blas_exec_space>&& /* __exec */
                         ,
                         __ex::mdspan<_ElementType, __ex::extents<_SizeType, _ext...>, _Layout, _Accessor> __x)
  -> decltype(__ex::linalg::vector_norm2_detail::vector_norm2_return_type_deducer(__x))
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(vector_norm2, BLAS);
#endif

  using __Scalar = decltype(__ex::linalg::vector_norm2_detail::vector_norm2_return_type_deducer(__x));

  // TODO: cublas only supports certain combinations for _Scalar and _ElementType
  //      could always convert __alpha to be consistent with _ElementType
  //      or could throw an exception if _Scalar and _ElementType don'__t match up
  __Scalar __ret = __bl::BlasNrm2<_ElementType>::__nrm2(__x.extent(0), __x.data_handle(), 1);
  return __ret;
}

} // namespace __nvhpc_std

#endif
