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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_DOT_CUBLAS_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_CUBLAS_BLAS1_DOT_CUBLAS_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __cb = __cublas_std;

template <class _ElementType1,
          class _SizeType1,
          ::std::size_t... _ext1,
          class _Layout1,
          class _Accessor1,
          class _ElementType2,
          class _SizeType2,
          ::std::size_t... _ext2,
          class _Layout2,
          class _Accessor2>
auto __dot_impl(__nvhpc_exec<__cublas_exec_space<__nvhpc_sync>>&& /* __exec */
                ,
                __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1...>, _Layout1, _Accessor1> __v1,
                __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2...>, _Layout2, _Accessor2> __v2)
  -> decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2))
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(dot, cuBLAS);
#endif

  using _Scalar = decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2));

  _Scalar __ret = 0;

  // TODO extent and elementtype compatibility checks

  // TODO: cublas only supports certain combinations for _Scalar and _ElementType
  //      could always convert __alpha to be consistent with _ElementType
  //      or could throw an exception if _Scalar and _ElementType don't match up
  __cb::__check_cublas_status(
    __cb::__cublas_dot(__cb::__get_cublas_handle(), __v1.extent(0), __v1.data_handle(), __v2.data_handle(), &__ret),
    "dot",
    "cublas_dot");

  // No sync is needed since __cublas_dot is blocking by default

  return __ret;
}

template <class _SyncType,
          class _ElementType1,
          class _SizeType1,
          ::std::size_t... _ext1,
          class _Layout1,
          class _Accessor1,
          class _ElementType2,
          class _SizeType2,
          ::std::size_t... _ext2,
          class _Layout2,
          class _Accessor2>
auto __dotc_impl(__nvhpc_exec<__cublas_exec_space<_SyncType>>&& /* __exec */
                 ,
                 __ex::mdspan<_ElementType1, __ex::extents<_SizeType1, _ext1...>, _Layout1, _Accessor1> __v1,
                 __ex::mdspan<_ElementType2, __ex::extents<_SizeType2, _ext2...>, _Layout2, _Accessor2> __v2)
  -> decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2))
{
#ifdef STDBLAS_VERBOSE
  __STDBLAS_BACKEND_MESSAGE(dotc, cuBLAS);
#endif

  using _Scalar = decltype(__ex::linalg::dot_detail::dot_return_type_deducer(__v1, __v2));

  _Scalar __ret = 0;

  // TODO extent and elementtype compatibility checks

  // TODO: cublas only supports certain combinations for _Scalar and _ElementType
  //      could always convert __alpha to be consistent with _ElementType
  //      or could throw an exception if _Scalar and _ElementType don't match up
  __cb::__check_cublas_status(
    __cb::__cublas_dotc(__cb::__get_cublas_handle(), __v1.extent(0), __v1.data_handle(), __v2.data_handle(), &__ret),
    "dotc",
    "cublas_dotc");

  // No sync is needed since __cublas_dotc is blocking by default

  return __ret;
}

} // namespace __nvhpc_std

#endif
