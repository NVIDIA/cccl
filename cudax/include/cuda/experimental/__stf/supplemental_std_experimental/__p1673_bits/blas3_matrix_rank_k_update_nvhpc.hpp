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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_K_UPDATE_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_RANK_K_UPDATE_NVHPC_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __la = std::experimental::linalg;

template <class _exec_space,
          class _Scalar,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_C,
          class _SizeType_C,
          ::std::size_t _numRows_C,
          ::std::size_t _numCols_C,
          class _Layout_C,
          class _Accessor_C,
          class _Triangle>
void symmetric_matrix_rank_k_update(
  __nvhpc_exec<_exec_space>&& __exec,
  _Scalar __alpha,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C,
  _Triangle __t)
{
  constexpr __types_supported =
    __data_types_supported(__stdblas_data_type<_ElementType_A>, __stdblas_output_data_type<_ElementType_C>);
  constexpr __A_supported = __input_supported<_Layout_A, _Accessor_A>();
  constexpr __C_supported = __output_supported<_Layout_C, _Accessor_C>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
  __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
  __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
  __STDBLAS_STATIC_ASSERT_OUTPUT(__C_supported, __C);
#endif

  if constexpr (__types_supported && __A_supported && __C_supported)
  {
    using __A_t =
      typename __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A>;
    using __valA_t = typename __A_t::value_type;
    using __ptrA_t = typename std::unique_ptr<__valA_t, std::function<void(__valA_t*)>>;

    char __op_A;
    bool __conj_A;
    std::tuple<__ptrA_t, __A_t> __work_A{__ptrA_t(), __A};

    __extract_ops_no_C(__A, false /* __is_operate_on_transposed */, &__op_A, &__conj_A);

    // TODO check for precision loss
    _Scalar const __my_alpha(__extract_scaling_factor(__A));

    if constexpr (__la::is_complex<__valA_t>::value)
    {
      if (__conj_A)
      {
        __work_A = __create_conjugate(__exec, __A);
      }
    }

    __symmetric_matrix_rank_k_update_impl(
      std::forward<__nvhpc_exec<_exec_space>>(__exec),
      _ElementType_C{__alpha * __my_alpha * __my_alpha},
      _ElementType_C{__extract_scaling_factor(__C)},
      (__conj_A ? std::get<1>(__work_A) : __A),
      __C,
      __op_A,
      __ex::linalg::__get_fill_mode(__C, __t));
  }
  else
  {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(symmetric_matrix_rank_k_update);
#endif
    __ex::linalg::symmetric_matrix_rank_k_update(std::execution::seq, __alpha, __A, __C, __t);
  }
}

template <class _exec_space,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_C,
          class _SizeType_C,
          ::std::size_t _numRows_C,
          ::std::size_t _numCols_C,
          class _Layout_C,
          class _Accessor_C,
          class _Triangle>
void symmetric_matrix_rank_k_update(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C,
  _Triangle __t)
{
  symmetric_matrix_rank_k_update(std::forward<__nvhpc_exec<_exec_space>>(__exec), _ElementType_A(1.), __A, __C, __t);
}

} // namespace __nvhpc_std

#endif
