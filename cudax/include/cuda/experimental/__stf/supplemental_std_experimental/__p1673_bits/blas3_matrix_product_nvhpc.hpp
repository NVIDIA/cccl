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

#ifndef INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_NVHPC_HPP_
#define INCLUDE_EXPERIMENTAL___P1673_BITS_BLAS3_MATRIX_PRODUCT_NVHPC_HPP_

namespace __nvhpc_std
{

namespace __ex = std::experimental;
namespace __la = std::experimental::linalg;

template <class _exec_space,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_B,
          class _SizeType_B,
          ::std::size_t _numRows_B,
          ::std::size_t _numCols_B,
          class _Layout_B,
          class _Accessor_B,
          class _ElementType_C,
          class _SizeType_C,
          ::std::size_t _numRows_C,
          ::std::size_t _numCols_C,
          class _Layout_C,
          class _Accessor_C>
void __matrix_product_nvhpc(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
  __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C,
  _ElementType_C __beta)
{
  using __A_t =
    typename __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A>;
  using __B_t =
    typename __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B>;
  using __valA_t = typename __A_t::value_type;
  using __valB_t = typename __B_t::value_type;
  using __ptrA_t = typename std::unique_ptr<__valA_t, std::function<void(__valA_t*)>>;
  using __ptrB_t = typename std::unique_ptr<__valB_t, std::function<void(__valB_t*)>>;

  bool __is_operate_on_transposed = (!__ex::linalg::__is_column_major(__C));
  bool __conj_A, __conj_B;
  char __op_A, __op_B;
  std::tuple<__ptrA_t, __A_t> __work_A{__ptrA_t(), __A};
  std::tuple<__ptrB_t, __B_t> __work_B{__ptrB_t(), __B};

  __extract_ops(__A, __is_operate_on_transposed, &__op_A, &__conj_A);
  __extract_ops(__B, __is_operate_on_transposed, &__op_B, &__conj_B);

  if constexpr (__la::is_complex<__valA_t>::value)
  {
    if (__conj_A)
    {
      __work_A = __create_conjugate(__exec, __A);
    }
  }
  if constexpr (__la::is_complex<__valB_t>::value)
  {
    if (__conj_B)
    {
      __work_B = __create_conjugate(__exec, __B);
    }
  }

  _ElementType_C __alpha = __extract_scaling_factor(__A) * __extract_scaling_factor(__B);

  if (__is_operate_on_transposed) // C is in row-major layout, compute C^t = B^t * A^t
  {
    __matrix_product_impl(
      std::forward<__nvhpc_exec<_exec_space>>(__exec),
      (__conj_B ? std::get<1>(__work_B) : __B),
      __op_B,
      (__conj_A ? std::get<1>(__work_A) : __A),
      __op_A,
      __C,
      __alpha,
      __beta,
      __is_operate_on_transposed);
  }
  else // C is in column-major layout, compute C = A * B
  {
    __matrix_product_impl(
      std::forward<__nvhpc_exec<_exec_space>>(__exec),
      (__conj_A ? std::get<1>(__work_A) : __A),
      __op_A,
      (__conj_B ? std::get<1>(__work_B) : __B),
      __op_B,
      __C,
      __alpha,
      __beta,
      __is_operate_on_transposed);
  }
}

template <
  class _exec_space,
  class _ElementType_A,
  class _SizeType_A,
  ::std::size_t _numRows_A,
  ::std::size_t _numCols_A,
  class _Layout_A,
  class _Accessor_A,
  class _ElementType_B,
  class _SizeType_B,
  ::std::size_t _numRows_B,
  ::std::size_t _numCols_B,
  class _Layout_B,
  class _Accessor_B,
  class _ElementType_E,
  class _SizeType_E,
  ::std::size_t _numRows_E,
  ::std::size_t _numCols_E,
  class _Layout_E,
  class _Accessor_E,
  class _ElementType_C,
  class _SizeType_C,
  ::std::size_t _numRows_C,
  ::std::size_t _numCols_C,
  class _Layout_C,
  class _Accessor_C>
void matrix_product(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
  __ex::mdspan<_ElementType_E, __ex::extents<_SizeType_E, _numRows_E, _numCols_E>, _Layout_E, _Accessor_E> __E,
  __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C)
{
  using __base_exec = decltype(__base_exec_mapper<__nvhpc_exec<_exec_space>>{}.__map());

  constexpr bool __types_supported =
    (__stdblas_data_type<_ElementType_E> == __stdblas_data_type<_ElementType_C>
     && __gemm_data_types_supported<__base_exec>(
       __stdblas_data_type<_ElementType_A>,
       __stdblas_data_type<_ElementType_B>,
       __stdblas_output_data_type<_ElementType_C>));
  constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
  constexpr bool __B_supported = __input_supported<_Layout_B, _Accessor_B>();
  constexpr bool __E_supported = __input_supported<_Layout_E, _Accessor_E>();
  constexpr bool __C_supported = __output_supported<_Layout_C, _Accessor_C>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
  __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
  __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
  __STDBLAS_STATIC_ASSERT_INPUT(__B_supported, __B);
  __STDBLAS_STATIC_ASSERT_INPUT(__E_supported, __E);
  __STDBLAS_STATIC_ASSERT_OUTPUT(__C_supported, __C);
#endif

  if constexpr (__types_supported && __A_supported && __B_supported && __E_supported && __C_supported)
  {
    copy(std::forward<__nvhpc_exec<_exec_space>>(__exec), __E, __C);

    __matrix_product_nvhpc(__nvhpc_exec<_exec_space>(), __A, __B, __C, _ElementType_C(1.0));
  }
  else
  {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(matrix_product);
#endif
    __la::matrix_product(std::execution::seq, __A, __B, __E, __C);
  }
}

template <class _exec_space,
          class _ElementType_A,
          class _SizeType_A,
          ::std::size_t _numRows_A,
          ::std::size_t _numCols_A,
          class _Layout_A,
          class _Accessor_A,
          class _ElementType_B,
          class _SizeType_B,
          ::std::size_t _numRows_B,
          ::std::size_t _numCols_B,
          class _Layout_B,
          class _Accessor_B,
          class _ElementType_C,
          class _SizeType_C,
          ::std::size_t _numRows_C,
          ::std::size_t _numCols_C,
          class _Layout_C,
          class _Accessor_C>
void matrix_product(
  __nvhpc_exec<_exec_space>&& __exec,
  __ex::mdspan<_ElementType_A, __ex::extents<_SizeType_A, _numRows_A, _numCols_A>, _Layout_A, _Accessor_A> __A,
  __ex::mdspan<_ElementType_B, __ex::extents<_SizeType_B, _numRows_B, _numCols_B>, _Layout_B, _Accessor_B> __B,
  __ex::mdspan<_ElementType_C, __ex::extents<_SizeType_C, _numRows_C, _numCols_C>, _Layout_C, _Accessor_C> __C)
{
  using __base_exec = decltype(__base_exec_mapper<__nvhpc_exec<_exec_space>>{}.__map());

  constexpr bool __types_supported = __gemm_data_types_supported<__base_exec>(
    __stdblas_data_type<_ElementType_A>,
    __stdblas_data_type<_ElementType_B>,
    __stdblas_output_data_type<_ElementType_C>);
  constexpr bool __A_supported = __input_supported<_Layout_A, _Accessor_A>();
  constexpr bool __B_supported = __input_supported<_Layout_B, _Accessor_B>();
  constexpr bool __C_supported = __output_supported<_Layout_C, _Accessor_C>();

#ifndef STDBLAS_FALLBACK_UNSUPPORTED_CASES
  __STDBLAS_STATIC_ASSERT_TYPES(__types_supported);
  __STDBLAS_STATIC_ASSERT_INPUT(__A_supported, __A);
  __STDBLAS_STATIC_ASSERT_INPUT(__B_supported, __B);
  __STDBLAS_STATIC_ASSERT_OUTPUT(__C_supported, __C);
#endif

  if constexpr (__types_supported && __A_supported && __B_supported && __C_supported)
  {
    __matrix_product_nvhpc(std::forward<__nvhpc_exec<_exec_space>>(__exec), __A, __B, __C, _ElementType_C(0.0));
  }
  else
  {
#ifdef STDBLAS_VERBOSE
    __STDBLAS_COMPILE_TIME_FALLBACK_MESSAGE(matrix_product);
#endif
    __la::matrix_product(std::execution::seq, __A, __B, __C);
  }
}

} // namespace __nvhpc_std

#endif
