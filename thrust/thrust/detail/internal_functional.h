/*
 *  Copyright 2008-2018 NVIDIA Corporation
 *
 *  Licensed under the Apache License, Version 2.0 (the "License");
 *  you may not use this file except in compliance with the License.
 *  You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 *  Unless required by applicable law or agreed to in writing, software
 *  distributed under the License is distributed on an "AS IS" BASIS,
 *  WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 *  See the License for the specific language governing permissions and
 *  limitations under the License.
 */

/*! \file internal_functional.inl
 *  \brief Non-public functionals used to implement algorithm internals.
 */

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <thrust/detail/raw_reference_cast.h>
#include <thrust/detail/static_assert.h>
#include <thrust/detail/type_traits.h>
#include <thrust/iterator/detail/tuple_of_iterator_references.h>
#include <thrust/iterator/iterator_traits.h>
#include <thrust/tuple.h>

#include <cuda/__functional/address_stability.h>
#include <cuda/__iterator/discard_iterator.h>
#include <cuda/__iterator/tabulate_output_iterator.h>
#include <cuda/__iterator/transform_input_output_iterator.h>
#include <cuda/__iterator/transform_output_iterator.h>
#include <cuda/std/__cccl/memory_wrapper.h> // for ::new
#include <cuda/std/__new/device_new.h>
#include <cuda/std/__tuple_dir/get.h>
#include <cuda/std/__tuple_dir/tuple_element.h>
#include <cuda/std/__type_traits/enable_if.h>
#include <cuda/std/__type_traits/is_const.h>
#include <cuda/std/__type_traits/is_convertible.h>
#include <cuda/std/__type_traits/is_reference.h>
#include <cuda/std/__type_traits/type_identity.h>

THRUST_NAMESPACE_BEGIN

namespace detail
{
// convert a predicate to a 0 or 1 integral value
template <typename Predicate, typename IntegralType>
struct predicate_to_integral
{
  Predicate pred;

  template <typename T>
  _CCCL_HOST_DEVICE IntegralType operator()(const T& x)
  {
    return pred(x) ? IntegralType(1) : IntegralType(0);
  }
};

// note that equal_to_value does not force conversion from T2 -> T1 as equal_to does
template <typename T2>
struct equal_to_value
{
  T2 rhs;

  // need this ctor for nvcc 12.0 + clang14 to make copy ctor of not_fn_t<equal_to_value> work. Check test:
  // thrust.cpp.cuda.cpp20.test.remove.
  _CCCL_HOST_DEVICE equal_to_value(const T2& rhs)
      : rhs(rhs)
  {}

  template <typename T1>
  _CCCL_HOST_DEVICE bool operator()(const T1& lhs) const
  {
    return lhs == rhs;
  }
};

template <typename Predicate>
struct tuple_binary_predicate
{
  template <typename Tuple>
  _CCCL_HOST_DEVICE bool operator()(const Tuple& t) const
  {
    return pred(::cuda::std::get<0>(t), ::cuda::std::get<1>(t));
  }

  mutable Predicate pred;
};

template <class Predicate, class NewType, class OutputType>
struct new_value_if_f
{
  Predicate pred;
  NewType new_value;

  template <class T>
  _CCCL_DEVICE_API OutputType operator()(T const& x)
  {
    return pred(x) ? new_value : x;
  }

  template <class T, class P>
  _CCCL_DEVICE_API OutputType operator()(T const& x, P const& y)
  {
    return pred(y) ? new_value : x;
  }
};

// We need to mark proxy iterators as such
template <>
inline constexpr bool is_proxy_reference_v<::cuda::discard_iterator::__discard_proxy> = true;

template <class Fn, class Index>
inline constexpr bool is_proxy_reference_v<::cuda::__tabulate_proxy<Fn, Index>> = true;

template <class Iter, class Fn>
inline constexpr bool is_proxy_reference_v<::cuda::__transform_output_proxy<Iter, Fn>> = true;

template <class Iter, class InputFn, class OutputFn>
inline constexpr bool is_proxy_reference_v<::cuda::__transform_input_output_proxy<Iter, InputFn, OutputFn>> = true;

template <typename T>
inline constexpr bool is_non_const_reference_v =
  !::cuda::std::is_const_v<T> && (::cuda::std::is_reference_v<T> || detail::is_proxy_reference_v<T>);

template <typename T>
inline constexpr bool is_tuple_of_iterator_references_v = false;

template <typename... Ts>
inline constexpr bool is_tuple_of_iterator_references_v<tuple_of_iterator_references<Ts...>> = true;

// use this enable_if to avoid assigning to temporaries in the transform functors below
// XXX revisit this problem with c++11 perfect forwarding
template <typename T>
using enable_if_assignable_ref =
  ::cuda::std::enable_if_t<is_non_const_reference_v<T> || is_tuple_of_iterator_references_v<T>, int>;

template <typename UnaryFunction>
struct unary_transform_functor
{
  UnaryFunction f;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Tuple, enable_if_assignable_ref<::cuda::std::tuple_element_t<1, Tuple>> = 0>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    ::cuda::std::get<1>(t) = f(::cuda::std::get<0>(t));
  }
};

template <typename BinaryFunction>
struct binary_transform_functor
{
  BinaryFunction f;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Tuple, enable_if_assignable_ref<::cuda::std::tuple_element_t<2, Tuple>> = 0>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    ::cuda::std::get<2>(t) = f(::cuda::std::get<0>(t), ::cuda::std::get<1>(t));
  }
};

template <typename UnaryFunction, typename Predicate>
struct unary_transform_if_functor
{
  UnaryFunction unary_op;
  Predicate pred;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Tuple, enable_if_assignable_ref<::cuda::std::tuple_element_t<1, Tuple>> = 0>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    if (pred(::cuda::std::get<0>(t)))
    {
      ::cuda::std::get<1>(t) = unary_op(::cuda::std::get<0>(t));
    }
  }
}; // end unary_transform_if_functor

template <typename UnaryFunction, typename Predicate>
struct unary_transform_if_with_stencil_functor
{
  UnaryFunction unary_op;
  Predicate pred;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Tuple, enable_if_assignable_ref<::cuda::std::tuple_element_t<2, Tuple>> = 0>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    if (pred(::cuda::std::get<1>(t)))
    {
      ::cuda::std::get<2>(t) = unary_op(::cuda::std::get<0>(t));
    }
  }
}; // end unary_transform_if_with_stencil_functor

template <typename BinaryFunction, typename Predicate>
struct binary_transform_if_functor
{
  BinaryFunction binary_op;
  Predicate pred;

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Tuple, enable_if_assignable_ref<::cuda::std::tuple_element_t<3, Tuple>> = 0>
  _CCCL_HOST_DEVICE void operator()(Tuple t)
  {
    if (pred(::cuda::std::get<2>(t)))
    {
      ::cuda::std::get<3>(t) = binary_op(::cuda::std::get<0>(t), ::cuda::std::get<1>(t));
    }
  }
}; // end binary_transform_if_functor

template <typename T>
struct host_destroy_functor
{
  _CCCL_HOST void operator()(T& x) const
  {
    x.~T();
  } // end operator()()
}; // end host_destroy_functor

template <typename T>
struct device_destroy_functor
{
  // add __host__ to allow the omp backend to compile with nvcc
  _CCCL_HOST_DEVICE void operator()(T& x) const
  {
    x.~T();
  } // end operator()()
}; // end device_destroy_functor

template <typename System, typename T>
struct destroy_functor
    : thrust::detail::eval_if<::cuda::std::is_convertible_v<System, thrust::host_system_tag>,
                              ::cuda::std::type_identity<host_destroy_functor<T>>,
                              ::cuda::std::type_identity<device_destroy_functor<T>>>
{};

template <typename T>
struct fill_functor
{
  T exemplar;

  // explicit declaration is needed to avoid an exec check warning
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE fill_functor(const T& _exemplar)
      : exemplar(_exemplar)
  {}

  // explicit declaration is needed to avoid an exec check warning
  _CCCL_EXEC_CHECK_DISABLE
  fill_functor(const fill_functor& other) = default;

  // explicit declaration is needed to avoid an exec check warning
  _CCCL_EXEC_CHECK_DISABLE
  ~fill_functor() = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE T operator()() const
  {
    return exemplar;
  }
};

template <typename T>
struct uninitialized_fill_functor
{
  T exemplar;

  // explicit declaration is needed to avoid an exec check warning
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE uninitialized_fill_functor(const T& x)
      : exemplar(x)
  {}

  // explicit declaration is needed to avoid an exec check warning
  _CCCL_EXEC_CHECK_DISABLE
  uninitialized_fill_functor(const uninitialized_fill_functor& other) = default;

  // explicit declaration is needed to avoid an exec check warning
  _CCCL_EXEC_CHECK_DISABLE
  ~uninitialized_fill_functor() = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void operator()(T& x)
  {
    ::new (static_cast<void*>(&x)) T(exemplar);
  } // end operator()()
}; // end uninitialized_fill_functor

template <typename Compare>
struct compare_first
{
  Compare comp;

  template <typename Tuple1, typename Tuple2>
  _CCCL_HOST_DEVICE bool operator()(const Tuple1& x, const Tuple2& y)
  {
    return comp(thrust::raw_reference_cast(::cuda::std::get<0>(x)), thrust::raw_reference_cast(::cuda::std::get<0>(y)));
  }
}; // end compare_first
} // end namespace detail
THRUST_NAMESPACE_END

_CCCL_BEGIN_NAMESPACE_CUDA
template <typename Predicate, typename IntegralType>
struct proclaims_copyable_arguments<THRUST_NS_QUALIFIER::detail::predicate_to_integral<Predicate, IntegralType>>
    : proclaims_copyable_arguments<Predicate>
{};

template <typename Predicate>
struct proclaims_copyable_arguments<THRUST_NS_QUALIFIER::detail::tuple_binary_predicate<Predicate>>
    : proclaims_copyable_arguments<Predicate>
{};

template <class Predicate, class NewType, class OutputType>
struct proclaims_copyable_arguments<THRUST_NS_QUALIFIER::detail::new_value_if_f<Predicate, NewType, OutputType>>
    : proclaims_copyable_arguments<Predicate>
{};
_CCCL_END_NAMESPACE_CUDA
