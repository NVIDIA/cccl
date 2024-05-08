/*
 *  Copyright 2008-2022 NVIDIA Corporation
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

/*! \file type_traits.h
 *  \brief Temporarily define some type traits
 *         until nvcc can compile tr1::type_traits.
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

#include <cuda/std/type_traits>

THRUST_NAMESPACE_BEGIN

// forward declaration of device_reference
template <typename T>
class device_reference;

namespace detail
{
/// helper classes [4.3].
template <typename T, T v>
using integral_constant = ::cuda::std::integral_constant<T, v>;
using true_type         = ::cuda::std::true_type;
using false_type        = ::cuda::std::false_type;

template <typename T>
struct is_device_ptr : public false_type
{};

template <typename T>
struct is_non_bool_integral : public ::cuda::std::is_integral<T>
{};
template <>
struct is_non_bool_integral<bool> : public false_type
{};

template <typename T>
struct is_non_bool_arithmetic : public ::cuda::std::is_arithmetic<T>
{};
template <>
struct is_non_bool_arithmetic<bool> : public false_type
{};

template <typename T>
struct is_proxy_reference : public false_type
{};

template <typename T>
struct is_device_reference : public false_type
{};
template <typename T>
struct is_device_reference<thrust::device_reference<T>> : public true_type
{};

template <typename T1, typename T2>
struct is_one_convertible_to_the_other
    : public integral_constant<bool,
                               ::cuda::std::is_convertible<T1, T2>::value || ::cuda::std::is_convertible<T2, T1>::value>
{};

// mpl stuff
template <typename... Conditions>
struct or_;

template <>
struct or_<>
    : public integral_constant<bool,
                               false_type::value // identity for or_
                               >
{}; // end or_

template <typename Condition, typename... Conditions>
struct or_<Condition, Conditions...> : public integral_constant<bool, Condition::value || or_<Conditions...>::value>
{}; // end or_

template <typename... Conditions>
struct and_;

template <>
struct and_<>
    : public integral_constant<bool,
                               true_type::value // identity for and_
                               >
{}; // end and_

template <typename Condition, typename... Conditions>
struct and_<Condition, Conditions...> : public integral_constant<bool, Condition::value && and_<Conditions...>::value>
{}; // end and_

template <typename Boolean>
struct not_ : public integral_constant<bool, !Boolean::value>
{}; // end not_

template <bool B, class T, class F>
struct conditional
{
  typedef T type;
};

template <class T, class F>
struct conditional<false, T, F>
{
  typedef F type;
};

template <bool, typename Then, typename Else>
struct eval_if
{}; // end eval_if

template <typename Then, typename Else>
struct eval_if<true, Then, Else>
{
  typedef typename Then::type type;
}; // end eval_if

template <typename Then, typename Else>
struct eval_if<false, Then, Else>
{
  typedef typename Else::type type;
}; // end eval_if

template <typename T>
//  struct identity
//  XXX WAR nvcc's confusion with thrust::identity
struct identity_
{
  typedef T type;
}; // end identity

template <bool, typename T = void>
struct enable_if
{};
template <typename T>
struct enable_if<true, T>
{
  typedef T type;
};

template <bool, typename T>
struct lazy_enable_if
{};
template <typename T>
struct lazy_enable_if<true, T>
{
  typedef typename T::type type;
};

template <bool condition, typename T = void>
struct disable_if : enable_if<!condition, T>
{};
template <bool condition, typename T>
struct lazy_disable_if : lazy_enable_if<!condition, T>
{};

template <typename T1, typename T2, typename T = void>
struct enable_if_convertible : enable_if<::cuda::std::is_convertible<T1, T2>::value, T>
{};

template <typename T1, typename T2, typename T = void>
using enable_if_convertible_t = typename enable_if_convertible<T1, T2, T>::type;

template <typename T1, typename T2, typename T = void>
struct disable_if_convertible : disable_if<::cuda::std::is_convertible<T1, T2>::value, T>
{};

template <typename T1, typename T2, typename Result = void>
struct enable_if_different : enable_if<!::cuda::std::is_same<T1, T2>::value, Result>
{};

template <typename T>
struct is_numeric : and_<::cuda::std::is_convertible<int, T>, ::cuda::std::is_convertible<T, int>>
{}; // end is_numeric

// make_unsigned follows

namespace tt_detail
{

template <typename T>
struct make_unsigned_simple;

template <>
struct make_unsigned_simple<char>
{
  typedef unsigned char type;
};
template <>
struct make_unsigned_simple<signed char>
{
  typedef unsigned char type;
};
template <>
struct make_unsigned_simple<unsigned char>
{
  typedef unsigned char type;
};
template <>
struct make_unsigned_simple<short>
{
  typedef unsigned short type;
};
template <>
struct make_unsigned_simple<unsigned short>
{
  typedef unsigned short type;
};
template <>
struct make_unsigned_simple<int>
{
  typedef unsigned int type;
};
template <>
struct make_unsigned_simple<unsigned int>
{
  typedef unsigned int type;
};
template <>
struct make_unsigned_simple<long int>
{
  typedef unsigned long int type;
};
template <>
struct make_unsigned_simple<unsigned long int>
{
  typedef unsigned long int type;
};
template <>
struct make_unsigned_simple<long long int>
{
  typedef unsigned long long int type;
};
template <>
struct make_unsigned_simple<unsigned long long int>
{
  typedef unsigned long long int type;
};

template <typename T>
struct make_unsigned_base
{
  // remove cv
  using remove_cv_t = ::cuda::std::__remove_cv_t<T>;

  // get the simple unsigned type
  typedef typename make_unsigned_simple<remove_cv_t>::type unsigned_remove_cv_t;

  // add back const, volatile, both, or neither to the simple result
  typedef typename eval_if<
    ::cuda::std::is_const<T>::value&& ::cuda::std::is_volatile<T>::value,
    // add cv back
    ::cuda::std::add_cv<unsigned_remove_cv_t>,
    // check const & volatile individually
    eval_if<::cuda::std::is_const<T>::value,
            // add c back
            ::cuda::std::add_const<unsigned_remove_cv_t>,
            eval_if<::cuda::std::is_volatile<T>::value,
                    // add v back
                    ::cuda::std::add_volatile<unsigned_remove_cv_t>,
                    // original type was neither cv, return the simple unsigned result
                    identity_<unsigned_remove_cv_t>>>>::type type;
};

} // namespace tt_detail

template <typename T>
struct make_unsigned : tt_detail::make_unsigned_base<T>
{};

struct largest_available_float
{
  typedef double type;
};

// T1 wins if they are both the same size
template <typename T1, typename T2>
struct larger_type
    : thrust::detail::eval_if<(sizeof(T2) > sizeof(T1)), thrust::detail::identity_<T2>, thrust::detail::identity_<T1>>
{};

template <typename Base, typename Derived, typename Result = void>
struct enable_if_base_of : enable_if<::cuda::std::is_base_of<Base, Derived>::value, Result>
{};

template <typename T1, typename T2, typename Enable = void>
struct promoted_numerical_type;

template <typename T1, typename T2>
struct promoted_numerical_type<T1,
                               T2,
                               typename enable_if<and_<typename ::cuda::std::is_floating_point<T1>::type,
                                                       typename ::cuda::std::is_floating_point<T2>::type>::value>::type>
{
  typedef typename larger_type<T1, T2>::type type;
};

template <typename T1, typename T2>
struct promoted_numerical_type<T1,
                               T2,
                               typename enable_if<and_<typename ::cuda::std::is_integral<T1>::type,
                                                       typename ::cuda::std::is_floating_point<T2>::type>::value>::type>
{
  typedef T2 type;
};

template <typename T1, typename T2>
struct promoted_numerical_type<T1,
                               T2,
                               typename enable_if<and_<typename ::cuda::std::is_floating_point<T1>::type,
                                                       typename ::cuda::std::is_integral<T2>::type>::value>::type>
{
  typedef T1 type;
};

template <class F, class... Us>
using invoke_result = ::cuda::std::__invoke_of<F, Us...>;

template <class F, class... Us>
using invoke_result_t = typename invoke_result<F, Us...>::type;
} // namespace detail

using detail::false_type;
using detail::integral_constant;
using detail::true_type;

THRUST_NAMESPACE_END
