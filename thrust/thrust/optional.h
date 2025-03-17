///
// optional - An implementation of std::optional with extensions
// Written in 2017 by Sy Brand (@TartanLlama)
//
// To the extent possible under law, the author(s) have dedicated all
// copyright and related and neighboring rights to this software to the
// public domain worldwide. This software is distributed without any warranty.
//
// You should have received a copy of the CC0 Public Domain Dedication
// along with this software. If not, see
// <http://creativecommons.org/publicdomain/zero/1.0/>.
///

#pragma once

#include <thrust/detail/config.h>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header
#include <thrust/detail/type_traits.h>
#include <thrust/swap.h>

#include <cuda/std/__memory/addressof.h>
#include <cuda/std/__type_traits/void_t.h>

#define THRUST_OPTIONAL_VERSION_MAJOR 0
#define THRUST_OPTIONAL_VERSION_MINOR 2

#include <exception>
#include <functional>
#include <new>
#include <type_traits>
#include <utility>

_CCCL_SUPPRESS_DEPRECATED_PUSH

#if _CCCL_COMPILER(MSVC, ==, 19, 00)
#  define THRUST_OPTIONAL_MSVC2015
#endif

THRUST_NAMESPACE_BEGIN

#ifndef THRUST_MONOSTATE_INPLACE_MUTEX
#  define THRUST_MONOSTATE_INPLACE_MUTEX
/// \brief Used to represent an optional with no data; essentially a bool
class CCCL_DEPRECATED_BECAUSE("Use cuda::std::monostate instead") monostate
{};

/// \brief A tag type to tell optional to construct its value in-place
struct CCCL_DEPRECATED in_place_t
{
  explicit in_place_t() = default;
};
/// \brief A tag to tell optional to construct its value in-place
static constexpr in_place_t in_place{};
#endif

template <class T>
class CCCL_DEPRECATED_BECAUSE("Use cuda::std::optional") optional;

/// \exclude
namespace detail
{
#ifndef THRUST_TRAITS_MUTEX
#  define THRUST_TRAITS_MUTEX
// C++14-style aliases for brevity
template <class T>
using remove_const_t = typename std::remove_const<T>::type;
template <class T>
using remove_reference_t = typename std::remove_reference<T>::type;
template <class T>
using decay_t = typename std::decay<T>::type;
template <bool E, class T = void>
using enable_if_t = typename std::enable_if<E, T>::type;
template <bool B, class T, class F>
using conditional_t = typename std::conditional<B, T, F>::type;

// std::conjunction from C++17
template <class...>
struct conjunction : std::true_type
{};
template <class B>
struct conjunction<B> : B
{};
template <class B, class... Bs>
struct conjunction<B, Bs...> : std::conditional<bool(B::value), conjunction<Bs...>, B>::type
{};

// std::invoke from C++17
// https://stackoverflow.com/questions/38288042/c11-14-invoke-workaround
_CCCL_EXEC_CHECK_DISABLE
template <
  typename Fn,
  typename... Args,
#  ifdef THRUST_OPTIONAL_LIBCXX_MEM_FN_WORKAROUND
  typename = enable_if_t<!(is_pointer_to_non_const_member_func<Fn>::value && is_const_or_const_ref<Args...>::value)>,
#  endif
  typename = enable_if_t<std::is_member_pointer<decay_t<Fn>>::value>,
  int      = 0>
_CCCL_HOST_DEVICE constexpr auto
invoke(Fn&& f, Args&&... args) noexcept(noexcept(std::mem_fn(f)(std::forward<Args>(args)...)))
  -> decltype(std::mem_fn(f)(std::forward<Args>(args)...))
{
  return std::mem_fn(f)(std::forward<Args>(args)...);
}

_CCCL_EXEC_CHECK_DISABLE
template <typename Fn, typename... Args, typename = enable_if_t<!std::is_member_pointer<decay_t<Fn>>::value>>
_CCCL_HOST_DEVICE constexpr auto
invoke(Fn&& f, Args&&... args) noexcept(noexcept(std::forward<Fn>(f)(std::forward<Args>(args)...)))
  -> decltype(std::forward<Fn>(f)(std::forward<Args>(args)...))
{
  return std::forward<Fn>(f)(std::forward<Args>(args)...);
}
#endif

// Trait for checking if a type is a thrust::optional
template <class T>
struct is_optional_impl : std::false_type
{};
template <class T>
struct is_optional_impl<optional<T>> : std::true_type
{};
template <class T>
using is_optional = is_optional_impl<decay_t<T>>;

// Change void to thrust::monostate
template <class U>
using fixup_void = conditional_t<std::is_void<U>::value, monostate, U>;

template <class F, class U, class = invoke_result_t<F, U>>
using get_map_return = optional<fixup_void<invoke_result_t<F, U>>>;

// Check if invoking F for some Us returns void
template <class F, class = void, class... U>
struct returns_void_impl;
template <class F, class... U>
struct returns_void_impl<F, ::cuda::std::void_t<invoke_result_t<F, U...>>, U...>
    : std::is_void<invoke_result_t<F, U...>>
{};
template <class F, class... U>
using returns_void = returns_void_impl<F, void, U...>;

template <class T, class... U>
using enable_if_ret_void = enable_if_t<returns_void<T&&, U...>::value>;

template <class T, class... U>
using disable_if_ret_void = enable_if_t<!returns_void<T&&, U...>::value>;

template <class T, class U>
using enable_forward_value =
  detail::enable_if_t<std::is_constructible<T, U&&>::value && !std::is_same<detail::decay_t<U>, in_place_t>::value
                      && !std::is_same<optional<T>, detail::decay_t<U>>::value>;

template <class T, class U, class Other>
using enable_from_other = detail::enable_if_t<
  std::is_constructible<T, Other>::value && !std::is_constructible<T, optional<U>&>::value
  && !std::is_constructible<T, optional<U>&&>::value && !std::is_constructible<T, const optional<U>&>::value
  && !std::is_constructible<T, const optional<U>&&>::value && !std::is_convertible<optional<U>&, T>::value
  && !std::is_convertible<optional<U>&&, T>::value && !std::is_convertible<const optional<U>&, T>::value
  && !std::is_convertible<const optional<U>&&, T>::value>;

template <class T, class U>
using enable_assign_forward =
  detail::enable_if_t<!std::is_same<optional<T>, detail::decay_t<U>>::value
                      && !detail::conjunction<std::is_scalar<T>, std::is_same<T, detail::decay_t<U>>>::value
                      && std::is_constructible<T, U>::value && std::is_assignable<T&, U>::value>;

template <class T, class U, class Other>
using enable_assign_from_other = detail::enable_if_t<
  std::is_constructible<T, Other>::value && std::is_assignable<T&, Other>::value
  && !std::is_constructible<T, optional<U>&>::value && !std::is_constructible<T, optional<U>&&>::value
  && !std::is_constructible<T, const optional<U>&>::value && !std::is_constructible<T, const optional<U>&&>::value
  && !std::is_convertible<optional<U>&, T>::value && !std::is_convertible<optional<U>&&, T>::value
  && !std::is_convertible<const optional<U>&, T>::value && !std::is_convertible<const optional<U>&&, T>::value
  && !std::is_assignable<T&, optional<U>&>::value && !std::is_assignable<T&, optional<U>&&>::value
  && !std::is_assignable<T&, const optional<U>&>::value && !std::is_assignable<T&, const optional<U>&&>::value>;

#if _CCCL_COMPILER(MSVC)
// TODO make a version which works with MSVC
template <class T, class U = T>
struct is_swappable : std::true_type
{};

template <class T, class U = T>
struct is_nothrow_swappable : std::true_type
{};
#else
// https://stackoverflow.com/questions/26744589/what-is-a-proper-way-to-implement-is-swappable-to-test-for-the-swappable-concept
namespace swap_adl_tests
{
// if swap ADL finds this then it would call std::swap otherwise (same
// signature)
struct tag
{};

template <class T>
tag swap(T&, T&);
template <class T, std::size_t N>
tag swap(T (&a)[N], T (&b)[N]);

// helper functions to test if an unqualified swap is possible, and if it
// becomes std::swap
template <class, class>
std::false_type can_swap(...) noexcept(false);
template <class T, class U, class = decltype(swap(std::declval<T&>(), std::declval<U&>()))>
std::true_type can_swap(int) noexcept(noexcept(swap(std::declval<T&>(), std::declval<U&>())));

template <class, class>
std::false_type uses_std(...);
template <class T, class U>
std::is_same<decltype(swap(std::declval<T&>(), std::declval<U&>())), tag> uses_std(int);

template <class T>
struct is_std_swap_noexcept
    : std::integral_constant<bool,
                             std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value>
{};

template <class T, std::size_t N>
struct is_std_swap_noexcept<T[N]> : is_std_swap_noexcept<T>
{};

template <class T, class U>
struct is_adl_swap_noexcept : std::integral_constant<bool, noexcept(can_swap<T, U>(0))>
{};
} // namespace swap_adl_tests

template <class T, class U = T>
struct is_swappable
    : std::integral_constant<bool,
                             decltype(detail::swap_adl_tests::can_swap<T, U>(0))::value
                               && (!decltype(detail::swap_adl_tests::uses_std<T, U>(0))::value
                                   || (std::is_move_assignable<T>::value && std::is_move_constructible<T>::value))>
{};

template <class T, std::size_t N>
struct is_swappable<T[N], T[N]>
    : std::integral_constant<
        bool,
        decltype(detail::swap_adl_tests::can_swap<T[N], T[N]>(0))::value
          && (!decltype(detail::swap_adl_tests::uses_std<T[N], T[N]>(0))::value || is_swappable<T, T>::value)>
{};

template <class T, class U = T>
struct is_nothrow_swappable
    : std::integral_constant<bool,
                             is_swappable<T, U>::value
                               && ((decltype(detail::swap_adl_tests::uses_std<T, U>(0))::value
                                    && detail::swap_adl_tests::is_std_swap_noexcept<T>::value)
                                   || (!decltype(detail::swap_adl_tests::uses_std<T, U>(0))::value
                                       && detail::swap_adl_tests::is_adl_swap_noexcept<T, U>::value))>
{};
#endif

// The storage base manages the actual storage, and correctly propagates
// trivial destruction from T. This case is for when T is not trivially
// destructible.
template <class T, bool = ::std::is_trivially_destructible<T>::value>
struct optional_storage_base
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional_storage_base() noexcept
      : m_dummy()
      , m_has_value(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... U>
  _CCCL_HOST_DEVICE constexpr optional_storage_base(in_place_t, U&&... u)
      : m_value(std::forward<U>(u)...)
      , m_has_value(true)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE ~optional_storage_base()
  {
    if (m_has_value)
    {
      m_value.~T();
      m_has_value = false;
    }
  }

  struct dummy
  {};
  union
  {
    dummy m_dummy;
    T m_value;
  };

  bool m_has_value;
};

// This case is for when T is trivially destructible.
template <class T>
struct optional_storage_base<T, true>
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional_storage_base() noexcept
      : m_dummy()
      , m_has_value(false)
  {}

  _CCCL_EXEC_CHECK_DISABLE
  template <class... U>
  _CCCL_HOST_DEVICE constexpr optional_storage_base(in_place_t, U&&... u)
      : m_value(std::forward<U>(u)...)
      , m_has_value(true)
  {}

  // No destructor, so this class is trivially destructible

  struct dummy
  {};
  union
  {
    dummy m_dummy;
    T m_value;
  };

  bool m_has_value = false;
};

// This base class provides some handy member functions which can be used in
// further derived classes
template <class T>
struct optional_operations_base : optional_storage_base<T>
{
  using optional_storage_base<T>::optional_storage_base;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void hard_reset() noexcept
  {
    get().~T();
    this->m_has_value = false;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class... Args>
  _CCCL_HOST_DEVICE void construct(Args&&... args) noexcept
  {
    new (::cuda::std::addressof(this->m_value)) T(std::forward<Args>(args)...);
    this->m_has_value = true;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <class Opt>
  _CCCL_HOST_DEVICE void assign(Opt&& rhs)
  {
    if (this->has_value())
    {
      if (rhs.has_value())
      {
        this->m_value = std::forward<Opt>(rhs).get();
      }
      else
      {
        this->m_value.~T();
        this->m_has_value = false;
      }
    }

    if (rhs.has_value())
    {
      construct(std::forward<Opt>(rhs).get());
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE bool has_value() const
  {
    return this->m_has_value;
  }

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T& get() &
  {
    return this->m_value;
  }
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T& get() const&
  {
    return this->m_value;
  }
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T&& get() &&
  {
    return std::move(this->m_value);
  }
#ifndef THRUST_OPTIONAL_NO_CONSTRR
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T&& get() const&&
  {
    return std::move(this->m_value);
  }
#endif
};

// This class manages conditionally having a trivial copy constructor
// This specialization is for when T is trivially copy constructible
template <class T, bool = ::cuda::std::is_trivially_copy_constructible<T>::value>
struct optional_copy_base : optional_operations_base<T>
{
  using optional_operations_base<T>::optional_operations_base;
};

// This specialization is for when T is not trivially copy constructible
template <class T>
struct optional_copy_base<T, false> : optional_operations_base<T>
{
  using optional_operations_base<T>::optional_operations_base;

  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional_copy_base(const optional_copy_base& rhs)
  {
    if (rhs.has_value())
    {
      this->construct(rhs.get());
    }
    else
    {
      this->m_has_value = false;
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_base(optional_copy_base&& rhs) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_base& operator=(const optional_copy_base& rhs) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_base& operator=(optional_copy_base&& rhs) = default;
};

template <class T, bool = ::cuda::std::is_trivially_move_constructible<T>::value>
struct optional_move_base : optional_copy_base<T>
{
  using optional_copy_base<T>::optional_copy_base;
};
template <class T>
struct optional_move_base<T, false> : optional_copy_base<T>
{
  using optional_copy_base<T>::optional_copy_base;

  _CCCL_EXEC_CHECK_DISABLE
  optional_move_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_move_base(const optional_move_base& rhs) = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional_move_base(optional_move_base&& rhs) noexcept(std::is_nothrow_move_constructible<T>::value)
  {
    if (rhs.has_value())
    {
      this->construct(std::move(rhs.get()));
    }
    else
    {
      this->m_has_value = false;
    }
  }
  _CCCL_EXEC_CHECK_DISABLE
  optional_move_base& operator=(const optional_move_base& rhs) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_move_base& operator=(optional_move_base&& rhs) = default;
};

// This class manages conditionally having a trivial copy assignment operator
template <class T,
          bool = ::cuda::std::is_trivially_copy_assignable<T>::value
              && ::cuda::std::is_trivially_copy_constructible<T>::value
              && ::cuda::std::is_trivially_destructible<T>::value>
struct optional_copy_assign_base : optional_move_base<T>
{
  using optional_move_base<T>::optional_move_base;
};

template <class T>
struct optional_copy_assign_base<T, false> : optional_move_base<T>
{
  using optional_move_base<T>::optional_move_base;

  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_assign_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_assign_base(const optional_copy_assign_base& rhs) = default;

  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_assign_base(optional_copy_assign_base&& rhs) = default;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional_copy_assign_base& operator=(const optional_copy_assign_base& rhs)
  {
    this->assign(rhs);
    return *this;
  }
  _CCCL_EXEC_CHECK_DISABLE
  optional_copy_assign_base& operator=(optional_copy_assign_base&& rhs) = default;
};

template <class T,
          bool = ::cuda::std::is_trivially_destructible<T>::value
              && ::cuda::std::is_trivially_move_constructible<T>::value
              && ::cuda::std::is_trivially_move_assignable<T>::value>
struct optional_move_assign_base : optional_copy_assign_base<T>
{
  using optional_copy_assign_base<T>::optional_copy_assign_base;
};

template <class T>
struct optional_move_assign_base<T, false> : optional_copy_assign_base<T>
{
  using optional_copy_assign_base<T>::optional_copy_assign_base;

  _CCCL_EXEC_CHECK_DISABLE
  optional_move_assign_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_move_assign_base(const optional_move_assign_base& rhs) = default;

  _CCCL_EXEC_CHECK_DISABLE
  optional_move_assign_base(optional_move_assign_base&& rhs) = default;

  _CCCL_EXEC_CHECK_DISABLE
  optional_move_assign_base& operator=(const optional_move_assign_base& rhs) = default;

  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional_move_assign_base& operator=(optional_move_assign_base&& rhs) noexcept(
    std::is_nothrow_move_constructible<T>::value && std::is_nothrow_move_assignable<T>::value)
  {
    this->assign(std::move(rhs));
    return *this;
  }
};

// optional_delete_ctor_base will conditionally delete copy and move
// constructors depending on whether T is copy/move constructible
template <class T,
          bool EnableCopy = std::is_copy_constructible<T>::value,
          bool EnableMove = std::is_move_constructible<T>::value>
struct optional_delete_ctor_base
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(const optional_delete_ctor_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(optional_delete_ctor_base&&) noexcept = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(const optional_delete_ctor_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(optional_delete_ctor_base&&) noexcept = default;
};

template <class T>
struct optional_delete_ctor_base<T, true, false>
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(const optional_delete_ctor_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(optional_delete_ctor_base&&) noexcept = delete;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(const optional_delete_ctor_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(optional_delete_ctor_base&&) noexcept = default;
};

template <class T>
struct optional_delete_ctor_base<T, false, true>
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(const optional_delete_ctor_base&) = delete;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(optional_delete_ctor_base&&) noexcept = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(const optional_delete_ctor_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(optional_delete_ctor_base&&) noexcept = default;
};

template <class T>
struct optional_delete_ctor_base<T, false, false>
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(const optional_delete_ctor_base&) = delete;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base(optional_delete_ctor_base&&) noexcept = delete;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(const optional_delete_ctor_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_ctor_base& operator=(optional_delete_ctor_base&&) noexcept = default;
};

// optional_delete_assign_base will conditionally delete copy and move
// constructors depending on whether T is copy/move constructible + assignable
template <class T,
          bool EnableCopy = (std::is_copy_constructible<T>::value && std::is_copy_assignable<T>::value),
          bool EnableMove = (std::is_move_constructible<T>::value && std::is_move_assignable<T>::value)>
struct optional_delete_assign_base
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(const optional_delete_assign_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(optional_delete_assign_base&&) noexcept = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(const optional_delete_assign_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(optional_delete_assign_base&&) noexcept = default;
};

template <class T>
struct optional_delete_assign_base<T, true, false>
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(const optional_delete_assign_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(optional_delete_assign_base&&) noexcept = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(const optional_delete_assign_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(optional_delete_assign_base&&) noexcept = delete;
};

template <class T>
struct optional_delete_assign_base<T, false, true>
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(const optional_delete_assign_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(optional_delete_assign_base&&) noexcept = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(const optional_delete_assign_base&) = delete;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(optional_delete_assign_base&&) noexcept = default;
};

template <class T>
struct optional_delete_assign_base<T, false, false>
{
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base() = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(const optional_delete_assign_base&) = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base(optional_delete_assign_base&&) noexcept = default;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(const optional_delete_assign_base&) = delete;
  _CCCL_EXEC_CHECK_DISABLE
  optional_delete_assign_base& operator=(optional_delete_assign_base&&) noexcept = delete;
};

} // namespace detail

/// \brief A tag type to represent an empty optional
struct CCCL_DEPRECATED nullopt_t
{
  struct do_not_use
  {};
  _CCCL_HOST_DEVICE constexpr explicit nullopt_t(do_not_use, do_not_use) noexcept {}
};
/// \brief Represents an empty optional
/// \synopsis static constexpr nullopt_t nullopt;
///
/// *Examples*:
/// ```
/// thrust::optional<int> a = thrust::nullopt;
/// void foo (thrust::optional<int>);
/// foo(thrust::nullopt); //pass an empty optional
/// ```
#ifdef __CUDA_ARCH__
__device__ static constexpr
#else
static constexpr
#endif // __CUDA_ARCH__
  nullopt_t nullopt{nullopt_t::do_not_use{}, nullopt_t::do_not_use{}};

class CCCL_DEPRECATED bad_optional_access : public std::exception
{
public:
  bad_optional_access() = default;
  _CCCL_HOST const char* what() const noexcept
  {
    return "Optional has no value";
  }
};

/// An optional object is an object that contains the storage for another
/// object and manages the lifetime of this contained object, if any. The
/// contained object may be initialized after the optional object has been
/// initialized, and may be destroyed before the optional object has been
/// destroyed. The initialization state of the contained object is tracked by
/// the optional object.
template <class T>
class optional
    : private detail::optional_move_assign_base<T>
    , private detail::optional_delete_ctor_base<T>
    , private detail::optional_delete_assign_base<T>
{
  using base = detail::optional_move_assign_base<T>;

  static_assert(!std::is_same<T, in_place_t>::value, "instantiation of optional with in_place_t is ill-formed");
  static_assert(!std::is_same<detail::decay_t<T>, nullopt_t>::value,
                "instantiation of optional with nullopt_t is ill-formed");

public:
// The different versions for C++14 and 11 are needed because deduced return
// types are not SFINAE-safe. This provides better support for things like
// generic lambdas. C.f.
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0826r0
#if !defined(THRUST_OPTIONAL_GCC49) && !defined(THRUST_OPTIONAL_GCC54) && !defined(THRUST_OPTIONAL_GCC55)
  /// \group and_then
  /// Carries out some operation which returns an optional on the stored
  /// object if there is one. \requires `std::invoke(std::forward<F>(f),
  /// value())` returns a `std::optional<U>` for some `U`. \return Let `U` be
  /// the result of `std::invoke(std::forward<F>(f), value())`. Returns a
  /// `std::optional<U>`. The return value is empty if `*this` is empty,
  /// otherwise the return value of `std::invoke(std::forward<F>(f), value())`
  /// is returned.
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) &
  {
    using result = detail::invoke_result_t<F, T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) &&
  {
    using result = detail::invoke_result_t<F, T&&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) const&
  {
    using result = detail::invoke_result_t<F, const T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

#  ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) const&&
  {
    using result = detail::invoke_result_t<F, const T&&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : result(nullopt);
  }
#  endif
#else
  /// \group and_then
  /// Carries out some operation which returns an optional on the stored
  /// object if there is one. \requires `std::invoke(std::forward<F>(f),
  /// value())` returns a `std::optional<U>` for some `U`.
  /// \return Let `U` be the result of `std::invoke(std::forward<F>(f),
  /// value())`. Returns a `std::optional<U>`. The return value is empty if
  /// `*this` is empty, otherwise the return value of
  /// `std::invoke(std::forward<F>(f), value())` is returned.
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, T&> and_then(F&& f) &
  {
    using result = detail::invoke_result_t<F, T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, T&&> and_then(F&& f) &&
  {
    using result = detail::invoke_result_t<F, T&&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, const T&> and_then(F&& f) const&
  {
    using result = detail::invoke_result_t<F, const T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

#  ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, const T&&> and_then(F&& f) const&&
  {
    using result = detail::invoke_result_t<F, const T&&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : result(nullopt);
  }
#  endif
#endif

#if !defined(THRUST_OPTIONAL_GCC49) && !defined(THRUST_OPTIONAL_GCC54) && !defined(THRUST_OPTIONAL_GCC55)
  /// \brief Carries out some operation on the stored object if there is one.
  /// \return Let `U` be the result of `std::invoke(std::forward<F>(f),
  /// value())`. Returns a `std::optional<U>`. The return value is empty if
  /// `*this` is empty, otherwise an `optional<U>` is constructed from the
  /// return value of `std::invoke(std::forward<F>(f), value())` and is
  /// returned.
  ///
  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) &
  {
    return optional_map_impl(*this, std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) &&
  {
    return optional_map_impl(std::move(*this), std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) const&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) const&
  {
    return optional_map_impl(*this, std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) const&&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) const&&
  {
    return optional_map_impl(std::move(*this), std::forward<F>(f));
  }
#else
  /// \brief Carries out some operation on the stored object if there is one.
  /// \return Let `U` be the result of `std::invoke(std::forward<F>(f),
  /// value())`. Returns a `std::optional<U>`. The return value is empty if
  /// `*this` is empty, otherwise an `optional<U>` is constructed from the
  /// return value of `std::invoke(std::forward<F>(f), value())` and is
  /// returned.
  ///
  /// \group map
  /// \synopsis template <class F> auto map(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(optional_map_impl(std::declval<optional&>(), std::declval<F&&>())) map(F&& f) &
  {
    return optional_map_impl(*this, std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> auto map(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(optional_map_impl(std::declval<optional&&>(), std::declval<F&&>())) map(F&& f) &&
  {
    return optional_map_impl(std::move(*this), std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> auto map(F &&f) const&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(optional_map_impl(std::declval<const optional&>(), std::declval<F&&>()))
  map(F&& f) const&
  {
    return optional_map_impl(*this, std::forward<F>(f));
  }

#  ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group map
  /// \synopsis template <class F> auto map(F &&f) const&&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(optional_map_impl(std::declval<const optional&&>(), std::declval<F&&>()))
  map(F&& f) const&&
  {
    return optional_map_impl(std::move(*this), std::forward<F>(f));
  }
#  endif
#endif

  /// \brief Calls `f` if the optional is empty
  /// \requires `std::invoke_result_t<F>` must be void or convertible to
  /// `optional<T>`.
  /// \effects If `*this` has a value, returns `*this`.
  /// Otherwise, if `f` returns `void`, calls `std::forward<F>(f)` and returns
  /// `std::nullopt`. Otherwise, returns `std::forward<F>(f)()`.
  ///
  /// \group or_else
  /// \synopsis template <class F> optional<T> or_else (F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) &
  {
    if (has_value())
    {
      return *this;
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) &
  {
    return has_value() ? *this : std::forward<F>(f)();
  }

  /// \group or_else
  /// \synopsis template <class F> optional<T> or_else (F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) &&
  {
    if (has_value())
    {
      return std::move(*this);
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) &&
  {
    return has_value() ? std::move(*this) : std::forward<F>(f)();
  }

  /// \group or_else
  /// \synopsis template <class F> optional<T> or_else (F &&f) const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) const&
  {
    if (has_value())
    {
      return *this;
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) const&
  {
    return has_value() ? *this : std::forward<F>(f)();
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) const&&
  {
    if (has_value())
    {
      return std::move(*this);
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) const&&
  {
    return has_value() ? std::move(*this) : std::forward<F>(f)();
  }
#endif

  /// \brief Maps the stored value with `f` if there is one, otherwise returns
  /// `u`.
  ///
  /// \details If there is a value stored, then `f` is called with `**this`
  /// and the value is returned. Otherwise `u` is returned.
  ///
  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) &
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u);
  }

  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) &&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u);
  }

  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) const&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u);
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) const&&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u);
  }
#endif

  /// \brief Maps the stored value with `f` if there is one, otherwise calls
  /// `u` and returns the result.
  ///
  /// \details If there is a value stored, then `f` is
  /// called with `**this` and the value is returned. Otherwise
  /// `std::forward<U>(u)()` is returned.
  ///
  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) &
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u)();
  }

  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u)
  /// &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) &&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u)();
  }

  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u)
  /// const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) const&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u)();
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u)
  /// const &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) const&&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u)();
  }
#endif

  /// \return `u` if `*this` has a value, otherwise an empty optional.
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr optional<typename std::decay<U>::type> conjunction(U&& u) const
  {
    using result = optional<detail::decay_t<U>>;
    return has_value() ? result{u} : result{nullopt};
  }

  /// \return `rhs` if `*this` is empty, otherwise the current value.
  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) &
  {
    return has_value() ? *this : rhs;
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) const&
  {
    return has_value() ? *this : rhs;
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) &&
  {
    return has_value() ? std::move(*this) : rhs;
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) const&&
  {
    return has_value() ? std::move(*this) : rhs;
  }
#endif

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) &
  {
    return has_value() ? *this : std::move(rhs);
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) const&
  {
    return has_value() ? *this : std::move(rhs);
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) &&
  {
    return has_value() ? std::move(*this) : std::move(rhs);
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) const&&
  {
    return has_value() ? std::move(*this) : std::move(rhs);
  }
#endif

  /// Takes the value out of the optional, leaving it empty
  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() &
  {
    optional ret = *this;
    reset();
    return ret;
  }

  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() const&
  {
    optional ret = *this;
    reset();
    return ret;
  }

  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() &&
  {
    optional ret = std::move(*this);
    reset();
    return ret;
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() const&&
  {
    optional ret = std::move(*this);
    reset();
    return ret;
  }
#endif

  using value_type = T;

  /// Constructs an optional that does not contain a value.
  /// \group ctor_empty
  _CCCL_EXEC_CHECK_DISABLE
  constexpr optional() noexcept = default;

  /// \group ctor_empty
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional(nullopt_t) noexcept {}

  /// Copy constructor
  ///
  /// If `rhs` contains a value, the stored value is direct-initialized with
  /// it. Otherwise, the constructed optional is empty.
  _CCCL_EXEC_CHECK_DISABLE
  constexpr optional(const optional& rhs) = default;

  /// Move constructor
  ///
  /// If `rhs` contains a value, the stored value is direct-initialized with
  /// it. Otherwise, the constructed optional is empty.
  _CCCL_EXEC_CHECK_DISABLE
  constexpr optional(optional&& rhs) = default;

  /// Constructs the stored value in-place using the given arguments.
  /// \group in_place
  /// \synopsis template <class... Args> constexpr explicit optional(in_place_t, Args&&... args);
  _CCCL_EXEC_CHECK_DISABLE
  template <class... Args>
  _CCCL_HOST_DEVICE constexpr explicit optional(
    detail::enable_if_t<std::is_constructible<T, Args...>::value, in_place_t>, Args&&... args)
      : base(in_place, std::forward<Args>(args)...)
  {}

  /// \group in_place
  /// \synopsis template <class U, class... Args>\nconstexpr explicit optional(in_place_t, std::initializer_list<U>&,
  /// Args&&... args);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U, class... Args>
  _CCCL_HOST_DEVICE constexpr explicit optional(
    detail::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value, in_place_t>,
    std::initializer_list<U> il,
    Args&&... args)
  {
    this->construct(il, std::forward<Args>(args)...);
  }

  /// Constructs the stored value with `u`.
  /// \synopsis template <class U=T> constexpr optional(U &&u);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U                                                  = T,
            detail::enable_if_t<std::is_convertible<U&&, T>::value>* = nullptr,
            detail::enable_forward_value<T, U>*                      = nullptr>
  _CCCL_HOST_DEVICE constexpr optional(U&& u)
      : base(in_place, std::forward<U>(u))
  {}

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class U                                                   = T,
            detail::enable_if_t<!std::is_convertible<U&&, T>::value>* = nullptr,
            detail::enable_forward_value<T, U>*                       = nullptr>
  _CCCL_HOST_DEVICE constexpr explicit optional(U&& u)
      : base(in_place, std::forward<U>(u))
  {}

  /// Converting copy constructor.
  /// \synopsis template <class U> optional(const optional<U> &rhs);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U,
            detail::enable_from_other<T, U, const U&>*                    = nullptr,
            detail::enable_if_t<std::is_convertible<const U&, T>::value>* = nullptr>
  _CCCL_HOST_DEVICE optional(const optional<U>& rhs)
  {
    this->construct(*rhs);
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class U,
            detail::enable_from_other<T, U, const U&>*                     = nullptr,
            detail::enable_if_t<!std::is_convertible<const U&, T>::value>* = nullptr>
  _CCCL_HOST_DEVICE explicit optional(const optional<U>& rhs)
  {
    this->construct(*rhs);
  }

  /// Converting move constructor.
  /// \synopsis template <class U> optional(optional<U> &&rhs);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U,
            detail::enable_from_other<T, U, U&&>*                    = nullptr,
            detail::enable_if_t<std::is_convertible<U&&, T>::value>* = nullptr>
  _CCCL_HOST_DEVICE optional(optional<U>&& rhs)
  {
    this->construct(std::move(*rhs));
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class U,
            detail::enable_from_other<T, U, U&&>*                     = nullptr,
            detail::enable_if_t<!std::is_convertible<U&&, T>::value>* = nullptr>
  _CCCL_HOST_DEVICE explicit optional(optional<U>&& rhs)
  {
    this->construct(std::move(*rhs));
  }

  /// Destroys the stored value if there is one.
  _CCCL_EXEC_CHECK_DISABLE
  ~optional() = default;

  /// Assignment to empty.
  ///
  /// Destroys the current value if there is one.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional& operator=(nullopt_t) noexcept
  {
    if (has_value())
    {
      this->m_value.~T();
      this->m_has_value = false;
    }

    return *this;
  }

  /// Copy assignment.
  ///
  /// Copies the value from `rhs` if there is one. Otherwise resets the stored
  /// value in `*this`.
  _CCCL_EXEC_CHECK_DISABLE
  optional& operator=(const optional& rhs) = default;

  /// Move assignment.
  ///
  /// Moves the value from `rhs` if there is one. Otherwise resets the stored
  /// value in `*this`.
  _CCCL_EXEC_CHECK_DISABLE
  optional& operator=(optional&& rhs) = default;

  /// Assigns the stored value from `u`, destroying the old value if there was
  /// one.
  /// \synopsis optional &operator=(U &&u);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U = T, detail::enable_assign_forward<T, U>* = nullptr>
  _CCCL_HOST_DEVICE optional& operator=(U&& u)
  {
    if (has_value())
    {
      this->m_value = std::forward<U>(u);
    }
    else
    {
      this->construct(std::forward<U>(u));
    }

    return *this;
  }

  /// Converting copy assignment operator.
  ///
  /// Copies the value from `rhs` if there is one. Otherwise resets the stored
  /// value in `*this`.
  /// \synopsis optional &operator=(const optional<U> & rhs);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U, detail::enable_assign_from_other<T, U, const U&>* = nullptr>
  _CCCL_HOST_DEVICE optional& operator=(const optional<U>& rhs)
  {
    if (has_value())
    {
      if (rhs.has_value())
      {
        this->m_value = *rhs;
      }
      else
      {
        this->hard_reset();
      }
    }

    if (rhs.has_value())
    {
      this->construct(*rhs);
    }

    return *this;
  }

  // TODO check exception guarantee
  /// Converting move assignment operator.
  ///
  /// Moves the value from `rhs` if there is one. Otherwise resets the stored
  /// value in `*this`.
  /// \synopsis optional &operator=(optional<U> && rhs);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U, detail::enable_assign_from_other<T, U, U>* = nullptr>
  _CCCL_HOST_DEVICE optional& operator=(optional<U>&& rhs)
  {
    if (has_value())
    {
      if (rhs.has_value())
      {
        this->m_value = std::move(*rhs);
      }
      else
      {
        this->hard_reset();
      }
    }

    if (rhs.has_value())
    {
      this->construct(std::move(*rhs));
    }

    return *this;
  }

  /// Constructs the value in-place, destroying the current one if there is
  /// one.
  /// \group emplace
  _CCCL_EXEC_CHECK_DISABLE
  template <class... Args>
  _CCCL_HOST_DEVICE T& emplace(Args&&... args)
  {
    static_assert(std::is_constructible<T, Args&&...>::value, "T must be constructible with Args");

    *this = nullopt;
    this->construct(std::forward<Args>(args)...);
    return this->m_value;
  }

  /// \group emplace
  /// \synopsis template <class U, class... Args>\nT& emplace(std::initializer_list<U> il, Args &&... args);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U, class... Args>
  _CCCL_HOST_DEVICE detail::enable_if_t<std::is_constructible<T, std::initializer_list<U>&, Args&&...>::value, T&>
  emplace(std::initializer_list<U> il, Args&&... args)
  {
    *this = nullopt;
    this->construct(il, std::forward<Args>(args)...);
    return this->m_value;
  }

  /// Swaps this optional with the other.
  ///
  /// If neither optionals have a value, nothing happens.
  /// If both have a value, the values are swapped.
  /// If one has a value, it is moved to the other and the movee is left
  /// valueless.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void
  swap(optional& rhs) noexcept(std::is_nothrow_move_constructible<T>::value && detail::is_nothrow_swappable<T>::value)
  {
    if (has_value())
    {
      if (rhs.has_value())
      {
        using ::cuda::std::swap;
        swap(**this, *rhs);
      }
      else
      {
        new (::cuda::std::addressof(rhs.m_value)) T(std::move(this->m_value));
        this->m_value.T::~T();
      }
    }
    else if (rhs.has_value())
    {
      new (::cuda::std::addressof(this->m_value)) T(std::move(rhs.m_value));
      rhs.m_value.T::~T();
    }
  }

  /// \return a pointer to the stored value
  /// \requires a value is stored
  /// \group pointer
  /// \synopsis constexpr const T *operator->() const;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T* operator->() const
  {
    return ::cuda::std::addressof(this->m_value);
  }

  /// \group pointer
  /// \synopsis constexpr T *operator->();
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T* operator->()
  {
    return ::cuda::std::addressof(this->m_value);
  }

  /// \return the stored value
  /// \requires a value is stored
  /// \group deref
  /// \synopsis constexpr T &operator*();
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T& operator*() &
  {
    return this->m_value;
  }

  /// \group deref
  /// \synopsis constexpr const T &operator*() const;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T& operator*() const&
  {
    return this->m_value;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T&& operator*() &&
  {
    return std::move(this->m_value);
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T&& operator*() const&&
  {
    return std::move(this->m_value);
  }
#endif

  /// \return whether or not the optional has a value
  /// \group has_value
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr bool has_value() const noexcept
  {
    return this->m_has_value;
  }

  /// \group has_value
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr explicit operator bool() const noexcept
  {
    return this->m_has_value;
  }

  /// \return the contained value if there is one, otherwise throws
  /// [bad_optional_access]
  /// \group value
  /// \synopsis constexpr T &value();
  _CCCL_HOST constexpr T& value() &
  {
    if (has_value())
    {
      return this->m_value;
    }
    throw bad_optional_access();
  }
  /// \group value
  /// \synopsis constexpr const T &value() const;
  _CCCL_HOST constexpr const T& value() const&
  {
    if (has_value())
    {
      return this->m_value;
    }
    throw bad_optional_access();
  }
  /// \exclude
  _CCCL_HOST constexpr T&& value() &&
  {
    if (has_value())
    {
      return std::move(this->m_value);
    }
    throw bad_optional_access();
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \exclude
  _CCCL_HOST constexpr const T&& value() const&&
  {
    if (has_value())
    {
      return std::move(this->m_value);
    }
    throw bad_optional_access();
  }
#endif

  /// \return the stored value if there is one, otherwise returns `u`
  /// \group value_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr T value_or(U&& u) const&
  {
    static_assert(std::is_copy_constructible<T>::value && std::is_convertible<U&&, T>::value,
                  "T must be copy constructible and convertible from U");
    return has_value() ? **this : static_cast<T>(std::forward<U>(u));
  }

  /// \group value_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr T value_or(U&& u) &&
  {
    static_assert(std::is_move_constructible<T>::value && std::is_convertible<U&&, T>::value,
                  "T must be move constructible and convertible from U");
    return has_value() ? **this : static_cast<T>(std::forward<U>(u));
  }

  /// Destroys the stored value if one exists, making the optional empty
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void reset() noexcept
  {
    if (has_value())
    {
      this->m_value.~T();
      this->m_has_value = false;
    }
  }
};

/// \group relop
/// \brief Compares two optional objects
/// \details If both optionals contain a value, they are compared with `T`s
/// relational operators. Otherwise `lhs` and `rhs` are equal only if they are
/// both empty, and `lhs` is less than `rhs` only if `rhs` is empty and `lhs`
/// is not.
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator==(const optional<T>& lhs, const optional<U>& rhs)
{
  return lhs.has_value() == rhs.has_value() && (!lhs.has_value() || *lhs == *rhs);
}
/// \group relop
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator!=(const optional<T>& lhs, const optional<U>& rhs)
{
  return lhs.has_value() != rhs.has_value() || (lhs.has_value() && *lhs != *rhs);
}
/// \group relop
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator<(const optional<T>& lhs, const optional<U>& rhs)
{
  return rhs.has_value() && (!lhs.has_value() || *lhs < *rhs);
}
/// \group relop
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator>(const optional<T>& lhs, const optional<U>& rhs)
{
  return lhs.has_value() && (!rhs.has_value() || *lhs > *rhs);
}
/// \group relop
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator<=(const optional<T>& lhs, const optional<U>& rhs)
{
  return !lhs.has_value() || (rhs.has_value() && *lhs <= *rhs);
}
/// \group relop
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator>=(const optional<T>& lhs, const optional<U>& rhs)
{
  return !rhs.has_value() || (lhs.has_value() && *lhs >= *rhs);
}

/// \group relop_nullopt
/// \brief Compares an optional to a `nullopt`
/// \details Equivalent to comparing the optional to an empty optional
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator==(const optional<T>& lhs, nullopt_t) noexcept
{
  return !lhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator==(nullopt_t, const optional<T>& rhs) noexcept
{
  return !rhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator!=(const optional<T>& lhs, nullopt_t) noexcept
{
  return lhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator!=(nullopt_t, const optional<T>& rhs) noexcept
{
  return rhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator<(const optional<T>&, nullopt_t) noexcept
{
  return false;
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator<(nullopt_t, const optional<T>& rhs) noexcept
{
  return rhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator<=(const optional<T>& lhs, nullopt_t) noexcept
{
  return !lhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator<=(nullopt_t, const optional<T>&) noexcept
{
  return true;
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator>(const optional<T>& lhs, nullopt_t) noexcept
{
  return lhs.has_value();
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator>(nullopt_t, const optional<T>&) noexcept
{
  return false;
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator>=(const optional<T>&, nullopt_t) noexcept
{
  return true;
}
/// \group relop_nullopt
_CCCL_EXEC_CHECK_DISABLE
template <class T>
_CCCL_HOST_DEVICE inline constexpr bool operator>=(nullopt_t, const optional<T>& rhs) noexcept
{
  return !rhs.has_value();
}

/// \group relop_t
/// \brief Compares the optional with a value.
/// \details If the optional has a value, it is compared with the other value
/// using `T`s relational operators. Otherwise, the optional is considered
/// less than the value.
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator==(const optional<T>& lhs, const U& rhs)
{
  return lhs.has_value() ? *lhs == rhs : false;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator==(const U& lhs, const optional<T>& rhs)
{
  return rhs.has_value() ? lhs == *rhs : false;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator!=(const optional<T>& lhs, const U& rhs)
{
  return lhs.has_value() ? *lhs != rhs : true;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator!=(const U& lhs, const optional<T>& rhs)
{
  return rhs.has_value() ? lhs != *rhs : true;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator<(const optional<T>& lhs, const U& rhs)
{
  return lhs.has_value() ? *lhs < rhs : true;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator<(const U& lhs, const optional<T>& rhs)
{
  return rhs.has_value() ? lhs < *rhs : false;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator<=(const optional<T>& lhs, const U& rhs)
{
  return lhs.has_value() ? *lhs <= rhs : true;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator<=(const U& lhs, const optional<T>& rhs)
{
  return rhs.has_value() ? lhs <= *rhs : false;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator>(const optional<T>& lhs, const U& rhs)
{
  return lhs.has_value() ? *lhs > rhs : false;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator>(const U& lhs, const optional<T>& rhs)
{
  return rhs.has_value() ? lhs > *rhs : true;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator>=(const optional<T>& lhs, const U& rhs)
{
  return lhs.has_value() ? *lhs >= rhs : false;
}
/// \group relop_t
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U>
_CCCL_HOST_DEVICE inline constexpr bool operator>=(const U& lhs, const optional<T>& rhs)
{
  return rhs.has_value() ? lhs >= *rhs : true;
}

/// \synopsis template <class T>\nvoid swap(optional<T> &lhs, optional<T> &rhs);
_CCCL_EXEC_CHECK_DISABLE
template <class T,
          detail::enable_if_t<std::is_move_constructible<T>::value>* = nullptr,
          detail::enable_if_t<detail::is_swappable<T>::value>*       = nullptr>
_CCCL_HOST_DEVICE void swap(optional<T>& lhs, optional<T>& rhs) noexcept(noexcept(lhs.swap(rhs)))
{
  return lhs.swap(rhs);
}

namespace detail
{
struct i_am_secret
{};
} // namespace detail

_CCCL_EXEC_CHECK_DISABLE
template <class T = detail::i_am_secret,
          class U,
          class Ret = detail::conditional_t<std::is_same<T, detail::i_am_secret>::value, detail::decay_t<U>, T>>
CCCL_DEPRECATED_BECAUSE("Use cuda::std::make_optional")
_CCCL_HOST_DEVICE inline constexpr optional<Ret> make_optional(U&& v)
{
  return optional<Ret>(std::forward<U>(v));
}

_CCCL_EXEC_CHECK_DISABLE
template <class T, class... Args>
CCCL_DEPRECATED_BECAUSE("Use cuda::std::make_optional")
_CCCL_HOST_DEVICE inline constexpr optional<T> make_optional(Args&&... args)
{
  return optional<T>(in_place, std::forward<Args>(args)...);
}
_CCCL_EXEC_CHECK_DISABLE
template <class T, class U, class... Args>
CCCL_DEPRECATED_BECAUSE("Use cuda::std::make_optional")
_CCCL_HOST_DEVICE inline constexpr optional<T> make_optional(std::initializer_list<U> il, Args&&... args)
{
  return optional<T>(in_place, il, std::forward<Args>(args)...);
}

template <class T>
optional(T) -> optional<T>;

// Doxygen chokes on the trailing return types used below.
#if !defined(_CCCL_DOXYGEN_INVOKED)
/// \exclude
namespace detail
{
_CCCL_EXEC_CHECK_DISABLE
template <class Opt,
          class F,
          class Ret = decltype(detail::invoke(std::declval<F>(), *std::declval<Opt>())),
          detail::enable_if_t<!std::is_void<Ret>::value>* = nullptr>
_CCCL_HOST_DEVICE constexpr auto optional_map_impl(Opt&& opt, F&& f)
{
  return opt.has_value() ? detail::invoke(std::forward<F>(f), *std::forward<Opt>(opt)) : optional<Ret>(nullopt);
}

_CCCL_EXEC_CHECK_DISABLE
template <class Opt,
          class F,
          class Ret = decltype(detail::invoke(std::declval<F>(), *std::declval<Opt>())),
          detail::enable_if_t<std::is_void<Ret>::value>* = nullptr>
_CCCL_HOST_DEVICE auto optional_map_impl(Opt&& opt, F&& f)
{
  if (opt.has_value())
  {
    detail::invoke(std::forward<F>(f), *std::forward<Opt>(opt));
#  if _CCCL_COMPILER(MSVC)
    // MSVC fails to suppress the warning on make_optional
    _CCCL_SUPPRESS_DEPRECATED_PUSH
    return optional<monostate>(monostate{});
    _CCCL_SUPPRESS_DEPRECATED_POP
#  elif _CCCL_COMPILER(NVHPC)
    // NVHPC cannot have a diagnostic pop after a return statement
    _CCCL_SUPPRESS_DEPRECATED_PUSH
    auto o = optional<monostate>(monostate{});
    _CCCL_SUPPRESS_DEPRECATED_POP
    return ::cuda::std::move(o);
#  else
    _CCCL_SUPPRESS_DEPRECATED_PUSH
    return make_optional(monostate{});
    _CCCL_SUPPRESS_DEPRECATED_POP
#  endif
  }

  return optional<monostate>(nullopt);
}

} // namespace detail
#endif // !defined(_CCCL_DOXYGEN_INVOKED)

/// Specialization for when `T` is a reference. `optional<T&>` acts similarly
/// to a `T*`, but provides more operations and shows intent more clearly.
///
/// *Examples*:
///
/// ```
/// int i = 42;
/// thrust::optional<int&> o = i;
/// *o == 42; //true
/// i = 12;
/// *o = 12; //true
/// &*o == &i; //true
/// ```
///
/// Assignment has rebind semantics rather than assign-through semantics:
///
/// ```
/// int j = 8;
/// o = j;
///
/// &*o == &j; //true
/// ```
template <class T>
class optional<T&>
{
public:
// The different versions for C++14 and 11 are needed because deduced return
// types are not SFINAE-safe. This provides better support for things like
// generic lambdas. C.f.
// http://www.open-std.org/jtc1/sc22/wg21/docs/papers/2017/p0826r0
#if !defined(THRUST_OPTIONAL_GCC49) && !defined(THRUST_OPTIONAL_GCC54) && !defined(THRUST_OPTIONAL_GCC55)
  /// \group and_then
  /// Carries out some operation which returns an optional on the stored
  /// object if there is one. \requires `std::invoke(std::forward<F>(f),
  /// value())` returns a `std::optional<U>` for some `U`. \return Let `U` be
  /// the result of `std::invoke(std::forward<F>(f), value())`. Returns a
  /// `std::optional<U>`. The return value is empty if `*this` is empty,
  /// otherwise the return value of `std::invoke(std::forward<F>(f), value())`
  /// is returned.
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) &
  {
    using result = detail::invoke_result_t<F, T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) &&
  {
    using result = detail::invoke_result_t<F, T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) const&
  {
    using result = detail::invoke_result_t<F, const T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

#  ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto and_then(F&& f) const&&
  {
    using result = detail::invoke_result_t<F, const T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }
#  endif
#else
  /// \group and_then
  /// Carries out some operation which returns an optional on the stored
  /// object if there is one. \requires `std::invoke(std::forward<F>(f),
  /// value())` returns a `std::optional<U>` for some `U`. \return Let `U` be
  /// the result of `std::invoke(std::forward<F>(f), value())`. Returns a
  /// `std::optional<U>`. The return value is empty if `*this` is empty,
  /// otherwise the return value of `std::invoke(std::forward<F>(f), value())`
  /// is returned.
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, T&> and_then(F&& f) &
  {
    using result = detail::invoke_result_t<F, T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, T&> and_then(F&& f) &&
  {
    using result = detail::invoke_result_t<F, T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, const T&> and_then(F&& f) const&
  {
    using result = detail::invoke_result_t<F, const T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }

#  ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group and_then
  /// \synopsis template <class F>\nconstexpr auto and_then(F &&f) const &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr detail::invoke_result_t<F, const T&> and_then(F&& f) const&&
  {
    using result = detail::invoke_result_t<F, const T&>;
    static_assert(detail::is_optional<result>::value, "F must return an optional");

    return has_value() ? detail::invoke(std::forward<F>(f), **this) : result(nullopt);
  }
#  endif
#endif

#if !defined(THRUST_OPTIONAL_GCC49) && !defined(THRUST_OPTIONAL_GCC54) && !defined(THRUST_OPTIONAL_GCC55)
  /// \brief Carries out some operation on the stored object if there is one.
  /// \return Let `U` be the result of `std::invoke(std::forward<F>(f),
  /// value())`. Returns a `std::optional<U>`. The return value is empty if
  /// `*this` is empty, otherwise an `optional<U>` is constructed from the
  /// return value of `std::invoke(std::forward<F>(f), value())` and is
  /// returned.
  ///
  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) &
  {
    return detail::optional_map_impl(*this, std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) &&
  {
    return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) const&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) const&
  {
    return detail::optional_map_impl(*this, std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> constexpr auto map(F &&f) const&&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr auto map(F&& f) const&&
  {
    return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
  }
#else
  /// \brief Carries out some operation on the stored object if there is one.
  /// \return Let `U` be the result of `std::invoke(std::forward<F>(f),
  /// value())`. Returns a `std::optional<U>`. The return value is empty if
  /// `*this` is empty, otherwise an `optional<U>` is constructed from the
  /// return value of `std::invoke(std::forward<F>(f), value())` and is
  /// returned.
  ///
  /// \group map
  /// \synopsis template <class F> auto map(F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(detail::optional_map_impl(std::declval<optional&>(), std::declval<F&&>()))
  map(F&& f) &
  {
    return detail::optional_map_impl(*this, std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> auto map(F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(detail::optional_map_impl(std::declval<optional&&>(), std::declval<F&&>()))
  map(F&& f) &&
  {
    return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
  }

  /// \group map
  /// \synopsis template <class F> auto map(F &&f) const&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(detail::optional_map_impl(std::declval<const optional&>(), std::declval<F&&>()))
  map(F&& f) const&
  {
    return detail::optional_map_impl(*this, std::forward<F>(f));
  }

#  ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group map
  /// \synopsis template <class F> auto map(F &&f) const&&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F>
  _CCCL_HOST_DEVICE constexpr decltype(detail::optional_map_impl(std::declval<const optional&&>(), std::declval<F&&>()))
  map(F&& f) const&&
  {
    return detail::optional_map_impl(std::move(*this), std::forward<F>(f));
  }
#  endif
#endif

  /// \brief Calls `f` if the optional is empty
  /// \requires `std::invoke_result_t<F>` must be void or convertible to
  /// `optional<T>`. \effects If `*this` has a value, returns `*this`.
  /// Otherwise, if `f` returns `void`, calls `std::forward<F>(f)` and returns
  /// `std::nullopt`. Otherwise, returns `std::forward<F>(f)()`.
  ///
  /// \group or_else
  /// \synopsis template <class F> optional<T> or_else (F &&f) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) &
  {
    if (has_value())
    {
      return *this;
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) &
  {
    return has_value() ? *this : std::forward<F>(f)();
  }

  /// \group or_else
  /// \synopsis template <class F> optional<T> or_else (F &&f) &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) &&
  {
    if (has_value())
    {
      return std::move(*this);
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) &&
  {
    return has_value() ? std::move(*this) : std::forward<F>(f)();
  }

  /// \group or_else
  /// \synopsis template <class F> optional<T> or_else (F &&f) const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) const&
  {
    if (has_value())
    {
      return *this;
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> constexpr or_else(F&& f) const&
  {
    return has_value() ? *this : std::forward<F>(f)();
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::enable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) const&&
  {
    if (has_value())
    {
      return std::move(*this);
    }

    std::forward<F>(f)();
    return nullopt;
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, detail::disable_if_ret_void<F>* = nullptr>
  _CCCL_HOST_DEVICE optional<T> or_else(F&& f) const&&
  {
    return has_value() ? std::move(*this) : std::forward<F>(f)();
  }
#endif

  /// \brief Maps the stored value with `f` if there is one, otherwise returns
  /// `u`.
  ///
  /// \details If there is a value stored, then `f` is called with `**this`
  /// and the value is returned. Otherwise `u` is returned.
  ///
  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) &
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u);
  }

  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) &&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u);
  }

  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) const&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u);
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group map_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE U map_or(F&& f, U&& u) const&&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u);
  }
#endif

  /// \brief Maps the stored value with `f` if there is one, otherwise calls
  /// `u` and returns the result.
  ///
  /// \details If there is a value stored, then `f` is
  /// called with `**this` and the value is returned. Otherwise
  /// `std::forward<U>(u)()` is returned.
  ///
  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u) &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) &
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u)();
  }

  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u)
  /// &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) &&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u)();
  }

  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u)
  /// const &;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) const&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), **this) : std::forward<U>(u)();
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group map_or_else
  /// \synopsis template <class F, class U>\nauto map_or_else(F &&f, U &&u)
  /// const &&;
  _CCCL_EXEC_CHECK_DISABLE
  template <class F, class U>
  _CCCL_HOST_DEVICE detail::invoke_result_t<U> map_or_else(F&& f, U&& u) const&&
  {
    return has_value() ? detail::invoke(std::forward<F>(f), std::move(**this)) : std::forward<U>(u)();
  }
#endif

  /// \return `u` if `*this` has a value, otherwise an empty optional.
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr optional<typename std::decay<U>::type> conjunction(U&& u) const
  {
    using result = optional<detail::decay_t<U>>;
    return has_value() ? result{u} : result{nullopt};
  }

  /// \return `rhs` if `*this` is empty, otherwise the current value.
  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) &
  {
    return has_value() ? *this : rhs;
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) const&
  {
    return has_value() ? *this : rhs;
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) &&
  {
    return has_value() ? std::move(*this) : rhs;
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(const optional& rhs) const&&
  {
    return has_value() ? std::move(*this) : rhs;
  }
#endif

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) &
  {
    return has_value() ? *this : std::move(rhs);
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) const&
  {
    return has_value() ? *this : std::move(rhs);
  }

  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) &&
  {
    return has_value() ? std::move(*this) : std::move(rhs);
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group disjunction
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional disjunction(optional&& rhs) const&&
  {
    return has_value() ? std::move(*this) : std::move(rhs);
  }
#endif

  /// Takes the value out of the optional, leaving it empty
  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() &
  {
    optional ret = *this;
    reset();
    return ret;
  }

  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() const&
  {
    optional ret = *this;
    reset();
    return ret;
  }

  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() &&
  {
    optional ret = std::move(*this);
    reset();
    return ret;
  }

#ifndef THRUST_OPTIONAL_NO_CONSTRR
  /// \group take
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional take() const&&
  {
    optional ret = std::move(*this);
    reset();
    return ret;
  }
#endif

  using value_type = T&;

  /// Constructs an optional that does not contain a value.
  /// \group ctor_empty
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional() noexcept
      : m_value(nullptr)
  {}

  /// \group ctor_empty
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr optional(nullopt_t) noexcept
      : m_value(nullptr)
  {}

  /// Copy constructor
  ///
  /// If `rhs` contains a value, the stored value is direct-initialized with
  /// it. Otherwise, the constructed optional is empty.
  _CCCL_EXEC_CHECK_DISABLE
  constexpr optional(const optional& rhs) noexcept = default;

  /// Move constructor
  ///
  /// If `rhs` contains a value, the stored value is direct-initialized with
  /// it. Otherwise, the constructed optional is empty.
  _CCCL_EXEC_CHECK_DISABLE
  constexpr optional(optional&& rhs) = default;

  /// Constructs the stored value with `u`.
  /// \synopsis template <class U=T> constexpr optional(U &&u);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U = T, detail::enable_if_t<!detail::is_optional<detail::decay_t<U>>::value>* = nullptr>
  _CCCL_HOST_DEVICE constexpr optional(U&& u)
      : m_value(::cuda::std::addressof(u))
  {
    static_assert(std::is_lvalue_reference<U>::value, "U must be an lvalue");
  }

  /// \exclude
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr explicit optional(const optional<U>& rhs)
      : optional(*rhs)
  {}

  /// No-op
  _CCCL_EXEC_CHECK_DISABLE
  ~optional() = default;

  /// Assignment to empty.
  ///
  /// Destroys the current value if there is one.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE optional& operator=(nullopt_t) noexcept
  {
    m_value = nullptr;
    return *this;
  }

  /// Copy assignment.
  ///
  /// Rebinds this optional to the referee of `rhs` if there is one. Otherwise
  /// resets the stored value in `*this`.
  _CCCL_EXEC_CHECK_DISABLE
  optional& operator=(const optional& rhs) = default;

  /// Rebinds this optional to `u`.
  ///
  /// \requires `U` must be an lvalue reference.
  /// \synopsis optional &operator=(U &&u);
  _CCCL_EXEC_CHECK_DISABLE
  template <class U = T, detail::enable_if_t<!detail::is_optional<detail::decay_t<U>>::value>* = nullptr>
  _CCCL_HOST_DEVICE optional& operator=(U&& u)
  {
    static_assert(std::is_lvalue_reference<U>::value, "U must be an lvalue");
    m_value = ::cuda::std::addressof(u);
    return *this;
  }

  /// Converting copy assignment operator.
  ///
  /// Rebinds this optional to the referee of `rhs` if there is one. Otherwise
  /// resets the stored value in `*this`.
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE optional& operator=(const optional<U>& rhs)
  {
    m_value = ::cuda::std::addressof(rhs.value());
    return *this;
  }

  /// Constructs the value in-place, destroying the current one if there is
  /// one.
  ///
  /// \group emplace
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE T& emplace(U& u) noexcept
  {
    m_value = ::cuda::std::addressof(u);
    return *m_value;
  }

  /// Swaps this optional with the other.
  ///
  /// If neither optionals have a value, nothing happens.
  /// If both have a value, the values are swapped.
  /// If one has a value, it is moved to the other and the movee is left
  /// valueless.
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE void swap(optional& rhs) noexcept
  {
    std::swap(m_value, rhs.m_value);
  }

  /// \return a pointer to the stored value
  /// \requires a value is stored
  /// \group pointer
  /// \synopsis constexpr const T *operator->() const;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T* operator->() const
  {
    return m_value;
  }

  /// \group pointer
  /// \synopsis constexpr T *operator->();
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr T* operator->()
  {
    return m_value;
  }

  /// \return the stored value
  /// \requires a value is stored
  /// \group deref
  /// \synopsis constexpr T &operator*();
  _CCCL_EXEC_CHECK_DISABLE
  constexpr T& operator*()
  {
    return *m_value;
  }

  /// \group deref
  /// \synopsis constexpr const T &operator*() const;
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr const T& operator*() const
  {
    return *m_value;
  }

  /// \return whether or not the optional has a value
  /// \group has_value
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr bool has_value() const noexcept
  {
    return m_value != nullptr;
  }

  /// \group has_value
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE constexpr explicit operator bool() const noexcept
  {
    return m_value != nullptr;
  }

  /// \return the contained value if there is one, otherwise throws
  /// [bad_optional_access]
  /// \group value
  /// synopsis constexpr T &value();
  _CCCL_HOST constexpr T& value()
  {
    if (has_value())
    {
      return *m_value;
    }
    throw bad_optional_access();
  }
  /// \group value
  /// \synopsis constexpr const T &value() const;
  _CCCL_HOST constexpr const T& value() const
  {
    if (has_value())
    {
      return *m_value;
    }
    throw bad_optional_access();
  }

  /// \return the stored value if there is one, otherwise returns `u`
  /// \group value_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr T value_or(U&& u) const&
  {
    static_assert(std::is_copy_constructible<T>::value && std::is_convertible<U&&, T>::value,
                  "T must be copy constructible and convertible from U");
    return has_value() ? **this : static_cast<T>(std::forward<U>(u));
  }

  /// \group value_or
  _CCCL_EXEC_CHECK_DISABLE
  template <class U>
  _CCCL_HOST_DEVICE constexpr T value_or(U&& u) &&
  {
    static_assert(std::is_move_constructible<T>::value && std::is_convertible<U&&, T>::value,
                  "T must be move constructible and convertible from U");
    return has_value() ? **this : static_cast<T>(std::forward<U>(u));
  }

  /// Destroys the stored value if one exists, making the optional empty
  _CCCL_EXEC_CHECK_DISABLE
  void reset() noexcept
  {
    m_value = nullptr;
  }

private:
  T* m_value;
};

THRUST_NAMESPACE_END

namespace std
{
// TODO SFINAE
template <class T>
struct hash<THRUST_NS_QUALIFIER::optional<T>>
{
  _CCCL_EXEC_CHECK_DISABLE
  _CCCL_HOST_DEVICE ::std::size_t operator()(const THRUST_NS_QUALIFIER::optional<T>& o) const
  {
    if (!o.has_value())
    {
      return 0;
    }

    return std::hash<::cuda::std::remove_const_t<T>>()(*o);
  }
};
} // namespace std

_CCCL_SUPPRESS_DEPRECATED_POP
