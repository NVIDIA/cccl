#include <thrust/detail/config.h>

#include <thrust/detail/allocator/allocator_system.h>
#include <thrust/for_each.h>

#include <cuda/std/__host_stdlib/memory>
#include <cuda/std/__memory/allocator_traits.h>
#include <cuda/std/__memory/pointer_traits.h>
#include <cuda/std/tuple>

THRUST_NAMESPACE_BEGIN
namespace detail
{
// emplace_construct has 2 cases:
// if Allocator has an effectful member function construct(T*, Args...):
//   1. construct via the allocator
// else
//   2. construct via placement new
// Both are dispatched with for_each_n on the allocator's system.

template <typename Allocator, typename T, typename... Args>
inline constexpr bool has_effectful_member_construct = ::cuda::std::__has_construct<Allocator, T*, Args...>;

// std::allocator::construct's only effect is to invoke placement new
template <typename U, typename T, typename... Args>
inline constexpr bool has_effectful_member_construct<std::allocator<U>, T, Args...> = false;

template <typename Allocator, typename... Args>
struct emplace_via_allocator_construct
{
  Allocator& a;
  ::cuda::std::tuple<Args...> args;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& loc)
  {
    ::cuda::std::apply(
      [&](auto&... xs) {
        ::cuda::std::allocator_traits<Allocator>::construct(a, &loc, xs...);
      },
      args);
  }
};

template <typename... Args>
struct emplace_via_placement_new
{
  ::cuda::std::tuple<Args...> args;

  template <typename T>
  _CCCL_HOST_DEVICE void operator()(T& loc)
  {
    ::cuda::std::apply(
      [&](auto&... xs) {
        ::new (static_cast<void*>(&loc)) T(xs...);
      },
      args);
  }
};

// Build one object at loc from args, on the system (CPU or GPU) the allocator belongs to
template <typename Allocator, typename Pointer, typename... Args>
_CCCL_HOST_DEVICE void emplace_construct(Allocator& a, Pointer loc, Args... args)
{
  using T = typename ::cuda::std::pointer_traits<Pointer>::element_type;
  if constexpr (has_effectful_member_construct<Allocator, T, Args...>)
  {
    thrust::for_each_n(
      allocator_system<Allocator>::get(a), loc, 1, emplace_via_allocator_construct<Allocator, Args...>{a, {args...}});
  }
  else
  {
    thrust::for_each_n(allocator_system<Allocator>::get(a), loc, 1, emplace_via_placement_new<Args...>{{args...}});
  }
}
} // namespace detail
THRUST_NAMESPACE_END
