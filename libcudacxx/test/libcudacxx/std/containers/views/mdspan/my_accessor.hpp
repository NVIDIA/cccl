#ifndef _MY_ACCESSOR_HPP
#define _MY_ACCESSOR_HPP

#include "foo_customizations.hpp"

namespace Foo
{
// Same as Foo::foo_accessor but
// 1. Doesn't have a default constructor
// 2. Isn't contructible from the default accessor
template <class T>
struct my_accessor
{
  using offset_policy    = my_accessor;
  using element_type     = T;
  using reference        = T&;
  using data_handle_type = foo_ptr<T>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr my_accessor(int* ptr) noexcept
  {
    flag = ptr;
  }

  template <class OtherElementType>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr my_accessor(my_accessor<OtherElementType> other) noexcept
  {
    flag = other.flag;
  }

  _CCCL_HOST_DEVICE constexpr reference access(data_handle_type p, size_t i) const noexcept
  {
    return p.data[i];
  }

  _CCCL_HOST_DEVICE constexpr data_handle_type offset(data_handle_type p, size_t i) const noexcept
  {
    return data_handle_type(p.data + i);
  }
  int* flag;
};
} // namespace Foo

#endif
