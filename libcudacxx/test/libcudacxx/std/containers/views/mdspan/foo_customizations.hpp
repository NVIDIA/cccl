#ifndef _FOO_CUSTOMIZATIONS_HPP
#define _FOO_CUSTOMIZATIONS_HPP

// Taken from the reference implementation repo

namespace Foo
{
template <class T>
struct foo_ptr
{
  T* data;
  _CCCL_HOST_DEVICE constexpr foo_ptr(T* ptr)
      : data(ptr)
  {}
};

template <class T>
struct foo_accessor
{
  using offset_policy    = foo_accessor;
  using element_type     = T;
  using reference        = T&;
  using data_handle_type = foo_ptr<T>;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr foo_accessor(int* ptr = nullptr) noexcept
  {
    flag = ptr;
  }

  template <class OtherElementType>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr foo_accessor(cuda::std::default_accessor<OtherElementType>) noexcept
  {
    flag = nullptr;
  }

  template <class OtherElementType>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr foo_accessor(foo_accessor<OtherElementType> other) noexcept
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

  _CCCL_HOST_DEVICE friend constexpr void swap(foo_accessor& x, foo_accessor& y)
  {
    x.flag[0] = 99;
    y.flag[0] = 77;
    cuda::std::swap(x.flag, y.flag);
  }
};

struct layout_foo
{
  template <class Extents>
  class mapping;
};

template <class Extents>
class layout_foo::mapping
{
public:
  using extents_type = Extents;
  using index_type   = typename extents_type::index_type;
  using size_type    = typename extents_type::size_type;
  using rank_type    = typename extents_type::rank_type;
  using layout_type  = layout_foo;

private:
  static_assert(cuda::std::__detail::__is_extents_v<extents_type>,
                "layout_foo::mapping must be instantiated with a specialization of cuda::std::extents.");
  static_assert(extents_type::rank() < 3, "layout_foo only supports 0D, 1D and 2D");

  template <class>
  friend class mapping;

public:
  //--------------------------------------------------------------------------------

  _CCCL_HIDE_FROM_ABI constexpr mapping() noexcept               = default;
  _CCCL_HIDE_FROM_ABI constexpr mapping(mapping const&) noexcept = default;

  _CCCL_HOST_DEVICE constexpr mapping(extents_type const& __exts) noexcept
      : __extents(__exts)
  {}

  _LIBCUDACXX_TEMPLATE(class OtherExtents)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(cuda::std::is_constructible, extents_type, OtherExtents))
  __MDSPAN_CONDITIONAL_EXPLICIT((!cuda::std::is_convertible<OtherExtents, extents_type>::value)) // needs two () due to
                                                                                                 // comma
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
    mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      : __extents(other.extents())
  {
    /*
     * TODO: check precondition
     * other.required_span_size() is a representable value of type index_type
     */
  }

  _LIBCUDACXX_TEMPLATE(class OtherExtents)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(cuda::std::is_constructible, extents_type, OtherExtents))
  __MDSPAN_CONDITIONAL_EXPLICIT((!cuda::std::is_convertible<OtherExtents, extents_type>::value)) // needs two () due
                                                                                                 // to comma
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
    cuda::std::layout_right::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      : __extents(other.extents())
  {}

  _LIBCUDACXX_TEMPLATE(class OtherExtents)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(cuda::std::is_constructible, extents_type, OtherExtents)
                       && (extents_type::rank() <= 1))
  __MDSPAN_CONDITIONAL_EXPLICIT((!cuda::std::is_convertible<OtherExtents, extents_type>::value)) // needs two () due to
                                                                                                 // comma
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
    cuda::std::layout_left::mapping<OtherExtents> const& other) noexcept // NOLINT(google-explicit-constructor)
      : __extents(other.extents())
  {}

  _LIBCUDACXX_TEMPLATE(class OtherExtents)
  _LIBCUDACXX_REQUIRES(_CCCL_TRAIT(cuda::std::is_constructible, extents_type, OtherExtents))
  __MDSPAN_CONDITIONAL_EXPLICIT((extents_type::rank() > 0))
  _LIBCUDACXX_HIDE_FROM_ABI constexpr mapping(
    cuda::std::layout_stride::mapping<OtherExtents> const& other) // NOLINT(google-explicit-constructor)
      : __extents(other.extents())
  {
    /*
     * TODO: check precondition
     * other.required_span_size() is a representable value of type index_type
     */
    NV_IF_TARGET(NV_IS_HOST, (size_t stride = 1; for (rank_type r = __extents.rank(); r > 0; r--) {
                   assert(stride == other.stride(r - 1));
                   // if(stride != other.stride(r-1))
                   //   throw std::runtime_error("Assigning layout_stride to layout_foo with invalid strides.");
                   stride *= __extents.extent(r - 1);
                 }))
  }

  _CCCL_HIDE_FROM_ABI __MDSPAN_CONSTEXPR_14_DEFAULTED mapping& operator=(mapping const&) noexcept = default;

  _LIBCUDACXX_HIDE_FROM_ABI constexpr const extents_type& extents() const noexcept
  {
    return __extents;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type required_span_size() const noexcept
  {
    index_type value = 1;
    for (rank_type r = 0; r != extents_type::rank(); ++r)
    {
      value *= __extents.extent(r);
    }
    return value;
  }

  //--------------------------------------------------------------------------------

  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type operator()() const noexcept
  {
    return index_type(0);
  }

  template <class Indx0>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type operator()(Indx0 idx0) const noexcept
  {
    return static_cast<index_type>(idx0);
  }

  template <class Indx0, class Indx1>
  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type operator()(Indx0 idx0, Indx1 idx1) const noexcept
  {
    return static_cast<index_type>(idx0 * __extents.extent(0) + idx1);
  }

  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_unique() noexcept
  {
    return true;
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_exhaustive() noexcept
  {
    return true;
  }
  _LIBCUDACXX_HIDE_FROM_ABI static constexpr bool is_always_strided() noexcept
  {
    return true;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_unique() const noexcept
  {
    return true;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_exhaustive() const noexcept
  {
    return true;
  }
  _LIBCUDACXX_HIDE_FROM_ABI constexpr bool is_strided() const noexcept
  {
    return true;
  }

  _LIBCUDACXX_HIDE_FROM_ABI constexpr index_type stride(rank_type i) const noexcept
  {
    index_type value = 1;
    for (rank_type r = extents_type::rank() - 1; r > i; r--)
    {
      value *= __extents.extent(r);
    }
    return value;
  }

  template <class OtherExtents>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator==(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept
  {
    return lhs.extents() == rhs.extents();
  }

  // In C++ 20 the not equal exists if equal is found
#if !(__MDSPAN_HAS_CXX_20)
  template <class OtherExtents>
  _LIBCUDACXX_HIDE_FROM_ABI friend constexpr bool
  operator!=(mapping const& lhs, mapping<OtherExtents> const& rhs) noexcept
  {
    return lhs.extents() != rhs.extents();
  }
#endif

  // Not really public, but currently needed to implement fully constexpr useable submdspan:
  template <size_t N, class SizeType, size_t... E, size_t... Idx>
  _CCCL_HOST_DEVICE constexpr index_type
  __get_stride(cuda::std::extents<SizeType, E...>, cuda::std::integer_sequence<size_t, Idx...>) const
  {
    return __MDSPAN_FOLD_TIMES_RIGHT((Idx > N ? __extents.template __extent<Idx>() : 1), 1);
  }
  template <size_t N>
  _CCCL_HOST_DEVICE constexpr index_type __stride() const noexcept
  {
    return __get_stride<N>(__extents, std::make_index_sequence<extents_type::rank()>());
  }

private:
  _CCCL_NO_UNIQUE_ADDRESS extents_type __extents{};
};

} // namespace Foo
#endif
