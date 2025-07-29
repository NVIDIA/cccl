#pragma once

#include <thrust/execution_policy.h>

#include <iosfwd>

template <typename T, unsigned int N>
struct FixedVector
{
  T data[N];

  _CCCL_HOST_DEVICE FixedVector()
  {
    for (unsigned int i = 0; i < N; i++)
    {
      data[i] = T();
    }
  }

  _CCCL_HOST_DEVICE explicit FixedVector(T init)
  {
    for (unsigned int i = 0; i < N; i++)
    {
      data[i] = init;
    }
  }

  _CCCL_HOST_DEVICE FixedVector operator+(const FixedVector& bs) const
  {
    FixedVector output;
    for (unsigned int i = 0; i < N; i++)
    {
      output.data[i] = data[i] + bs.data[i];
    }
    return output;
  }

  _CCCL_HOST_DEVICE bool operator<(const FixedVector& bs) const
  {
    for (unsigned int i = 0; i < N; i++)
    {
      if (data[i] < bs.data[i])
      {
        return true;
      }
      if (bs.data[i] < data[i])
      {
        return false;
      }
    }
    return false;
  }

  _CCCL_HOST_DEVICE bool operator==(const FixedVector& bs) const
  {
    for (unsigned int i = 0; i < N; i++)
    {
      if (!(data[i] == bs.data[i]))
      {
        return false;
      }
    }
    return true;
  }
};

template <typename Key, typename Value>
struct key_value
{
  using key_type   = Key;
  using value_type = Value;

  _CCCL_HOST_DEVICE key_value()
      : key()
      , value()
  {}

  _CCCL_HOST_DEVICE key_value(key_type k, value_type v)
      : key(k)
      , value(v)
  {}

  _CCCL_HOST_DEVICE bool operator<(const key_value& rhs) const
  {
    return key < rhs.key;
  }

  _CCCL_HOST_DEVICE bool operator>(const key_value& rhs) const
  {
    return key > rhs.key;
  }

  _CCCL_HOST_DEVICE bool operator==(const key_value& rhs) const
  {
    return key == rhs.key && value == rhs.value;
  }

  _CCCL_HOST_DEVICE bool operator!=(const key_value& rhs) const
  {
    return !(*this == rhs);
  }

  friend std::ostream& operator<<(std::ostream& os, const key_value& kv)
  {
    return os << "(" << kv.key << ", " << kv.value << ")";
  }

  key_type key;
  value_type value;
};

struct user_swappable
{
  _CCCL_HOST_DEVICE user_swappable(bool swapped = false)
      : was_swapped(swapped)
  {}

  bool was_swapped;

  friend _CCCL_HOST_DEVICE bool operator==(const user_swappable& x, const user_swappable& y)
  {
    return x.was_swapped == y.was_swapped;
  }

  friend _CCCL_HOST_DEVICE void swap(user_swappable& x, user_swappable& y) noexcept
  {
    x.was_swapped = true;
    y.was_swapped = false;
  }
};

// A type that behaves as if it was a normal numeric type,
// so it can be used in the same tests as "normal" numeric types.
// NOTE: This is explicitly NOT proclaimed trivially reloctable.
class custom_numeric
{
public:
  _CCCL_HOST_DEVICE custom_numeric()
  {
    fill(0);
  }

  // Allow construction from any integral numeric.
  template <typename T, typename = typename std::enable_if<std::is_integral<T>::value>::type>
  _CCCL_HOST_DEVICE custom_numeric(const T& i)
  {
    fill(static_cast<int>(i));
  }

  _CCCL_HOST_DEVICE custom_numeric(const custom_numeric& other)
  {
    fill(other.value[0]);
  }

  _CCCL_HOST_DEVICE custom_numeric& operator=(int val)
  {
    fill(val);
    return *this;
  }

  _CCCL_HOST_DEVICE custom_numeric& operator=(const custom_numeric& other)
  {
    fill(other.value[0]);
    return *this;
  }

  // cast to void * instead of bool to fool overload resolution
  // WTB C++11 explicit conversion operators
  _CCCL_HOST_DEVICE operator void*() const
  {
    // static cast first to avoid MSVC warning C4312
    return reinterpret_cast<void*>(static_cast<std::size_t>(value[0]));
  }

#define DEFINE_OPERATOR(op)                               \
  _CCCL_HOST_DEVICE custom_numeric& operator op()         \
  {                                                       \
    fill(op value[0]);                                    \
    return *this;                                         \
  }                                                       \
  _CCCL_HOST_DEVICE custom_numeric operator op(int) const \
  {                                                       \
    custom_numeric ret(*this);                            \
    op ret;                                               \
    return ret;                                           \
  }

  DEFINE_OPERATOR(++)
  DEFINE_OPERATOR(--)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                            \
  _CCCL_HOST_DEVICE custom_numeric operator op() const \
  {                                                    \
    return custom_numeric(op value[0]);                \
  }

  DEFINE_OPERATOR(+)
  DEFINE_OPERATOR(-)
  DEFINE_OPERATOR(~)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                       \
  _CCCL_HOST_DEVICE custom_numeric operator op(const custom_numeric& other) const \
  {                                                                               \
    return custom_numeric(value[0] op other.value[0]);                            \
  }

  DEFINE_OPERATOR(+)
  DEFINE_OPERATOR(-)
  DEFINE_OPERATOR(*)
  DEFINE_OPERATOR(/)
  DEFINE_OPERATOR(%)
  DEFINE_OPERATOR(<<)
  DEFINE_OPERATOR(>>)
  DEFINE_OPERATOR(&)
  DEFINE_OPERATOR(|)
  DEFINE_OPERATOR(^)

#undef DEFINE_OPERATOR

#define CONCAT(X, Y) X##Y

#define DEFINE_OPERATOR(op)                                                              \
  _CCCL_HOST_DEVICE custom_numeric& operator CONCAT(op, =)(const custom_numeric & other) \
  {                                                                                      \
    fill(value[0] op other.value[0]);                                                    \
    return *this;                                                                        \
  }

  DEFINE_OPERATOR(+)
  DEFINE_OPERATOR(-)
  DEFINE_OPERATOR(*)
  DEFINE_OPERATOR(/)
  DEFINE_OPERATOR(%)
  DEFINE_OPERATOR(<<)
  DEFINE_OPERATOR(>>)
  DEFINE_OPERATOR(&)
  DEFINE_OPERATOR(|)
  DEFINE_OPERATOR(^)

#undef DEFINE_OPERATOR

#define DEFINE_OPERATOR(op)                                                                       \
  _CCCL_HOST_DEVICE friend bool operator op(const custom_numeric& lhs, const custom_numeric& rhs) \
  {                                                                                               \
    return lhs.value[0] op rhs.value[0];                                                          \
  }

  DEFINE_OPERATOR(==)
  DEFINE_OPERATOR(!=)
  DEFINE_OPERATOR(<)
  DEFINE_OPERATOR(<=)
  DEFINE_OPERATOR(>)
  DEFINE_OPERATOR(>=)
  DEFINE_OPERATOR(&&)
  DEFINE_OPERATOR(||)

#undef DEFINE_OPERATOR

  friend std::ostream& operator<<(std::ostream& os, const custom_numeric& val)
  {
    return os << "custom_numeric{" << val.value[0] << "}";
  }

private:
  int value[5];

  _CCCL_HOST_DEVICE void fill(int val)
  {
    for (int i = 0; i < 5; ++i)
    {
      value[i] = val;
    }
  }
};

namespace std
{
template <>
struct numeric_limits<custom_numeric> : numeric_limits<int>
{};
} // namespace std

_LIBCUDACXX_BEGIN_NAMESPACE_STD
template <>
struct numeric_limits<custom_numeric> : numeric_limits<int>
{};
_LIBCUDACXX_END_NAMESPACE_STD

// Inheriting from classes in anonymous namespaces is not allowed.
// The anonymous namespace tests don't use these, so just disable them:
#ifndef THRUST_USE_ANON_NAMESPACE

struct my_system : THRUST_NS_QUALIFIER::device_execution_policy<my_system>
{
  my_system(int) {}

  my_system(const my_system& other)
      : num_copies(other.num_copies + 1)
  {}

  void validate_dispatch()
  {
    correctly_dispatched = (num_copies == 0);
  }

  bool is_valid() const
  {
    return correctly_dispatched;
  }

private:
  bool correctly_dispatched = false;

  // count the number of copies so that we can validate
  // that dispatch does not introduce any
  unsigned int num_copies = 0;
};

struct my_tag : THRUST_NS_QUALIFIER::device_execution_policy<my_tag>
{};

#endif // THRUST_USE_ANON_NAMESPACE

namespace unittest
{

using std::int16_t;
using std::int32_t;
using std::int64_t;
using std::int8_t;

using std::uint16_t;
using std::uint32_t;
using std::uint64_t;
using std::uint8_t;

} // namespace unittest
