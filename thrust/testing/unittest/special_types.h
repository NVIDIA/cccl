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

  _CCCL_HOST_DEVICE
#if _CCCL_COMPILER(NVHPC)
  __attribute__((noinline))
#endif
  FixedVector
  operator+(const FixedVector& bs) const
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
