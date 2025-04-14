#ifndef TEST_SUPPORT_ARCHETYPES_H
#define TEST_SUPPORT_ARCHETYPES_H

#include <cuda/std/cassert>
#include <cuda/std/type_traits>

#include "test_macros.h"
#include "test_workarounds.h"

namespace ArchetypeBases
{

template <bool, class T>
struct DepType : T
{};

struct NullBase
{
#ifndef TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK

protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
  NullBase()                           = default;
  NullBase(NullBase const&)            = default;
  NullBase& operator=(NullBase const&) = default;
  NullBase(NullBase&&)                 = default;
  NullBase& operator=(NullBase&&)      = default;
};

template <class Derived, bool Explicit = false>
struct TestBase
{
  STATIC_MEMBER_VAR(alive, int)
  STATIC_MEMBER_VAR(constructed, int)
  STATIC_MEMBER_VAR(value_constructed, int)
  STATIC_MEMBER_VAR(default_constructed, int)
  STATIC_MEMBER_VAR(copy_constructed, int)
  STATIC_MEMBER_VAR(move_constructed, int)
  STATIC_MEMBER_VAR(assigned, int)
  STATIC_MEMBER_VAR(value_assigned, int)
  STATIC_MEMBER_VAR(copy_assigned, int)
  STATIC_MEMBER_VAR(move_assigned, int)
  STATIC_MEMBER_VAR(destroyed, int)

  __host__ __device__ static void reset()
  {
    assert(alive() == 0);
    alive() = 0;
    reset_constructors();
  }

  __host__ __device__ static void reset_constructors()
  {
    constructed() = value_constructed() = default_constructed() = copy_constructed() = move_constructed() = 0;
    assigned() = value_assigned() = copy_assigned() = move_assigned() = destroyed() = 0;
  }

  __host__ __device__ TestBase() noexcept
      : value(0)
  {
    ++alive();
    ++constructed();
    ++default_constructed();
  }
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit TestBase(int x) noexcept
      : value(x)
  {
    ++alive();
    ++constructed();
    ++value_constructed();
  }
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ TestBase(int x) noexcept
      : value(x)
  {
    ++alive();
    ++constructed();
    ++value_constructed();
  }
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit TestBase(int, int y) noexcept
      : value(y)
  {
    ++alive();
    ++constructed();
    ++value_constructed();
  }
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ TestBase(int, int y) noexcept
      : value(y)
  {
    ++alive();
    ++constructed();
    ++value_constructed();
  }
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit TestBase(std::initializer_list<int>& il, int = 0) noexcept
      : value(static_cast<int>(il.size()))
  {
    ++alive();
    ++constructed();
    ++value_constructed();
  }
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ explicit TestBase(std::initializer_list<int>& il, int = 0) noexcept
      : value(static_cast<int>(il.size()))
  {
    ++alive();
    ++constructed();
    ++value_constructed();
  }
  __host__ __device__ TestBase& operator=(int xvalue) noexcept
  {
    value = xvalue;
    ++assigned();
    ++value_assigned();
    return *this;
  }
#if !defined(TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK) && !TEST_COMPILER(NVRTC)

protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
  __host__ __device__ ~TestBase()
  {
    assert(value != -999);
    assert(alive() > 0);
    --alive();
    ++destroyed();
    value = -999;
  }
  __host__ __device__ explicit TestBase(TestBase const& o) noexcept
      : value(o.value)
  {
    assert(o.value != -1);
    assert(o.value != -999);
    ++alive();
    ++constructed();
    ++copy_constructed();
  }
  __host__ __device__ explicit TestBase(TestBase&& o) noexcept
      : value(o.value)
  {
    assert(o.value != -1);
    assert(o.value != -999);
    ++alive();
    ++constructed();
    ++move_constructed();
    o.value = -1;
  }
  __host__ __device__ TestBase& operator=(TestBase const& o) noexcept
  {
    assert(o.value != -1);
    assert(o.value != -999);
    ++assigned();
    ++copy_assigned();
    value = o.value;
    return *this;
  }
  __host__ __device__ TestBase& operator=(TestBase&& o) noexcept
  {
    assert(o.value != -1);
    assert(o.value != -999);
    ++assigned();
    ++move_assigned();
    value   = o.value;
    o.value = -1;
    return *this;
  }

public:
  int value;
};

template <bool Explicit = false>
struct ValueBase
{
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit constexpr ValueBase(int x)
      : value(x)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ constexpr ValueBase(int x)
      : value(x)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit constexpr ValueBase(int, int y)
      : value(y)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ constexpr ValueBase(int, int y)
      : value(y)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit constexpr ValueBase(std::initializer_list<int>& il, int = 0)
      : value(static_cast<int>(il.size()))
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ constexpr ValueBase(std::initializer_list<int>& il, int = 0)
      : value(static_cast<int>(il.size()))
  {}
  __host__ __device__ constexpr ValueBase& operator=(int xvalue) noexcept
  {
    value = xvalue;
    return *this;
  }
  //~ValueBase() { assert(value != -999); value = -999; }
  int value;
#ifndef TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK

protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
  __host__ __device__ constexpr static int check_value(int const& val)
  {
    assert(val != -1);
    assert(val != 999);
    return val;
  }
  __host__ __device__ constexpr static int check_value(int& val, int val_cp = 0)
  {
    assert(val != -1);
    assert(val != 999);
    val_cp = val;
    val    = -1;
    return val_cp;
  }
  __host__ __device__ constexpr ValueBase() noexcept
      : value(0)
  {}
  __host__ __device__ constexpr ValueBase(ValueBase const& o) noexcept
      : value(check_value(o.value))
  {}
  __host__ __device__ constexpr ValueBase(ValueBase&& o) noexcept
      : value(check_value(o.value))
  {}
  __host__ __device__ constexpr ValueBase& operator=(ValueBase const& o) noexcept
  {
    assert(o.value != -1);
    assert(o.value != -999);
    value = o.value;
    return *this;
  }
  __host__ __device__ constexpr ValueBase& operator=(ValueBase&& o) noexcept
  {
    assert(o.value != -1);
    assert(o.value != -999);
    value   = o.value;
    o.value = -1;
    return *this;
  }
};

template <bool Explicit = false>
struct TrivialValueBase
{
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit constexpr TrivialValueBase(int x)
      : value(x)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ constexpr TrivialValueBase(int x)
      : value(x)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit constexpr TrivialValueBase(int, int y)
      : value(y)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ constexpr TrivialValueBase(int, int y)
      : value(y)
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
  __host__ __device__ explicit constexpr TrivialValueBase(std::initializer_list<int>& il, int = 0)
      : value(static_cast<int>(il.size()))
  {}
  template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
  __host__ __device__ constexpr TrivialValueBase(std::initializer_list<int>& il, int = 0)
      : value(static_cast<int>(il.size()))
  {}
  int value;
#ifndef TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK

protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
  __host__ __device__ constexpr TrivialValueBase() noexcept
      : value(0)
  {}
};

} // namespace ArchetypeBases

//============================================================================//
// Trivial Implicit Test Types
namespace ImplicitTypes
{
#include "archetypes.ipp"
}

//============================================================================//
// Trivial Explicit Test Types
namespace ExplicitTypes
{
#define DEFINE_EXPLICIT explicit
#include "archetypes.ipp"
} // namespace ExplicitTypes

//============================================================================//
//
namespace NonConstexprTypes
{
#define DEFINE_CONSTEXPR
#include "archetypes.ipp"
} // namespace NonConstexprTypes

//============================================================================//
// Non-literal implicit test types
namespace NonLiteralTypes
{
#define DEFINE_ASSIGN_CONSTEXPR
#define DEFINE_DTOR(Name) \
  __host__ __device__ ~Name() {}
#include "archetypes.ipp"
} // namespace NonLiteralTypes

//============================================================================//
// Non-throwing implicit test types
namespace NonThrowingTypes
{
#define DEFINE_NOEXCEPT noexcept
#include "archetypes.ipp"
} // namespace NonThrowingTypes

//============================================================================//
// Non-Trivially Copyable Implicit Test Types
namespace NonTrivialTypes
{
#define DEFINE_CTOR \
  {                 \
  }
#define DEFINE_CTOR_ANNOTATIONS __host__ __device__
#define DEFINE_ASSIGN \
  {                   \
    return *this;     \
  }
#define DEFINE_ASSIGN_ANNOTATIONS __host__ __device__
#include "archetypes.ipp"
} // namespace NonTrivialTypes

//============================================================================//
// Implicit counting types
namespace TestTypes
{
#define DEFINE_CONSTEXPR
#define DEFINE_BASE(Name) ::ArchetypeBases::TestBase<Name>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
__host__ __device__ constexpr bool operator==(Tp const& L, Tp const& R) noexcept
{
  return L.value == R.value;
}

template <class Tp>
__host__ __device__ constexpr bool operator!=(Tp const& L, Tp const& R) noexcept
{
  return L.value != R.value;
}

} // namespace TestTypes

//============================================================================//
// Implicit counting types
namespace ExplicitTestTypes
{
#define DEFINE_CONSTEXPR
#define DEFINE_EXPLICIT   explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::TestBase<Name, true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
__host__ __device__ constexpr bool operator==(Tp const& L, Tp const& R) noexcept
{
  return L.value == R.value;
}

template <class Tp>
__host__ __device__ constexpr bool operator!=(Tp const& L, Tp const& R) noexcept
{
  return L.value != R.value;
}

} // namespace ExplicitTestTypes

//============================================================================//
// Implicit value types
namespace ConstexprTestTypes
{
#define DEFINE_BASE(Name) ::ArchetypeBases::ValueBase<>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
__host__ __device__ constexpr bool operator==(Tp const& L, Tp const& R) noexcept
{
  return L.value == R.value;
}

template <class Tp>
__host__ __device__ constexpr bool operator!=(Tp const& L, Tp const& R) noexcept
{
  return L.value != R.value;
}

} // end namespace ConstexprTestTypes

//============================================================================//
//
namespace ExplicitConstexprTestTypes
{
#define DEFINE_EXPLICIT   explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::ValueBase<true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
__host__ __device__ constexpr bool operator==(Tp const& L, Tp const& R) noexcept
{
  return L.value == R.value;
}

template <class Tp>
__host__ __device__ constexpr bool operator!=(Tp const& L, Tp const& R) noexcept
{
  return L.value != R.value;
}

} // end namespace ExplicitConstexprTestTypes

//============================================================================//
//
namespace TrivialTestTypes
{
#define DEFINE_BASE(Name) ::ArchetypeBases::TrivialValueBase<false>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
__host__ __device__ constexpr bool operator==(Tp const& L, Tp const& R) noexcept
{
  return L.value == R.value;
}

template <class Tp>
__host__ __device__ constexpr bool operator!=(Tp const& L, Tp const& R) noexcept
{
  return L.value != R.value;
}

} // end namespace TrivialTestTypes

//============================================================================//
//
namespace ExplicitTrivialTestTypes
{
#define DEFINE_EXPLICIT   explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::TrivialValueBase<true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
__host__ __device__ constexpr bool operator==(Tp const& L, Tp const& R) noexcept
{
  return L.value == R.value;
}

template <class Tp>
__host__ __device__ constexpr bool operator!=(Tp const& L, Tp const& R) noexcept
{
  return L.value != R.value;
}

} // end namespace ExplicitTrivialTestTypes

#endif // TEST_SUPPORT_ARCHETYPES_H
