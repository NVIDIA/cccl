#ifndef TEST_SUPPORT_ARCHETYPES_H
#define TEST_SUPPORT_ARCHETYPES_H

#include <cuda/std/type_traits>
#include <cuda/std/cassert>

#include "test_macros.h"
#include "test_workarounds.h"

namespace ArchetypeBases {

template <bool, class T>
struct DepType : T {};

struct NullBase {
#ifndef TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
  NullBase() = default;
  NullBase(NullBase const&) = default;
  NullBase& operator=(NullBase const&) = default;
  NullBase(NullBase &&) = default;
  NullBase& operator=(NullBase &&) = default;
};

template <class Derived, bool Explicit = false>
struct TestBase {
    STATIC_MEMBER_VAR(alive,               int);
    STATIC_MEMBER_VAR(constructed,         int);
    STATIC_MEMBER_VAR(value_constructed,   int);
    STATIC_MEMBER_VAR(default_constructed, int);
    STATIC_MEMBER_VAR(copy_constructed,    int);
    STATIC_MEMBER_VAR(move_constructed,    int);
    STATIC_MEMBER_VAR(assigned,            int);
    STATIC_MEMBER_VAR(value_assigned,      int);
    STATIC_MEMBER_VAR(copy_assigned,       int);
    STATIC_MEMBER_VAR(move_assigned,       int);
    STATIC_MEMBER_VAR(destroyed,           int);

    TEST_HOST_DEVICE static void reset() {
        assert(alive() == 0);
        alive() = 0;
        reset_constructors();
    }

    TEST_HOST_DEVICE static void reset_constructors() {
      constructed() = value_constructed() = default_constructed() =
        copy_constructed() = move_constructed() = 0;
      assigned() = value_assigned() = copy_assigned() = move_assigned() = destroyed() = 0;
    }

    TEST_HOST_DEVICE TestBase() noexcept : value(0) {
        ++alive(); ++constructed(); ++default_constructed();
    }
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit TestBase(int x) noexcept : value(x) {
        ++alive(); ++constructed(); ++value_constructed();
    }
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE TestBase(int x) noexcept : value(x) {
        ++alive(); ++constructed(); ++value_constructed();
    }
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit TestBase(int, int y) noexcept : value(y) {
        ++alive(); ++constructed(); ++value_constructed();
    }
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE TestBase(int, int y) noexcept : value(y) {
        ++alive(); ++constructed(); ++value_constructed();
    }
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit TestBase(std::initializer_list<int>& il, int = 0) noexcept
      : value(static_cast<int>(il.size())) {
        ++alive(); ++constructed(); ++value_constructed();
    }
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit TestBase(std::initializer_list<int>& il, int = 0) noexcept : value(static_cast<int>(il.size())) {
        ++alive(); ++constructed(); ++value_constructed();
    }
    TEST_HOST_DEVICE TestBase& operator=(int xvalue) noexcept {
      value = xvalue;
      ++assigned(); ++value_assigned();
      return *this;
    }
#if !defined(TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK) && !defined(TEST_COMPILER_NVRTC)
protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
    TEST_HOST_DEVICE ~TestBase() {
      assert(value != -999); assert(alive() > 0);
      --alive(); ++destroyed(); value = -999;
    }
    TEST_HOST_DEVICE explicit TestBase(TestBase const& o) noexcept : value(o.value) {
        assert(o.value != -1); assert(o.value != -999);
        ++alive(); ++constructed(); ++copy_constructed();
    }
    TEST_HOST_DEVICE explicit TestBase(TestBase && o) noexcept : value(o.value) {
        assert(o.value != -1); assert(o.value != -999);
        ++alive(); ++constructed(); ++move_constructed();
        o.value = -1;
    }
    TEST_HOST_DEVICE TestBase& operator=(TestBase const& o) noexcept {
      assert(o.value != -1); assert(o.value != -999);
      ++assigned(); ++copy_assigned();
      value = o.value;
      return *this;
    }
    TEST_HOST_DEVICE TestBase& operator=(TestBase&& o) noexcept {
        assert(o.value != -1); assert(o.value != -999);
        ++assigned(); ++move_assigned();
        value = o.value;
        o.value = -1;
        return *this;
    }
public:
    int value;
};

template <bool Explicit = false>
struct ValueBase {
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit constexpr ValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE constexpr ValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit constexpr ValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE constexpr ValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit constexpr ValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE constexpr ValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 ValueBase& operator=(int xvalue) noexcept {
        value = xvalue;
        return *this;
    }
    //~ValueBase() { assert(value != -999); value = -999; }
    int value;
#ifndef TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
    TEST_HOST_DEVICE constexpr static int check_value(int const& val) {
#if TEST_STD_VER < 2014
      return val == -1 || val == 999 ? (TEST_THROW(42), 0) : val;
#else
      assert(val != -1); assert(val != 999);
      return val;
#endif
    }
    TEST_HOST_DEVICE constexpr static int check_value(int& val, int val_cp = 0) {
#if TEST_STD_VER < 2014
      return val_cp = val, val = -1, (val_cp == -1 || val_cp == 999 ? (TEST_THROW(42), 0) : val_cp);
#else
      assert(val != -1); assert(val != 999);
      val_cp = val;
      val = -1;
      return val_cp;
#endif
    }
    TEST_HOST_DEVICE constexpr ValueBase() noexcept : value(0) {}
    TEST_HOST_DEVICE constexpr ValueBase(ValueBase const& o) noexcept : value(check_value(o.value)) {
    }
    TEST_HOST_DEVICE constexpr ValueBase(ValueBase && o) noexcept : value(check_value(o.value)) {
    }
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 ValueBase& operator=(ValueBase const& o) noexcept {
        assert(o.value != -1); assert(o.value != -999);
        value = o.value;
        return *this;
    }
    TEST_HOST_DEVICE TEST_CONSTEXPR_CXX14 ValueBase& operator=(ValueBase&& o) noexcept {
        assert(o.value != -1); assert(o.value != -999);
        value = o.value;
        o.value = -1;
        return *this;
    }
};


template <bool Explicit = false>
struct TrivialValueBase {
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit constexpr TrivialValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE constexpr TrivialValueBase(int x) : value(x) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit constexpr TrivialValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE constexpr TrivialValueBase(int, int y) : value(y) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && Explicit, bool>::type = true>
    TEST_HOST_DEVICE explicit constexpr TrivialValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    template <bool Dummy = true, typename cuda::std::enable_if<Dummy && !Explicit, bool>::type = true>
    TEST_HOST_DEVICE constexpr TrivialValueBase(std::initializer_list<int>& il, int = 0) : value(static_cast<int>(il.size())) {}
    int value;
#ifndef TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
protected:
#endif // !TEST_WORKAROUND_C1XX_BROKEN_ZA_CTOR_CHECK
    TEST_HOST_DEVICE constexpr TrivialValueBase() noexcept : value(0) {}
};

}

//============================================================================//
// Trivial Implicit Test Types
namespace ImplicitTypes {
#include "archetypes.ipp"
}

//============================================================================//
// Trivial Explicit Test Types
namespace ExplicitTypes {
#define DEFINE_EXPLICIT explicit
#include "archetypes.ipp"
}

//============================================================================//
//
namespace NonConstexprTypes {
#define DEFINE_CONSTEXPR
#include "archetypes.ipp"
}

//============================================================================//
// Non-literal implicit test types
namespace NonLiteralTypes {
#define DEFINE_ASSIGN_CONSTEXPR
#define DEFINE_DTOR(Name) TEST_HOST_DEVICE ~Name() {}
#include "archetypes.ipp"
}

//============================================================================//
// Non-throwing implicit test types
namespace NonThrowingTypes {
#define DEFINE_NOEXCEPT noexcept
#include "archetypes.ipp"
}

//============================================================================//
// Non-Trivially Copyable Implicit Test Types
namespace NonTrivialTypes {
#define DEFINE_CTOR {}
#define DEFINE_CTOR_ANNOTATIONS TEST_HOST_DEVICE
#define DEFINE_ASSIGN { return *this; }
#define DEFINE_ASSIGN_ANNOTATIONS TEST_HOST_DEVICE
#include "archetypes.ipp"
}

//============================================================================//
// Implicit counting types
namespace TestTypes {
#define DEFINE_CONSTEXPR
#define DEFINE_BASE(Name) ::ArchetypeBases::TestBase<Name>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
TEST_HOST_DEVICE constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
TEST_HOST_DEVICE constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

}

//============================================================================//
// Implicit counting types
namespace ExplicitTestTypes {
#define DEFINE_CONSTEXPR
#define DEFINE_EXPLICIT explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::TestBase<Name, true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
TEST_HOST_DEVICE constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
TEST_HOST_DEVICE constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

}

//============================================================================//
// Implicit value types
namespace ConstexprTestTypes {
#define DEFINE_BASE(Name) ::ArchetypeBases::ValueBase<>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
TEST_HOST_DEVICE constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
TEST_HOST_DEVICE constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // end namespace ConstexprTestTypes


//============================================================================//
//
namespace ExplicitConstexprTestTypes {
#define DEFINE_EXPLICIT explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::ValueBase<true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
TEST_HOST_DEVICE constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
TEST_HOST_DEVICE constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // end namespace ExplicitConstexprTestTypes


//============================================================================//
//
namespace TrivialTestTypes {
#define DEFINE_BASE(Name) ::ArchetypeBases::TrivialValueBase<false>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
TEST_HOST_DEVICE constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
TEST_HOST_DEVICE constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // end namespace TrivialTestTypes

//============================================================================//
//
namespace ExplicitTrivialTestTypes {
#define DEFINE_EXPLICIT explicit
#define DEFINE_BASE(Name) ::ArchetypeBases::TrivialValueBase<true>
#include "archetypes.ipp"

using TestType = AllCtors;

// Add equality operators
template <class Tp>
TEST_HOST_DEVICE constexpr bool operator==(Tp const& L, Tp const& R) noexcept {
  return L.value == R.value;
}

template <class Tp>
TEST_HOST_DEVICE constexpr bool operator!=(Tp const& L, Tp const& R) noexcept {
  return L.value != R.value;
}

} // end namespace ExplicitTrivialTestTypes

#endif // TEST_SUPPORT_ARCHETYPES_H
