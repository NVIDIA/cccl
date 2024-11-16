/*
 * Copyright (c) 2021-2024 NVIDIA Corporation
 *
 * Licensed under the Apache License Version 2.0 with LLVM Exceptions
 * (the "License"); you may not use this file except in compliance with
 * the License. You may obtain a copy of the License at
 *
 *   https://llvm.org/LICENSE.txt
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */
#ifndef __CUDAX_DETAIL_BASIC_ANY_H
#define __CUDAX_DETAIL_BASIC_ANY_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/__algorithm/max.h>
#include <cuda/std/__concepts/__concept_macros.h>
#include <cuda/std/__concepts/convertible_to.h>
#include <cuda/std/__concepts/copyable.h>
#include <cuda/std/__concepts/derived_from.h>
#include <cuda/std/__concepts/equality_comparable.h>
#include <cuda/std/__concepts/movable.h>
#include <cuda/std/__concepts/regular.h>
#include <cuda/std/__concepts/same_as.h>
#include <cuda/std/__exception/terminate.h>
#include <cuda/std/__new/launder.h>
#include <cuda/std/__type_traits/conditional.h>
#include <cuda/std/__type_traits/is_callable.h>
#include <cuda/std/__type_traits/is_same.h>
#include <cuda/std/__type_traits/is_valid_expansion.h>
#include <cuda/std/__type_traits/maybe_const.h>
#include <cuda/std/__type_traits/negation.h>
#include <cuda/std/__type_traits/remove_cvref.h>
#include <cuda/std/__type_traits/remove_pointer.h>
#include <cuda/std/__type_traits/type_identity.h>
#include <cuda/std/__type_traits/type_list.h>
#include <cuda/std/__type_traits/type_set.h>
#include <cuda/std/__utility/declval.h>
#include <cuda/std/__utility/in_place.h>
#include <cuda/std/__utility/swap.h>
#include <cuda/std/__utility/typeid.h>

#include <cuda/experimental/__detail/config.cuh>
#include <cuda/experimental/__detail/utility.cuh>

#include <typeinfo> // IWYU pragma: keep (for std::bad_cast)

#include <nv/target>

//! @file basic_any.hpp
//!
//! @brief This file provides the implementation of the `basic_any` class
//! template.
//!
//! The `basic_any` class template generates a type-erasing wrapper for types
//! described by a user-provided _interface_. An interface is a class template
//! that has member and/or friend functions that dispatch to the appropriate
//! pseudo-virtual functions, and a nested `overrides` type alias that maps the
//! members of a given type to the pseudo-virtuals. Objects that implement the
//! interface will be stored in-situ in the `basic_any` object if possible;
//! otherwise, they will be stored on the heap.
//!
//! A simple interface looks like this (where `cudax` is an alias for
//! `cuda::experimental`):
//!
//! @code
//! template <class...>
//! struct icat : cudax::interface<icat>
//! {
//!   void meow() const
//!   {
//!     // dispatch to the icat::meow override:
//!     cudax::virtcall<&icat::meow>(this);
//!   }
//!
//!   template <class Cat>
//!   using overrides = cudax::overrides_for<Cat, &Cat::meow>;
//! };
//!
//! // `any_cat` is a type-erasing wrapper that can store any object whose type
//! // satisfies the `icat` interface.
//! using any_cat = cudax::basic_any<icat<>>;
//! @endcode
//!
//! The `any_cat` type can now be used to store any object that implements the
//! `icat` interface. For example:
//!
//! @code
//! struct siamese
//! {
//!   void meow() const
//!   {
//!     std::cout << "meow" << std::endl;
//!   }
//! };
//!
//! any_cat cat = siamese{};
//! cat.meow(); // prints "meow"
//! @endcode
//!
//!
//! ## The `basic_any` type
//!
//! For an interface `I`, the `basic_any<I>` has the following member functions
//! in addition to those provided by `I`:
//!
//! @code
//! void swap(basic_any<I>& other) noexcept;
//!
//! template <class T, class... Args>
//! T& emplace(Args&&... args);
//!
//! template <class T, class U, class... Args>
//! T& emplace(std::initializer_list<U> il, Args&&... args);
//!
//! bool has_value() const noexcept;
//!
//! void reset() noexcept;
//!
//! const std::type_info& type() const noexcept;
//!
//! const std::type_info& interface() const noexcept;
//!
//! bool __in_situ() const noexcept;
//! @endcode
//!
//! These members behave as they do for `std::any`. The `interface()` member
//! function returns the `std::type_info` object for the _dynamic_ interface of
//! the stored object, which could be different from `I` if the `basic_any`
//! object was constructed from a `basic_any<J>`, where `J` extends `I` (see
//! the next section).
//!
//! ## Interface Extension
//!
//! An interface can extend other interfaces. We can make `any_cat` copyable by
//! extending the `cudax::icopyable` interface:
//!
//! @code
//! template <class...>
//! struct icat
//!   : cudax::interface<icat, cudax::extends<cudax::icopyable<>>>
//! {
//!   ... as before ...
//! @endcode
//!
//! The `cudax::extends` template is variadic to allow extending multiple
//! interfaces. Interface extension works like virtual inheritance in C++; the
//! same interface can be inherited multiple times, either directly or
//! indirectly, without creating any ambiguity.
//!
//! Conversion from a "derived" `basic_any` (e.g., `any_cat`) to a "base"
//! `basic_any` (e.g., `basic_any<icopyable<>>`) is supported.
//!
//! ## Pointers and References
//!
//! The `basic_any` class template can hold pointers and references to objects
//! that implement the interface. For example:
//!
//! @code
//! // `any_cat_ptr` is a type-erasing wrapper that can store a pointer to any
//! // (non-const) object whose type implements the `icat` interface.
//! using any_cat_ptr = cudax::basic_any<icat<>*>; // note the pointer type!
//!
//! // store a pointer to a siamese cat in a basic_any object
//! siamese fluffy;
//! any_cat_ptr pcat1 = &fluffy;
//! pcat1->meow(); // prints "meow"
//!
//! // a basic_any pointer can also point to a type-erased object:
//! any_cat cat2 = fluffy;
//! any_cat_ptr pcat2 = &cat2;
//! pcat2->meow(); // prints "meow"
//! @endcode
//!
//! If the base interface (in this case, `icat<>`) extends `icopyable<>` or
//! `imovable<>`, then you can copy or move out of a dereferenced basic_any
//! pointer. For example:
//!
//! @code
//! any_cat_ptr pcat = &fluffy;
//! any_cat copycat = *pcat;            // copy from fluffy
//! any_cat movecat = std::move(*pcat); // move from fluffy
//! @endcode
//!
//! Type-erased references are supported as well. For example:
//!
//! @code
//! // `any_cat_ref` is a type-erasing wrapper that can store a (non-owning)
//! // reference to any (non-const) object whose type implements the `icat`
//! // interface.
//! using any_cat_ref = cudax::basic_any<icat<>&>; // note the reference type!
//!
//! // store a non-const reference to a siamese cat in a basic_any object:
//! siamese fluffy;
//! any_cat_ref rcat1 = fluffy;
//! rcat1.meow(); // prints "meow"
//! @endcode
//!
//! All the conversion sequences that are valid for pointers and references are
//! also valid for basic_any pointers and references. For example, a
//! `basic_any<I*>` can be implicitly converted to a `basic_any<I const*>`, and
//! a `basic_any<IDerived&>` is implicitly convertible to a
//! `basic_any<IBase&>`.
//!
//! @warning **You cannot use `std::move` to move a value out of a basic_any
//! reference.** Although `any_cat value = std::move(rcat1);` will compile and
//! looks perfectly reasonable, it will not actually move the value out of
//! `rcat1`. Instead, it will copy the value. `std::move(rcat1)` is an rvalue
//! reference of type `any_cat_ref`, which continues to behave like an lvalue.
//! For this reason, basic_any references all have a `move` member function that
//! can be used to move the value out of the reference: `any_cat value =
//! rcat1.move();`.
//!
//! ## In-Situ Storage
//!
//! Small objects with non-throwing move constructors can be stored in-situ in
//! the `basic_any` object. The size of the in-situ buffer is determined by the
//! interface's `::size` member, which defaults to `3 * sizeof(void*)` and
//! cannot be smaller than `sizeof(void*)`. The alignment of the in-situ buffer
//! is determined by the interface's `::align` member, which defaults to
//! `alignof(std::max_align_t)` and cannot be smaller than `alignof(void*)`.
//!
//! For convenience, the `cudax::interface` template lets you specify the size
//! and alignment of the in-situ buffer directly:
//!
//! @code
//! template <class...>
//! struct icat
//!   : cudax::interface<icat,
//!                      cudax::extends<cudax::icopyable<>>,
//!                      2 * sizeof(void*),  // in-situ buffer size
//!                      alignof(void*)>     // in-situ buffer alignment
//! {
//!   ... as before ...
//! @endcode
//!
//! An object of type `T` will be stored in-situ in a `basic_any<I>` object if
//! the following conditions are met:
//!
//! - `I::size >= sizeof(T)`, and
//! - `I::align % alignof(T) == 0`, and
//! - `std::is_nothrow_move_constructible_v<T> == true`
//!
//! If any of these conditions are not met, the object will be stored on the
//! heap. When such a `basic_any` object is copied, a new `T` object will be
//! dynamically allocated and initialized with the old object. In other words,
//! `basic_any` objects have value semantics regardless of how the wrapped
//! object is stored. All `basic_any` objects are nothrow move constructible.
//!
//! `basic_any` pointer and reference types (`basic_any<I*>` and
//! `basic_any<I&>`) ignore the interface's `::size` and `::align` members.
//! Rather than an internal buffer, they store the pointer or reference
//! directly.
//!
//! ## Explicit Casting
//!
//! TODO
//!
//! ## Defining An Interface
//!
//! An interface must be a variadic class template that inherits from the
//! `cudax::interface` template as described above. The members of the interface
//! stub functions, aka thunks, that both define the interface and dispatch to
//! the type-erased pseudo-virtual overrides. You use the `cudax::virtcall`
//! function to dispatch to the pseudo-virtuals, using the member pointer of the
//! thunk as a key for dispatching to the correct function.
//!
//! The following example shows how to define an interface that will give us the
//! behavior of `std::function`. The interface itself needs to be parameterized
//! by the function signature, which involves some contortions:
//!
//! @code
//! // We use an outer template to bind the function signature to the interface.
//! template <class Signature>
//! struct _with_signature;
//!
//! template <class Ret, class... Args>
//! struct _with_signature<Ret(Args...)>
//! {
//!   // The actual interface is a variadic class template that inherits from
//!   // `cudax::interface` and contains the thunks.
//!   template <class...>
//!   struct _ifuntion
//!     : cudax::interface<_ifunction, cudax::extends<cudax::icopyable<>>>
//!   {
//!     Ret operator()(Args... args) const
//!     {
//!       return cudax::virtcall<&_ifunction::operator()>(this, std::forward<Args>(args)...);
//!     }
//!
//!     template <class F>
//!     using overrides = cudax::overrides_for<F, static_cast<Ret (F::*)(Args...)
//!     const>(&F::operator())>;
//!   };
//! };
//!
//! template <class Signature>
//! using ifunction = _with_signature<Signature>::template _ifunction<>;
//!
//! template <class Signature>
//! using any_function = cudax::basic_any<ifunction<Signature>>;
//! @endcode
//!
//! With the `any_function` template defined above, we can now store any callable
//! object that has the same signature as the `Signature` template parameter.
//!
//! @code
//! any_function<int(int, int)> fn = std::plus<int>{};
//! assert(fn(1, 2) == 3);
//! @endcode
//!
//! Within a thunk of an interface `I<>`, the `this` pointer has type
//! `I<J>*`, where `J` is the interface used to parameterize the `basic_any`
//! object. For example, imagine an interface `ialley_cat` that extends `icat`.
//! `basic_any<ialley_cat<>>` inherits publicly from `icat<ialley_cat<>>`, which
//! will be the type of the `this` pointer in the `icat<...>::meow()` thunk.
//!
//! Sometimes it is necessary to access the full `basic_any` object from within
//! the thunk. You can do this by passing the `this` pointer to the
//! `cudax::basic_any_from` function, which will return a pointer to the full
//! `basic_any` object; a `basic_any<ialley_cat<>>*` in this case.
//!
//! ## Using Free Functions As Overrides
//!
//! So far, we have only seen how to bind member functions to an interface. But
//! what if we want to bind a free function? The `cudax::overrides_for` template can
//! bind free functions to an interface as well. Here is an example that uses a
//! free function, `_equal_to<T>`, to define the `iequality_comparable`
//! interface:
//!
//! @code
//! template <class T>
//! auto _equal_to(T const& lhs, std::type_info const& rhs_type, void const* rhs)
//!   -> decltype(lhs == lhs) // SFINAE out if `lhs == lhs` is ill-formed
//! {
//!   return (rhs_type == typeid(T)) ? (lhs == *static_cast<T const*>(rhs)) : false;
//! }
//!
//! template <class...>
//! struct iequality_comparable : cudax::interface<iequality_comparable>
//! {
//!   friend bool operator==(iequality_comparable const& lhs, iequality_comparable const& rhs)
//!   {
//!     // downcast the `rhs` object to the correct specialization of `basic_any`:
//!     auto const& rhs_any  = cudax::basic_any_from(rhs);
//!     // get the object pointer from the `rhs` object:
//!     void const* rhs_optr = cudax::any_cast<void const>(&rhs_any);
//!     // dispatch to the `_equal_to` override:
//!     return cudax::virtcall<&_equal_to<iequality_comparable>>(this, rhs_any.type(), rhs_optr);
//!   }
//!
//!   friend bool operator!=(iequality_comparable const& lhs, iequality_comparable const& rhs)
//!   {
//!     return !(lhs == rhs);
//!   }
//!
//!   template <class T>
//!   using overrides = cudax::overrides_for<T, &_equal_to<T>>;
//! };
//! @endcode
//!
//! ## Implementation Details
//!
//! For an interface `I<>`, let `IBases<>...` be the (flattened, uniqued) list
//! of interfaces that `I<>` extends, either directly or indirectly.
//! `basic_any<I<>>` inherits publicly from `I<I<>>` and `IBases<I<>>...`.
//! Therefore, the members of `basic_any<I<>>` are the thunks of `I<>`
//! and all the interfaces that `I<>` extends.
//!
//! The `basic_any<I<>>` type looks roughly like this:
//!
//! @code
//! template <template <class...> class I>
//! struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<I<>> : I<I<>>, IBases<I<>>...
//! {
//!   // constructors, destructor, assignment operators, etc.
//!
//! private:
//!   // a tagged pointer to the object's vtable
//!   __tagged_ptr<__vtable<I<>> const*> __vptr_;
//!
//!   // the object's buffer
//!   alignas(I<>::align) std::byte __buffer_[I<>::size];
//! };
//! @endcode
//!
//! The `__tagged_ptr` type wraps a pointer that uses the lowest order bit to
//! indicate whether the buffer stores the object in-situ, or if it stores a
//! pointer to the object on the heap.
//!
//! The type `__vtable<I<>>` is a struct that aggregates the vtables of `I<>`
//! and all the interfaces that `I<>` extends. It also aggregates an "RTTI"
//! struct that contains metadata about the object and the vtable.
//! `__vtable<I<>>` is an alias for a type that looks like this:
//!
//! @code
//! template <class I, class... {base-interfaces-of-I}>
//! struct __vtable_tuple
//!   : __rtti_ex
//!   , __vtable_for<I>
//!   , __vtable_for<{base-interfaces-of-I}>...
//! {
//!    //... constexpr constructors ...
//!
//!    template <class Interface>
//!    __vtable_for<Interface> const* __query_interface(Interface) const noexcept;
//! };
//! @endcode
//!
//! The `__rtti_ex` struct contains a map from interface typeid's to the pointer
//! to the vtable for that interface. This map is used for `dynamic_cast`-like
//! functionality. `__rtti_ex` is derived from a `__rtti` class that has
//! metadata about the type-erased object like its typeid, size, and alignment.
//!
//! `__vtable_for<I<>>` names a struct that contains function pointers --
//! "overrides" -- for the pseudo-virtual functions of `I<>`. It is actually an
//! alias for a nested struct of the `cudax::overrides_for` template. The actual vtable
//! struct looks like this:
//!
//! @code
//! template <class InterfaceOrObject, auto... Mbrs>
//! struct overrides_for
//! {
//!   // InterfaceOrObject is either the type of the unspecialized interface or
//!   // the type of an object that implements the interface and that is being
//!   // type-erased.
//!   struct __vtable
//!     : __vtable_base         // described below
//!     , __virtual_fn<Mbrs>... // the pseudo-virtual overrides
//!   {
//!     // Within the __vtable struct, InterfaceOrObject is the type of the
//!     // unspecialized interface.
//!
//!     // ... constexpr constructors ...
//!
//!     template <class Interface>
//!     __vtable_for<Interface> const* __query_interface(Interface) const noexcept;
//!
//!     // Pointers to the vtables of I's bases:
//!     using I = InterfaceOrObject;
//!     __vtable_base const* __vptr_map[{count-of-I's-bases}];
//!   };
//! };
//! @endcode
//!
//! Here is the one trick that makes the whole of `basic_any` work:
//! `__vtable_for<I<>>` is an alias for `I<>::overrides<I<>>::__vtable`. For the
//! `icat` interface defined above, the `icat<>::overrides` alias looks like this:
//!
//! @code
//! template <class T>
//! using overrides = cudax::overrides_for<T, &T::meow>;
//! @endcode
//!
//! So `__vtable_for<icat<>>` is an alias for `cudax::overrides_for<icat<>,
//! &icat<>::meow>::__vtable`. `icat<>::meow` is the thunk in the `icat<>`
//! interface.
//!
//! We can now describe how pseudo-virtual dispatch works. Given an object `cat`
//! of type `basic_any<ialley_cat<>>` that we defined above, a call of
//! `cat.meow()` does the following:
//!
//! - Within the `meow` thunk, `this` has type `icat<ialley_cat<>>*`.
//!   The thunk calls `cudax::virtcall<&icat::meow>(this)`.
//!
//! - `&icat::meow` is a pointer to the `icat<ialley_cat<>>::meow` thunk. The
//!   `virtcall` infers from the `this` argument that the unspecialized
//!   interface is `icat<>`. `virtcall` maps the `&icat<ialley_cat<>>::meow`
//!   member pointer to the `&icat<>::meow` member pointer by building a map
//!   from `icat<ialley_cat<>>::overrides<icat<ialley_cat<>>` to
//!   `icat<>::overrides<icat<>>`.
//!
//! - Having established that the override for `icat<>::meow` is the one that
//!   should be called, `virtcall` calls
//!   `basic_any_from({the-this-ptr-from-above})` to obtain a pointer to the
//!   full `basic_any<ialley_cat<>>` object. From there, it pulls out the vtable
//!   pointer (which has type `__vtable<ialley_cat<>> const*`) and casts it to
//!   `__vtable_for<icat<>> const*`. (As described above,
//!   `__vtable<ialley_cat<>>` inherits from `__vtable_for<icat<>>`.) From
//!   there, it further casts the pointer to `__virtual_fn<&icat<>::meow>
//!   const*`, which contains the override for the `&icat<>::meow`
//!   pseudo-virtual.
//!
//! The `__virtual_fn` template wraps a function pointer that dispatches to the
//! correct member of the type-erased object. It is parameterized by the member
//! pointer of the thunk. `__virtual_fn<&icat<>::meow>` is handled by a
//! partial specialization that looks like this:
//!
//! @code
//! template <auto Mbr, class = decltype(Mbr)>
//! struct __virtual_fn;
//!
//! template <auto Mbr, class R, class C, class... Args>
//! struct __virtual_fn<Mbr, R (C::*)(Args...)>
//! {
//!   using __function_t = R(*)(void*, Args...);
//!   __function_t __fn_;
//! };
//! @endcode
//!
//! When a `basic_any<icat<>>` object is constructed from an object `fluffy` of
//! type `siamese`, a global constexpr object of type `__vtable<icat<>>` is
//! instantiated and initialized with `__tag<siamese>{}` (where `__tag` is an
//! empty struct). That will cause the base sub-object `__vtable_for<icat<>>` --
//! aka, `icat<>::overrides<icat<>>::__vtable`, aka `cudax::overrides_for<icat<>,
//! &icat<>::meow>::__vtable` -- to be initialized with an object of type
//! `icat<>::overrides<siamese>`, aka `cudax::overrides_for<siamese, &siamese::meow>`.
//! And that causes `__virtual_fn<&icat<>::meow>` to be initialized with a
//! pointer to a function that casts its `void*` argument to `siamese*` and
//! dispatches to its `meow` member function. So that's how the vtable gets
//! populated with the correct function pointers. In the `basic_any`
//! constructor, a pointer to that vtable is saved in its `__vptr_` member.

//------------------------------------------------------------------------------

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wmaybe-uninitialized")

// warning #20012-D: __device__ annotation is ignored on a function("foo") that
// is explicitly defaulted on its first declaration
_CCCL_NV_DIAG_SUPPRESS(20012)

#if defined(_CCCL_CUDA_COMPILER_NVCC) || defined(_CCCL_CUDA_COMPILER_NVHPC)
// WAR for NVBUG #4924416
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) ::cuda::experimental::__constant_war(__VA_ARGS__)
namespace cuda::experimental
{
template <class _Tp>
_CCCL_NODISCARD _CUDAX_API constexpr _Tp __constant_war(_Tp __val) noexcept
{
  return __val;
}
} // namespace cuda::experimental
#else
#  define _CUDAX_FNPTR_CONSTANT_WAR(...) __VA_ARGS__
#endif

// Some functions defined here have their addresses appear in public types
// (e.g., in `cudax::overrides_for` specializations). If the function is declared
// `__attribute__((visibility("hidden")))`, and if the address appears, say, in
// the type of a member of a class that is declared
// `__attribute__((visibility("default")))`, GCC complains bitterly. So we
// avoid declaring those functions `hidden`. Instead of the typical `_CUDAX_API`
// macro, we use `_CUDAX_PUBLIC_API` for those functions.
#define _CUDAX_PUBLIC_API _CCCL_HOST_DEVICE

namespace cuda::experimental
{
template <class _Interface>
struct __ireference;

template <class _Interface, class _Select = void>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any;

template <class _Interface, class _Select>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES basic_any<__ireference<_Interface>, _Select>;

template <class _Interface>
struct basic_any<_Interface*>;

template <class _Interface>
struct basic_any<_Interface&>;

template <class _InterfaceOrModel, auto... _Mbrs>
struct overrides_for;

template <class _Interface, auto... _Mbrs>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES __basic_vtable;

struct __rtti;

template <size_t NbrBases>
struct __rtti_ex;

template <class...>
struct extends;

template <template <class...> class, class = extends<>, size_t = 0, size_t = 0>
struct interface;

template <class _Interface, class... _Super>
using __rebind_interface = typename _Interface::template __rebind<_Super...>;

struct iunknown;

template <class...>
struct __iset;

template <class...>
struct __iset_vptr;

template <class...>
struct imovable;

template <class...>
struct icopyable;

template <class...>
struct iequality_comparable;

template <class... _Tp>
using __tag = _CUDA_VSTD::__type_list_ptr<_Tp...>;

template <auto...>
struct __ctag_;

template <auto... _Is>
using __ctag = __ctag_<_Is...>*;

template <class _Tp>
using __identity = _Tp;

constexpr size_t __word                 = sizeof(void*);
constexpr size_t __default_buffer_size  = 3 * __word;
constexpr size_t __default_buffer_align = alignof(_CUDA_VSTD::max_align_t);

_CCCL_NODISCARD _CUDAX_API inline constexpr size_t __buffer_size(size_t __size)
{
  /// round up to the nearest multiple of `__word`, which is the size of a
  /// void*.
  return ((__size ? (_CUDA_VSTD::max)(__size, sizeof(void*)) : __default_buffer_size) + __word - 1) / __word * __word;
}

_CCCL_NODISCARD _CUDAX_API inline constexpr size_t __buffer_align(size_t __align)
{
  /// need to be able to store a void* in the buffer.
  return __align ? (_CUDA_VSTD::max)(__align, alignof(void*)) : __default_buffer_align;
}

// constructible_from using list initialization syntax.
// clang-format off
template <class _Tp, class... _Args>
_LIBCUDACXX_CONCEPT __list_initializable_from =
  _LIBCUDACXX_REQUIRES_EXPR((_Tp, variadic _Args), _Args&&... __args)
  (
    _Tp{static_cast<_Args&&>(__args)...}
  );
// clang-format on

template <class _Tp>
_CCCL_NODISCARD _CUDAX_API inline constexpr bool __is_small(size_t __size, size_t __align) noexcept
{
  return (sizeof(_Tp) <= __size) && (__align % alignof(_Tp) == 0) && _CUDA_VSTD::is_nothrow_move_constructible_v<_Tp>;
}

_CUDAX_API inline void __swap_ptr_ptr(void* __lhs, void* __rhs) noexcept
{
  _CUDA_VSTD::swap(*static_cast<void**>(__lhs), *static_cast<void**>(__rhs));
}

template <class _Tp, class _Up, class _Vp = decltype(true ? __identity<_Tp*>() : __identity<_Up*>())>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API bool __ptr_eq(_Tp* __lhs, _Up* __rhs) noexcept
{
  return static_cast<_Vp>(__lhs) == static_cast<_Vp>(__rhs);
}

_CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr bool __ptr_eq(detail::__ignore, detail::__ignore) noexcept
{
  return false;
}

enum class __vtable_kind : uint8_t
{
  __normal,
  __rtti,
};

inline constexpr uint8_t __vtable_version = 0;

struct __vtable_base : detail::__immovable
{
  _CUDAX_API constexpr __vtable_base(
    __vtable_kind __kind, uint16_t __nbr_interfaces, _CUDA_VSTD::__type_info_ref __self) noexcept
      : __kind_(__kind)
      , __nbr_interfaces_(__nbr_interfaces)
      , __typeid_(&__self)
  {}

  uint8_t __version_                    = __vtable_version;
  __vtable_kind __kind_                 = __vtable_kind::__normal;
  uint16_t __nbr_interfaces_            = 0;
  uint32_t const __cookie_              = 0xDEADBEEF;
  _CUDA_VSTD::__type_info_ptr __typeid_ = nullptr;
};

static_assert(sizeof(__vtable_base) == sizeof(uint64_t) + sizeof(void*));

template <class _Ptr>
struct __tagged_ptr;

template <class _Tp>
struct __tagged_ptr<_Tp*>
{
  _CUDAX_API void __set(_Tp* __pv, bool __flag) noexcept
  {
    __ptr_ = reinterpret_cast<uintptr_t>(__pv) | uintptr_t(__flag);
  }

  _CCCL_NODISCARD _CUDAX_API _Tp* __get() const noexcept
  {
    return reinterpret_cast<_Tp*>(__ptr_ & ~uintptr_t(1));
  }

  _CCCL_NODISCARD _CUDAX_API bool __flag() const noexcept
  {
    return static_cast<bool>(__ptr_ & uintptr_t(1));
  }

  uintptr_t __ptr_ = 0;
};

///
/// _override
///
template <class _Tp, auto _Mbr>
struct __override_;

template <class _Tp, auto _Mbr>
using __override_t = __override_<_Tp, _Mbr>*;

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_GCC("-Wstrict-aliasing")

template <class _Fn, class _Cp>
_CUDAX_API auto __class_of_(_Fn _Cp::*) -> _Cp;

template <class _Mbr>
using __class_of = decltype(__cudax::__class_of_(_Mbr()));

/// We use a C-style cast instead of a static_cast because a C-style cast will
/// ignore accessibility, letting us cast to a private base class.
template <class _DstPtr, class _Src>
_CUDAX_TRIVIAL_API _DstPtr __c_style_cast(_Src* __ptr) noexcept
{
  static_assert(_CUDA_VSTD::is_pointer_v<_DstPtr>, "");
  static_assert(_CUDA_VSTD::is_base_of_v<_CUDA_VSTD::remove_pointer_t<_DstPtr>, _Src>, "invalid C-style cast detected");
  return (_DstPtr) __ptr;
}

template <class _Tp, auto _Fn, class _Ret, bool _IsConst, bool _IsNothrow, class... _Args>
_CCCL_NODISCARD _CUDAX_API _Ret __override_fn_([[maybe_unused]] _CUDA_VSTD::__maybe_const<_IsConst, void>* __pv,
                                               [[maybe_unused]] _Args... __args) noexcept(_IsNothrow)
{
  using __value_type = _CUDA_VSTD::__maybe_const<_IsConst, _Tp>;

  if constexpr (_CUDA_VSTD::is_same_v<_Tp, void>)
  {
    // This instantiation is created only during the computation of the vtable
    // type. It is never actually called.
    _CCCL_ASSERT(false, "should never get here");
    _CUDA_VSTD_NOVERSION::terminate();
  }
  else if constexpr (_CUDA_VSTD::is_member_function_pointer_v<decltype(_Fn)>)
  {
    // _Fn may be a pointer to a member function of a private base of _Tp. So
    // after static_cast-ing to _Tp*, we need to use a C-style cast to get a
    // pointer to the correct base class.
    using __class_type  = _CUDA_VSTD::__maybe_const<_IsConst, __class_of<decltype(_Fn)>>;
    __class_type& __obj = *__cudax::__c_style_cast<__class_type*>(static_cast<__value_type*>(__pv));
    return (__obj.*_Fn)(static_cast<_Args&&>(__args)...);
  }
  else
  {
    __value_type& __obj = *static_cast<__value_type*>(__pv);
    return (*_Fn)(__obj, static_cast<_Args&&>(__args)...);
  }
}

_CCCL_DIAG_POP

template <class...>
struct __undefined;

template <class _Fn, class _Tp = void, auto _Mbr = 0>
extern __undefined<_Fn> __virtual_override_fn;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void*, _Args...)> //
  __virtual_override_fn<_Ret (*)(_Cp&, _Args...), _Tp, _Mbr> = //
  &__override_fn_<_Tp, _Mbr, _Ret, false, false, _Args...>;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void const*, _Args...)>
  __virtual_override_fn<_Ret (*)(_Cp const&, _Args...), _Tp, _Mbr> =
    &__override_fn_<_Tp, _Mbr, _Ret, true, false, _Args...>;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void*, _Args...) noexcept>
  __virtual_override_fn<_Ret (*)(_Cp&, _Args...) noexcept, _Tp, _Mbr> =
    &__override_fn_<_Tp, _Mbr, _Ret, false, true, _Args...>;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void const*, _Args...) noexcept>
  __virtual_override_fn<_Ret (*)(_Cp const&, _Args...) noexcept, _Tp, _Mbr> =
    &__override_fn_<_Tp, _Mbr, _Ret, true, true, _Args...>;

// TODO: Add support for member functions with reference qualifiers.

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void*, _Args...)> //
  __virtual_override_fn<_Ret (_Cp::*)(_Args...), _Tp, _Mbr> = //
  &__override_fn_<_Tp, _Mbr, _Ret, false, false, _Args...>;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void const*, _Args...)> //
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) const, _Tp, _Mbr> =
    &__override_fn_<_Tp, _Mbr, _Ret, true, false, _Args...>;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void*, _Args...) noexcept>
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) noexcept, _Tp, _Mbr> =
    &__override_fn_<_Tp, _Mbr, _Ret, false, true, _Args...>;

template <class _Tp, auto _Mbr, class _Ret, class _Cp, class... _Args>
inline constexpr __identity<_Ret (*)(void const*, _Args...) noexcept>
  __virtual_override_fn<_Ret (_Cp::*)(_Args...) const noexcept, _Tp, _Mbr> =
    &__override_fn_<_Tp, _Mbr, _Ret, true, true, _Args...>;

template <class _Ret, class... _Args>
_CUDAX_API _Ret __get_virtual_result(_Ret (*)(_Args...));

template <class _Ret, class... _Args>
_CUDAX_API _Ret __get_virtual_result(_Ret (*)(_Args...) noexcept) noexcept;

template <class _Ret, class... _Args>
_CUDAX_API _CUDA_VSTD::false_type __is_virtual_const(_Ret (*)(void*, _Args...));

template <class _Ret, class... _Args>
_CUDAX_API _CUDA_VSTD::true_type __is_virtual_const(_Ret (*)(void const*, _Args...));

///
/// __virtual_fn
///
template <auto _Fn>
struct __virtual_fn
{
  using __function_t _CCCL_NODEBUG_ALIAS = decltype(__virtual_override_fn<decltype(_Fn)>);
  using __result_t _CCCL_NODEBUG_ALIAS   = decltype(__get_virtual_result(__function_t{}));

  static constexpr bool __const_fn   = decltype(__is_virtual_const(__function_t{}))::value;
  static constexpr bool __nothrow_fn = noexcept(__get_virtual_result(__function_t{}));

  template <class _Tp, auto _Mbr>
  _CUDAX_API constexpr __virtual_fn(__override_t<_Tp, _Mbr>) noexcept
      : __fn_(__virtual_override_fn<decltype(_Fn), _Tp, _Mbr>)
  {}

  __function_t __fn_;
};

///
/// __ireference
///
/// Note: a `basic_any<__ireference<_Interface>>&&` is an rvalue reference, whereas
/// a `basic_any<_Interface&>&&` is an lvalue reference.
template <class _Interface>
struct __ireference : _Interface
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expected a class type");
  static constexpr size_t __size_      = sizeof(void*);
  static constexpr size_t __align_     = alignof(void*);
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  using interface _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::remove_const_t<_Interface>;
};

///
/// _Interface type traits
///
template <class _Interface>
extern _CUDA_VSTD::remove_const_t<_Interface> __remove_ireference_v;

template <class _Interface>
extern _Interface __remove_ireference_v<__ireference<_Interface>>;

template <class _Interface>
extern _Interface __remove_ireference_v<__ireference<_Interface const>>;

template <class _Interface>
using __remove_ireference_t = decltype(__remove_ireference_v<_Interface>);

template <class _Interface>
inline constexpr bool __is_value_v = _CUDA_VSTD::is_class_v<_Interface>;

template <class _Interface>
inline constexpr bool __is_value_v<__ireference<_Interface>> = false;

template <class _Interface>
inline constexpr bool __is_lvalue_reference_v = false;

template <class _Interface>
inline constexpr bool __is_lvalue_reference_v<__ireference<_Interface const>> = true;

template <class _Interface>
inline constexpr bool __is_lvalue_reference_v<_Interface&> = true;

///
/// __rtti
///
template <class _Tp>
_CUDAX_API void __dtor_fn(void* __pv, bool __small) noexcept;

template <class _Interface, class _Tp = __remove_ireference_t<_Interface>>
using __overrides_for = typename _Interface::template overrides<_Tp>;

template <class _Interface>
using __vtable_for = typename __overrides_for<_Interface>::__vtable;

template <class _Interface>
using __vptr_for = typename __overrides_for<_Interface>::__vptr_t;

///
/// semi-regular overrides
///
template <class _Tp>
_CUDAX_API void __dtor_fn(void* __pv, bool __small) noexcept
{
  __small ? static_cast<_Tp*>(__pv)->~_Tp() //
          : delete *static_cast<_Tp**>(__pv);
}

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Tp>)
_CUDAX_PUBLIC_API void __move_fn(_Tp& __src, void* __dst) noexcept
{
  ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
}

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API bool __try_move_fn(_Tp& __src, void* __dst, size_t __size, size_t __align)
{
  if (__is_small<_Tp>(__size, __align))
  {
    ::new (__dst) _Tp(static_cast<_Tp&&>(__src));
    return true;
  }
  else
  {
    ::new (__dst) __identity<_Tp*>(new _Tp(static_cast<_Tp&&>(__src)));
    return false;
  }
}

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::copyable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API bool __copy_fn(_Tp const& __src, void* __dst, size_t __size, size_t __align)
{
  if (__is_small<_Tp>(__size, __align))
  {
    ::new (__dst) _Tp(__src);
    return true;
  }
  else
  {
    ::new (__dst) __identity<_Tp*>(new _Tp(__src));
    return false;
  }
}

_LIBCUDACXX_TEMPLATE(class _Tp)
_LIBCUDACXX_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
_CCCL_NODISCARD _CUDAX_PUBLIC_API bool
__equal_fn(_Tp const& __self, _CUDA_VSTD::__type_info_ref __type, void const* __other)
{
  if (_CCCL_TYPEID(_Tp) == __type)
  {
    return __self == *static_cast<_Tp const*>(__other);
  }
  return false;
}

template <class _Interface, class _Fn>
using __bases_of = _CUDA_VSTD::__type_call<
  _CUDA_VSTD::__type_concat<_CUDA_VSTD::__type_list<iunknown, _CUDA_VSTD::remove_const_t<_Interface>>,
                            typename _Interface::template __ibases<_CUDA_VSTD::__type_quote<_CUDA_VSTD::__type_list>>>,
  _Fn>;

///
/// `basic_any_from`
///
/// @brief This function is for use in the thunks in an interface to get
/// a pointer or a reference to the full `basic_any` object.
///
template <template <class...> class _Interface, class _Super>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API auto basic_any_from(_Interface<_Super>&& __self) noexcept -> basic_any<_Super>&&
{
  return static_cast<basic_any<_Super>&&>(__self);
}

template <template <class...> class _Interface, class _Super>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API auto basic_any_from(_Interface<_Super>& __self) noexcept -> basic_any<_Super>&
{
  return static_cast<basic_any<_Super>&>(__self);
}

template <template <class...> class _Interface, class _Super>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API auto
basic_any_from(_Interface<_Super> const& __self) noexcept -> basic_any<_Super> const&
{
  return static_cast<basic_any<_Super> const&>(__self);
}

template <template <class...> class _Interface>
_CCCL_NODISCARD _CUDAX_API auto basic_any_from(_Interface<> const&) noexcept -> basic_any<_Interface<>> const&
{
  // This overload is selected when called from the thunk of an unspecialized
  // interface; e.g., `icat<>` rather than `icat<ialley_cat<>>`. The thunks of
  // unspecialized interfaces are never called, they just need to exist.
  _CCCL_ASSERT(false, "should never get here");
  _CUDA_VSTD_NOVERSION::terminate();
}

template <template <class...> class _Interface, class _Super>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API auto basic_any_from(_Interface<_Super>* __self) noexcept -> basic_any<_Super>*
{
  return static_cast<basic_any<_Super>*>(__self);
}

template <template <class...> class _Interface, class _Super>
_CCCL_NODISCARD _CUDAX_TRIVIAL_API auto
basic_any_from(_Interface<_Super> const* __self) noexcept -> basic_any<_Super> const*
{
  return static_cast<basic_any<_Super> const*>(__self);
}

template <template <class...> class _Interface>
_CCCL_NODISCARD _CUDAX_API auto basic_any_from(_Interface<> const*) noexcept -> basic_any<_Interface<>> const*
{
  // See comment above about the use of `basic_any_from` in the thunks of
  // unspecialized interfaces.
  _CCCL_ASSERT(false, "should never get here");
  _CUDA_VSTD_NOVERSION::terminate();
}

template <class _Interface>
_CUDAX_API auto __is_basic_any_test(basic_any<_Interface>&&) -> basic_any<_Interface>&&;
template <class _Interface>
_CUDAX_API auto __is_basic_any_test(basic_any<_Interface>&) -> basic_any<_Interface>&;
template <class _Interface>
_CUDAX_API auto __is_basic_any_test(basic_any<_Interface> const&) -> basic_any<_Interface> const&;

// clang-format off
template <class _Tp>
_LIBCUDACXX_CONCEPT __is_basic_any =
  _LIBCUDACXX_REQUIRES_EXPR((_Tp), _Tp& __value)
  (
    __is_basic_any_test(__value)
  );
// clang-format on

///
/// set subsumption
///
template <class _I1, class _I2>
inline constexpr bool __subsumes = false;

template <class _Interface>
inline constexpr bool __subsumes<_Interface, _Interface> = true;

template <class... _Set>
inline constexpr bool __subsumes<__iset<_Set...>, __iset<_Set...>> = true;

template <class... _Subset, class... _Superset>
inline constexpr bool __subsumes<__iset<_Subset...>, __iset<_Superset...>> =
  _CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Superset...>, _Subset...>;

///
/// RTTI
///
///
/// bad_any_cast
///
struct bad_any_cast : ::std::bad_cast
{
  _CUDAX_API bad_any_cast() noexcept                               = default;
  _CUDAX_API bad_any_cast(bad_any_cast const&) noexcept            = default;
  _CUDAX_API ~bad_any_cast() noexcept override                     = default;
  _CUDAX_API bad_any_cast& operator=(bad_any_cast const&) noexcept = default;

  //_CUDAX_API
  char const* what() const noexcept override
  {
    return "cannot cast value to target type";
  }
};

_CCCL_NORETURN _CUDAX_API inline void __throw_bad_any_cast()
{
#ifndef _CCCL_NO_EXCEPTIONS
  NV_IF_ELSE_TARGET(NV_IS_HOST, (throw bad_any_cast();), (_CUDA_VSTD_NOVERSION::terminate();))
#else // ^^^ !_CCCL_NO_EXCEPTIONS ^^^ / vvv _CCCL_NO_EXCEPTIONS vvv
  _CUDA_VSTD_NOVERSION::terminate();
#endif // _CCCL_NO_EXCEPTIONS
}

struct __base_vptr
{
  _CUDAX_API __base_vptr() = default;

  _CUDAX_TRIVIAL_API constexpr __base_vptr(__vtable_base const* __vptr) noexcept
      : __vptr_(__vptr)
  {}

  template <class _VTable>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API explicit constexpr operator _VTable const*() const noexcept
  {
    _CCCL_ASSERT(_CCCL_TYPEID(_VTable) == *__vptr_->__typeid_, "bad vtable cast detected");
    return static_cast<_VTable const*>(__vptr_);
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API explicit constexpr operator bool() const noexcept
  {
    return __vptr_ != nullptr;
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr __vtable_base const* operator->() const noexcept
  {
    return __vptr_;
  }

#if defined(__cpp_lib_three_way_comparison)
  _CUDAX_API bool operator==(__base_vptr const& __other) const noexcept = default;
#else
  _CCCL_NODISCARD_FRIEND _CUDAX_API constexpr bool operator==(__base_vptr __lhs, __base_vptr __rhs) noexcept
  {
    return __lhs.__vptr_ == __rhs.__vptr_;
  }

  _CCCL_NODISCARD_FRIEND _CUDAX_API constexpr bool operator!=(__base_vptr __lhs, __base_vptr __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif

  __vtable_base const* __vptr_{};
};

struct __base_info
{
  _CUDA_VSTD::__type_info_ptr __typeid_;
  __base_vptr __vptr_;
};

inline constexpr size_t __half_size_t_bits = sizeof(size_t) * CHAR_BIT / 2;

// The metadata for the type-erased object. All vtables have an rtti sub-object,
// which contains a sub-object of this type.
struct __object_metadata
{
  size_t __size_ : __half_size_t_bits;
  size_t __align_ : __half_size_t_bits;
  _CUDA_VSTD::__type_info_ptr __object_typeid_;
  _CUDA_VSTD::__type_info_ptr __pointer_typeid_;
  _CUDA_VSTD::__type_info_ptr __const_pointer_typeid_;
};

template <class _Tp>
inline constexpr __object_metadata __object_metadata_v = {
  sizeof(_Tp), alignof(_Tp), &_CCCL_TYPEID(_Tp), &_CCCL_TYPEID(_Tp*), &_CCCL_TYPEID(_Tp const*)};

// All vtables have an rtti sub-object. This object has several responsibilities:
// * It contains the destructor for the type-erased object.
// * It contains the metadata for the type-erased object.
// * It contains a map from the base interfaces typeids to their vtables for use
//   in dynamic_cast-like functionality.
struct __rtti : __vtable_base
{
  template <class _Tp, class _Super, class... _Interfaces>
  _CUDAX_TRIVIAL_API constexpr __rtti(
    __tag<_Tp, _Super>, __tag<_Interfaces...>, __base_info const* __base_vptr_map) noexcept
      : __vtable_base{__vtable_kind::__rtti, sizeof...(_Interfaces), _CCCL_TYPEID(__rtti)}
      , __dtor_(&__dtor_fn<_Tp>)
      , __object_info_(&__object_metadata_v<_Tp>)
      , __interface_typeid_{&_CCCL_TYPEID(_Super)}
      , __base_vptr_map_{__base_vptr_map}
  {}

  template <class... _Interfaces>
  _CCCL_NODISCARD _CUDAX_API __vptr_for<__iset<_Interfaces...>> __query_interface(__iset<_Interfaces...>) const noexcept
  {
    // TODO: find a way to check at runtime that the requested __iset is a subset
    // of the interfaces in the vtable.
    return static_cast<__vptr_for<__iset<_Interfaces...>>>(this);
  }

  // Sequentially search the base_vptr_map for the requested interface by
  // comparing typeids. If the requested interface is found, return a pointer to
  // its vtable; otherwise, return nullptr.
  template <class _Interface>
  _CCCL_NODISCARD _CUDAX_API __vptr_for<_Interface> __query_interface(_Interface) const noexcept
  {
    // On sane implementations, comparing type_info objects first compares their
    // addresses and, if that fails, it does a string comparison. What we want is
    // to check _all_ the addresses first, and only if they all fail, resort to
    // string comparisons. So do two passes over the __base_vptr_map.
    constexpr _CUDA_VSTD::__type_info_ref __id = _CCCL_TYPEID(_Interface);

    for (size_t __i = 0; __i < __nbr_interfaces_; ++__i)
    {
      if (&__id == __base_vptr_map_[__i].__typeid_)
      {
        return static_cast<__vptr_for<_Interface>>(__base_vptr_map_[__i].__vptr_);
      }
    }

    for (size_t __i = 0; __i < __nbr_interfaces_; ++__i)
    {
      if (__id == *__base_vptr_map_[__i].__typeid_)
      {
        return static_cast<__vptr_for<_Interface>>(__base_vptr_map_[__i].__vptr_);
      }
    }

    return nullptr;
  }

  void (*__dtor_)(void*, bool) noexcept;
  __object_metadata const* __object_info_;
  _CUDA_VSTD::__type_info_ptr __interface_typeid_ = nullptr;
  __base_info const* __base_vptr_map_;
};

template <size_t _NbrInterfaces>
struct __rtti_ex : __rtti
{
  template <class _Tp, class _Super, class... _Interfaces, class _VPtr>
  _CUDAX_API constexpr __rtti_ex(__tag<_Tp, _Super> __type, __tag<_Interfaces...> __ibases, _VPtr __self) noexcept
      : __rtti{__type, __ibases, __base_vptr_array}
      , __base_vptr_array{{&_CCCL_TYPEID(_Interfaces), static_cast<__vptr_for<_Interfaces>>(__self)}...}
  {}

  __base_info __base_vptr_array[_NbrInterfaces];
};

///
/// overrides_for
///
template <class _InterfaceOrModel, auto... _Mbrs>
struct overrides_for
{
  static_assert(!_CUDA_VSTD::is_const_v<_InterfaceOrModel>, "expected a class type");
  using __vtable _CCCL_NODEBUG_ALIAS = __basic_vtable<_InterfaceOrModel, _Mbrs...>;
  using __vptr_t _CCCL_NODEBUG_ALIAS = __vtable const*;
};

template <class... _Interfaces>
struct overrides_for<__iset<_Interfaces...>>
{
  using __vtable _CCCL_NODEBUG_ALIAS = __basic_vtable<__iset<_Interfaces...>>;
  using __vptr_t _CCCL_NODEBUG_ALIAS = __iset_vptr<_Interfaces...>;
};

template <>
struct overrides_for<iunknown>
{
  using __vtable _CCCL_NODEBUG_ALIAS = detail::__ignore; // no vtable, rtti is added explicitly in __vtable_tuple
  using __vptr_t _CCCL_NODEBUG_ALIAS = __rtti const*;
};

///
/// interface
///
template <template <class...> class _Interface, class... _Bases, size_t Size, size_t Align>
struct interface<_Interface, extends<_Bases...>, Size, Align>
{
  static constexpr size_t size  = (_CUDA_VSTD::max)({Size, _Bases::size...});
  static constexpr size_t align = (_CUDA_VSTD::max)({Align, _Bases::align...});

  template <class... _Super>
  using __rebind _CCCL_NODEBUG_ALIAS = _Interface<_Super...>;

  template <class _Fn>
  using __ibases _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call<
    _CUDA_VSTD::__type_concat<__bases_of<_Bases, _CUDA_VSTD::__type_quote<_CUDA_VSTD::__type_list>>...>,
    _Fn>;

  template <class _Tp>
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp>;
};

template <template <class...> class _Interface, class Extends, size_t Size, size_t Align>
_CUDAX_API auto __is_interface_test(interface<_Interface, Extends, Size, Align> const&) -> void;

// clang-format off
template <class _Tp>
_LIBCUDACXX_CONCEPT __is_interface =
  _LIBCUDACXX_REQUIRES_EXPR((_Tp), _Tp& __value)
  (
    __is_interface_test(__value)
  );
// clang-format on

///
/// __iunknown
///
struct iunknown : interface<_CUDA_VSTD::__type_always<iunknown>::__call>
{};

///
/// __iset
///
template <class... _Interfaces>
struct __iset_
{
  template <class...>
  struct __interface_ : interface<__interface_, extends<_Interfaces...>>
  {};
};

template <class... _Interfaces>
struct __iset : __iset_<_Interfaces...>::template __interface_<>
{};

// flatten any nested sets
template <class _Interface>
using __iset_flatten = _CUDA_VSTD::__as_type_list<
  _CUDA_VSTD::
    conditional_t<detail::__is_specialization_of<_Interface, __iset>, _Interface, _CUDA_VSTD::__type_list<_Interface>>>;

// flatten all sets into one, remove duplicates, and sort the elements.
// TODO: sort!
// template <class... _Interfaces>
// using iset _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_call<
//   _CUDA_VSTD::__type_unique<_CUDA_VSTD::__type_sort<_CUDA_VSTD::__type_concat<__iset_flatten<_Interfaces>...>>>,
//   _CUDA_VSTD::__type_quote<__iset>>;
template <class... _Interfaces>
using iset =
  _CUDA_VSTD::__type_call<_CUDA_VSTD::__type_unique<_CUDA_VSTD::__type_concat<__iset_flatten<_Interfaces>...>>,
                          _CUDA_VSTD::__type_quote<__iset>>;

///
/// Virtual table pointers
///

template <class... _Interfaces>
struct __iset_vptr : __base_vptr
{
  using __iset_vtable _CCCL_NODEBUG_ALIAS = __vtable_for<__iset<_Interfaces...>>;

  _CUDAX_API __iset_vptr() = default;

  _CUDAX_API constexpr __iset_vptr(__iset_vtable const* __vptr) noexcept
      : __base_vptr(__vptr)
  {}

  _CUDAX_API constexpr __iset_vptr(__base_vptr __vptr) noexcept
      : __base_vptr(__vptr)
  {}

  // Permit narrowing conversions from a super-set __vptr. Warning: we can't
  // simply constrain this because then the ctor from __base_vptr would be
  // selected instead, giving the wrong result.
  template <class... _Others>
  _CUDAX_API __iset_vptr(__iset_vptr<_Others...> __vptr) noexcept
      : __base_vptr(__vptr->__query_interface(iunknown()))
  {
    static_assert(_CUDA_VSTD::__type_set_contains_v<_CUDA_VSTD::__make_type_set<_Others...>, _Interfaces...>, "");
    _CCCL_ASSERT(__vptr_->__kind_ == __vtable_kind::__rtti && __vptr_->__cookie_ == 0xDEADBEEF,
                 "query_interface returned a bad pointer to the iunknown vtable");
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr __iset_vptr const* operator->() const noexcept
  {
    return this;
  }

  template <class _Interface>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr __vptr_for<_Interface> __query_interface(_Interface) const noexcept
  {
    if (__vptr_->__kind_ == __vtable_kind::__normal)
    {
      return static_cast<__iset_vtable const*>(__vptr_)->__query_interface(_Interface{});
    }
    else
    {
      return static_cast<__rtti const*>(__vptr_)->__query_interface(_Interface{});
    }
  }
};

template <class... _Interfaces>
struct __tagged_ptr<__iset_vptr<_Interfaces...>>
{
  _CUDAX_TRIVIAL_API void __set(__iset_vptr<_Interfaces...> __vptr, bool __flag) noexcept
  {
    __ptr_ = reinterpret_cast<uintptr_t>(__vptr.__vptr_) | uintptr_t(__flag);
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API __iset_vptr<_Interfaces...> __get() const noexcept
  {
    return __iset_vptr<_Interfaces...>{reinterpret_cast<__vtable_base const*>(__ptr_ & ~uintptr_t(1))};
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API bool __flag() const noexcept
  {
    return static_cast<bool>(__ptr_ & uintptr_t(1));
  }

  uintptr_t __ptr_ = 0;
};

///
/// extension_of
///
template <class _Base>
struct __has_base_fn
{
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::bool_constant<_CUDA_VSTD::__is_included_in_v<_Base, _Interfaces...>>;
};

template <class... _Bases>
struct __has_base_fn<__iset<_Bases...>>
{
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::bool_constant<(__subsumes<__iset<_Bases...>, _Interfaces> || ...)>;
};

template <class _Derived, class _Base, class = void>
inline constexpr bool __extension_of = false;

template <class _Derived, class _Base>
inline constexpr bool
  __extension_of<_Derived,
                 _Base,
                 _CUDA_VSTD::enable_if_t<_CUDA_VSTD::is_class_v<_Derived> && _CUDA_VSTD::is_class_v<_Base>>> =
    __bases_of<_Derived, __has_base_fn<_CUDA_VSTD::remove_const_t<_Base>>>::value;

template <class _Derived, class _Base>
_LIBCUDACXX_CONCEPT extension_of = __extension_of<_Derived, _Base>;

///
/// __unique_interfaces
///
/// Given an interface, return a list that contains the interface and all its
/// bases, but with duplicates removed.
///
template <class _Interface, class _Fn = _CUDA_VSTD::__type_quote<_CUDA_VSTD::__type_list>>
using __unique_interfaces =
  _CUDA_VSTD::__type_apply<_Fn, __bases_of<_Interface, _CUDA_VSTD::__type_quote<_CUDA_VSTD::__make_type_set>>>;

///
/// __index_of: find the index of an interface in a list of unique interfaces
///
_CCCL_NODISCARD _CUDAX_API constexpr size_t __find_first(_CUDA_VSTD::initializer_list<bool> __il)
{
  auto __it = __il.begin();
  while (__it != __il.end() && !*__it)
  {
    ++__it;
  }
  return static_cast<size_t>(__it - __il.begin());
}

template <class _Interface>
struct __find_index_of
{
  template <class... _Interfaces>
  static constexpr size_t __index = __find_first({__subsumes<_Interface, _Interfaces>...});

  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::integral_constant<size_t, __index<_Interfaces...>>;
};

template <class _Interface, class _Super>
using __index_of = _CUDA_VSTD::__type_apply<__find_index_of<_Interface>, __unique_interfaces<_Super>>;

///
/// __basic_vtable
///
template <class _Interface, auto... _Mbrs>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES __basic_vtable
    : __vtable_base
    , __virtual_fn<_Mbrs>...
{
  using interface _CCCL_NODEBUG_ALIAS = _Interface;
  static constexpr size_t __cbases    = _CUDA_VSTD::__type_list_size<__unique_interfaces<interface>>::value;

  template <class _VPtr, class _Tp, auto... _OtherMembers, class... _Interfaces>
  _CUDAX_API constexpr __basic_vtable(_VPtr __vptr, overrides_for<_Tp, _OtherMembers...>, __tag<_Interfaces...>) noexcept
      : __vtable_base{__vtable_kind::__normal, __cbases, _CCCL_TYPEID(__basic_vtable)}
      , __virtual_fn<_Mbrs>{__override_t<_Tp, _OtherMembers>{}}...
      , __vptr_map_{__base_vptr{__vptr->__query_interface(_Interfaces())}...}
  {}

  template <class _Tp, class _VPtr>
  _CUDAX_API constexpr __basic_vtable(__tag<_Tp>, _VPtr __vptr) noexcept
      : __basic_vtable{
          __vptr, __overrides_for<interface, _Tp>(), __unique_interfaces<interface, _CUDA_VSTD::__type_quote<__tag>>()}
  {}

  _CCCL_NODISCARD _CUDAX_API __vptr_for<interface> __query_interface(interface) const noexcept
  {
    return this;
  }

  template <class... _Others>
  _CCCL_NODISCARD _CUDAX_API __vptr_for<__iset<_Others...>> __query_interface(__iset<_Others...>) const noexcept
  {
    using __remainder =
      _CUDA_VSTD::__type_list_size<_CUDA_VSTD::__type_find<__unique_interfaces<interface>, __iset<_Others...>>>;
    constexpr size_t __index = __cbases - __remainder::value;
    if constexpr (__index < __cbases)
    {
      // `_Interface` extends __iset<_Others...> exactly. We can return an actual
      // vtable pointer.
      return static_cast<__vtable_for<__iset<_Others...>> const*>(__vptr_map_[__index]);
    }
    else
    {
      // Otherwise, we have to return a subset vtable pointer, which does
      // dynamic interface lookup.
      return static_cast<__vptr_for<__iset<_Others...>>>(__query_interface(iunknown()));
    }
  }

  template <class _Other>
  _CCCL_NODISCARD _CUDAX_API __vptr_for<_Other> __query_interface(_Other) const noexcept
  {
    constexpr size_t __index = __index_of<_Other, interface>::value;
    static_assert(__index < __cbases);
    return static_cast<__vptr_for<_Other>>(__vptr_map_[__index]);
  }

  __base_vptr __vptr_map_[__cbases];
};

///
/// __try_vptr_cast
///
/// This function ignores const qualification on the source and destination
/// interfaces.
///
template <class _SrcInterface, class _DstInterface>
_CCCL_NODISCARD _CUDAX_API auto
__try_vptr_cast(__vptr_for<_SrcInterface> __src_vptr) noexcept -> __vptr_for<_DstInterface>
{
  static_assert(_CUDA_VSTD::is_class_v<_SrcInterface> && _CUDA_VSTD::is_class_v<_DstInterface>, "expected class types");
  if (__src_vptr == nullptr)
  {
    return nullptr;
  }
  else if constexpr (_CUDA_VSTD::is_same_v<_SrcInterface const, _DstInterface const>)
  {
    return __src_vptr;
  }
  else if constexpr (extension_of<_SrcInterface, _DstInterface>)
  {
    /// Fast up-casts:
    return __src_vptr->__query_interface(_DstInterface());
  }
  else
  {
    /// Slow down-casts and cross-casts:
    __rtti const* rtti = __src_vptr->__query_interface(iunknown());
    return rtti->__query_interface(_DstInterface());
  }
}

template <class _SrcInterface, class _DstInterface>
_CCCL_NODISCARD _CUDAX_API auto __vptr_cast(__vptr_for<_SrcInterface> __src_vptr) //
  noexcept(_CUDA_VSTD::is_same_v<_SrcInterface, _DstInterface>) //
  -> __vptr_for<_DstInterface>
{
  if constexpr (_CUDA_VSTD::is_same_v<_SrcInterface, _DstInterface>)
  {
    return __src_vptr;
  }
  else
  {
    auto __dst_vptr = __try_vptr_cast<_SrcInterface, _DstInterface>(__src_vptr);
    if (!__dst_vptr && __src_vptr)
    {
      __throw_bad_any_cast();
    }
    return __dst_vptr;
  }
}

///
/// conversions
///
/// Can one basic_any type convert to another? Implicitly? Explicitly?
/// Statically? Dynamically? We answer these questions by mapping two
/// cvref qualified basic_any types to archetype types, and then using
/// the built-in language rules to determine if the conversion is valid.
///
struct __immovable_archetype
{
  _CUDAX_API __immovable_archetype()                        = default;
  _CUDAX_API __immovable_archetype(__immovable_archetype&&) = delete;

  template <class _Value>
  _CUDAX_API __immovable_archetype(_Value) noexcept;
  template <class _Value>
  _CUDAX_API __immovable_archetype(_Value*) = delete;
};

struct __movable_archetype : __immovable_archetype
{
  _CUDAX_API __movable_archetype() = default;
  _CUDAX_API __movable_archetype(__movable_archetype&&) noexcept;
};

struct __copyable_archetype : __movable_archetype
{
  _CUDAX_API __copyable_archetype() = default;
  _CUDAX_API __copyable_archetype(__copyable_archetype const&);
};

template <class _Interface>
using _archetype_base = _CUDA_VSTD::conditional_t<
  extension_of<_Interface, icopyable<>>,
  __copyable_archetype,
  _CUDA_VSTD::conditional_t<extension_of<_Interface, imovable<>>, __movable_archetype, __immovable_archetype>>;

template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<_Interface>&&) -> _Interface;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<__ireference<_Interface>>&&) -> _Interface;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<_Interface>&) -> _Interface&;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<_Interface> const&) -> _Interface const&;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<_Interface>*) -> _Interface*;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<_Interface> const*) -> _Interface const*;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<__ireference<_Interface>>*) -> _Interface*;
template <class _Interface>
_CUDAX_API auto __interface_from(basic_any<__ireference<_Interface>> const*) -> _Interface*;

template <class _Interface>
_CUDAX_API auto __as_archetype(_Interface&&) -> _archetype_base<_Interface>;
template <class _Interface>
_CUDAX_API auto __as_archetype(_Interface&) -> _archetype_base<_Interface>&;
template <class _Interface>
_CUDAX_API auto __as_archetype(_Interface const&) -> _archetype_base<_Interface> const&;
template <class _Interface>
_CUDAX_API auto __as_archetype(_Interface*) -> _archetype_base<_Interface>*;
template <class _Interface>
_CUDAX_API auto __as_archetype(_Interface const*) -> _archetype_base<_Interface> const*;
template <class _Interface>
_CUDAX_API auto __as_archetype(__ireference<_Interface>) -> _archetype_base<_Interface>&;
template <class _Interface>
_CUDAX_API auto __as_archetype(__ireference<_Interface const>) -> _archetype_base<_Interface> const&;

template <class _Interface>
_CUDAX_API auto __as_immovable(_Interface&&) -> __immovable_archetype;
template <class _Interface>
_CUDAX_API auto __as_immovable(_Interface&) -> __immovable_archetype&;
template <class _Interface>
_CUDAX_API auto __as_immovable(_Interface const&) -> __immovable_archetype const&;
template <class _Interface>
_CUDAX_API auto __as_immovable(_Interface*) -> __immovable_archetype*;
template <class _Interface>
_CUDAX_API auto __as_immovable(_Interface const*) -> __immovable_archetype const*;

template <class CvAny>
using __normalized_interface_of = decltype(__interface_from(_CUDA_VSTD::declval<CvAny>()));

template <class CvAny>
using __src_archetype_of = decltype(__as_archetype(__interface_from(_CUDA_VSTD::declval<CvAny>())));

template <class CvAny>
using __dst_archetype_of = decltype(__as_immovable(__as_archetype(__interface_from(_CUDA_VSTD::declval<CvAny>()))));

// If the archetypes are implicitly convertible, then it is possible to
// dynamically cast from the source to the destination. The cast may fail,
// but at least it is possible.
template <class _SrcCvAny, class _DstCvAny>
_LIBCUDACXX_CONCEPT __any_castable_to =
  _CUDA_VSTD::is_convertible_v<__src_archetype_of<_SrcCvAny>, __dst_archetype_of<_DstCvAny>>;

// If the archetypes are implicitly convertible **and** the source interface
// is an extension of the destination one, then it is possible to implicitly
// convert from the source to the destination.
template <class _SrcCvAny, class _DstCvAny>
_LIBCUDACXX_CONCEPT __any_convertible_to =
  __any_castable_to<_SrcCvAny, _DstCvAny> && //
  extension_of<typename _CUDA_VSTD::remove_reference_t<_SrcCvAny>::interface_type,
               typename _CUDA_VSTD::remove_reference_t<_DstCvAny>::interface_type>;

///
/// basic_any_access
///
struct __basic_any_access
{
  template <class _Interface>
  _CUDAX_TRIVIAL_API static auto __make() noexcept -> basic_any<_Interface>
  {
    return basic_any<_Interface>{};
  }

  _LIBCUDACXX_TEMPLATE(class _SrcCvAny, class _DstInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<_SrcCvAny, basic_any<_DstInterface>>)
  _CUDAX_TRIVIAL_API static void __cast_to(_SrcCvAny&& __from, basic_any<_DstInterface>& __to) //
    noexcept(noexcept(__to.__convert_from(static_cast<_SrcCvAny&&>(__from))))
  {
    static_assert(detail::__is_specialization_of<_CUDA_VSTD::remove_cvref_t<_SrcCvAny>, basic_any>);
    __to.__convert_from(static_cast<_SrcCvAny&&>(__from));
  }

  _LIBCUDACXX_TEMPLATE(class _SrcCvAny, class _DstInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<_SrcCvAny*, basic_any<_DstInterface>>)
  _CUDAX_TRIVIAL_API static void __cast_to(_SrcCvAny* __from, basic_any<_DstInterface>& __to) //
    noexcept(noexcept(__to.__convert_from(__from)))
  {
    static_assert(detail::__is_specialization_of<_CUDA_VSTD::remove_const_t<_SrcCvAny>, basic_any>);
    __to.__convert_from(__from);
  }

  template <class _Interface>
  _CUDAX_TRIVIAL_API static auto __get_vptr(basic_any<_Interface> const& __self) noexcept -> __vptr_for<_Interface>
  {
    return __self.__get_vptr();
  }

  template <class _Interface>
  _CUDAX_TRIVIAL_API static void* __get_optr(basic_any<_Interface>& __self) noexcept
  {
    return __self.__get_optr();
  }

  template <class _Interface>
  _CUDAX_TRIVIAL_API static void const* __get_optr(basic_any<_Interface> const& __self) noexcept
  {
    return __self.__get_optr();
  }
};

///
/// __virtuals_map
///

/// The virtuals map is an extra convenience for interface authors. To make a
/// virtual function call, the user must provide the member function pointer
/// corresponding to the virtual, as in:
///
/// @code
/// template <class...>
/// struct ifoo {
///   void meow(auto... __args) {
///     // dispatch to the &ifoo<>::meow virtual function
///     // NB: the `<>` after `ifoo` is significant!
///     virtcall<&ifoo<>::meow>(this, __args...);
///     //            ^^
///   }
///  ...
/// };
/// @endcode
///
/// When taking the address of the member, it is very easy to forget the `<>`
/// after the interface name, which would result in a compilation error --
/// except for the virtuals map, which substitutes the correct member function
/// pointer for the user so they don't have to think about it.
template <auto _Mbr, auto _BoundMbr>
struct __virtuals_map_pair
{
  // map ifoo<>::meow to itself
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr auto operator()(__ctag<_Mbr>) const noexcept
  {
    return _Mbr;
  }

  // map ifoo<_Super>::meow to ifoo<>::meow
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API constexpr auto operator()(__ctag<_BoundMbr>) const noexcept
  {
    return _Mbr;
  }
};

template <class, class>
struct __virtuals_map;

template <class _Interface, auto... _Mbrs, class _BoundInterface, auto... _BoundMbrs>
struct __virtuals_map<overrides_for<_Interface, _Mbrs...>, overrides_for<_BoundInterface, _BoundMbrs...>>
    : __virtuals_map_pair<_Mbrs, _BoundMbrs>...
{
  using __virtuals_map_pair<_Mbrs, _BoundMbrs>::operator()...;
};

template <class _Interface, class _Super>
using __virtuals_map_for =
  __virtuals_map<__overrides_for<_Interface>, __overrides_for<__rebind_interface<_Interface, _Super>>>;

///
/// virtcall
///

// If the interface is __ireference<MyInterface const>, then calls to non-const
// member functions are not allowed.
template <auto, class... _Interface>
inline constexpr bool __valid_virtcall = sizeof...(_Interface) == 1;

template <auto _Mbr, class _Interface>
inline constexpr bool __valid_virtcall<_Mbr, __ireference<_Interface const>> = __virtual_fn<_Mbr>::__const_fn;

template <auto _Mbr, class _Interface, class _Super, class _Self, class... _Args>
_CUDAX_API auto __virtcall(_Self* __self, _Args&&... __args) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  auto* __vptr = __basic_any_access::__get_vptr(*__self)->__query_interface(_Interface());
  auto* __obj  = __basic_any_access::__get_optr(*__self);
  // map the member function pointer to the correct one if necessary
  constexpr auto _Mbr2 = __virtuals_map_for<_Interface, _Super>{}(__ctag<_Mbr>());
  return __vptr->__virtual_fn<_Mbr2>::__fn_(__obj, static_cast<_Args&&>(__args)...);
}

_LIBCUDACXX_TEMPLATE(auto _Mbr, template <class...> class _Interface, class _Super, class... _Args)
_LIBCUDACXX_REQUIRES(__valid_virtcall<_Mbr, _Super>)
_CUDAX_TRIVIAL_API auto virtcall(_Interface<_Super>* __self, _Args&&... __args) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  return __cudax::__virtcall<_Mbr, _Interface<>, _Super>(
    __cudax::basic_any_from(__self), static_cast<_Args&&>(__args)...);
}

_LIBCUDACXX_TEMPLATE(auto _Mbr, template <class...> class _Interface, class _Super, class... _Args)
_LIBCUDACXX_REQUIRES(__valid_virtcall<_Mbr, _Super>)
_CUDAX_TRIVIAL_API auto virtcall(_Interface<_Super> const* __self, _Args&&... __args) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  return __cudax::__virtcall<_Mbr, _Interface<>, _Super>(
    __cudax::basic_any_from(__self), static_cast<_Args&&>(__args)...);
}

_LIBCUDACXX_TEMPLATE(auto _Mbr, template <class...> class _Interface, class... _Super, class... _Args)
_LIBCUDACXX_REQUIRES((!__valid_virtcall<_Mbr, _Super...>) )
_CUDAX_TRIVIAL_API auto virtcall(_Interface<_Super...> const*, _Args&&...) //
  noexcept(__virtual_fn<_Mbr>::__nothrow_fn) //
  -> typename __virtual_fn<_Mbr>::__result_t
{
  constexpr bool __const_correct_virtcall = __valid_virtcall<_Mbr, _Super...> || sizeof...(_Super) == 0;
  // If this static assert fires, then you have called a non-const member
  // function on a `basic_any<I const&>`. This would violate const-correctness.
  static_assert(__const_correct_virtcall, "This function call is not const correct.");
  // This overload can also be selected when called from the thunks of
  // unspecialized interfaces. Those thunks should never be called, but they
  // must exist to satisfy the compiler.
  _CCCL_ASSERT(false, "should never get here");
  _CUDA_VSTD_NOVERSION::terminate();
}

///
/// semi-regular interfaces
///
template <class...>
struct imovable : interface<imovable>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::movable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS =
    overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__try_move_fn<_Tp>), _CUDAX_FNPTR_CONSTANT_WAR(&__move_fn<_Tp>)>;

  _CUDAX_API void __move_to(void* __pv) noexcept
  {
    return __cudax::virtcall<&__move_fn<imovable>>(this, __pv);
  }

  _CCCL_NODISCARD _CUDAX_API bool __move_to(void* __pv, size_t __size, size_t __align)
  {
    return __cudax::virtcall<&__try_move_fn<imovable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct icopyable : interface<icopyable, extends<imovable<>>>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::copyable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__copy_fn<_Tp>)>;

  _CCCL_NODISCARD _CUDAX_API bool __copy_to(void* __pv, size_t __size, size_t __align) const
  {
    return virtcall<&__copy_fn<icopyable>>(this, __pv, __size, __align);
  }
};

template <class...>
struct iequality_comparable : interface<iequality_comparable>
{
  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::equality_comparable<_Tp>)
  using overrides _CCCL_NODEBUG_ALIAS = overrides_for<_Tp, _CUDAX_FNPTR_CONSTANT_WAR(&__equal_fn<_Tp>)>;

#ifndef _LIBCUDACXX_HAS_NO_SPACESHIP_OPERATOR
  _CCCL_NODISCARD _CUDAX_API bool operator==(iequality_comparable const& __other) const
  {
    auto const& __other = __cudax::basic_any_from(__other);
    void const* __obj   = __basic_any_access::__get_optr(__other);
    return __cudax::virtcall<&__equal_fn<iequality_comparable>>(this, __other.type(), __obj);
  }
#else
  _CCCL_NODISCARD_FRIEND _CUDAX_API bool
  operator==(iequality_comparable const& __left, iequality_comparable const& __right)
  {
    auto const& __rhs = __cudax::basic_any_from(__right);
    void const* __obj = __basic_any_access::__get_optr(__rhs);
    return __cudax::virtcall<&__equal_fn<iequality_comparable>>(&__left, __rhs.type(), __obj);
  }

  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_API bool
  operator!=(iequality_comparable const& __left, iequality_comparable const& __right)
  {
    return !(__left == __right);
  }
#endif
};

///
/// interface satisfaction
///
template <class...>
struct __iempty : interface<__iempty>
{};

template <class _Tp>
struct __satisfaction_fn
{
  template <class _Interface>
  using __does_not_satisfy _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::_Not<_CUDA_VSTD::_IsValidExpansion<__overrides_for, _Interface, _Tp>>;

  // Try to find an unsatisfied interface. If we find one, we return it (it's at
  // the front of the list returned from __type_find_if). If we don't find one
  // (that is, if the returned list is empty), we return __iempty<>.
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = _CUDA_VSTD::__type_front< // take the front of the list
    _CUDA_VSTD::__type_push_back< // add __iempty<> to the end of the list
      _CUDA_VSTD::__type_find_if< // find the first unsatisfied interface if any, returns a list
        _CUDA_VSTD::__type_list<_Interfaces...>,
        _CUDA_VSTD::__type_quote1<__does_not_satisfy>>,
      __iempty<>>>;
};

template <class _Interface, class _Tp, class = void>
struct __unsatisfied_interface
{};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface, _Tp, _CUDA_VSTD::enable_if_t<_CUDA_VSTD::is_class_v<_Interface>>>
{
  using type _CCCL_NODEBUG_ALIAS = __unique_interfaces<_Interface, __satisfaction_fn<_Tp>>;
};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface*, _Tp*> : __unsatisfied_interface<_Interface, _Tp>
{};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface const*, _Tp const*> : __unsatisfied_interface<_Interface, _Tp>
{};

template <class _Interface, class _Tp>
struct __unsatisfied_interface<_Interface&, _Tp> : __unsatisfied_interface<_Interface, _Tp>
{};

template <class _Tp, class _Interface>
_LIBCUDACXX_CONCEPT __has_overrides = _CUDA_VSTD::_IsValidExpansion<__overrides_for, _Interface, _Tp>::value;

// The `__satisfies` concept checks if a type satisfies an interface. It does
// this by trying to instantiate `__overrides_for<_X, _Tp>` for all _X, where _X is
// _Interface or one of its bases. If any of the `__member_of` instantiations
// are ill-formed, then _Tp does not satisfy _Interface.
//
// `__satisfies` is implemented by searching through the list of interfaces for
// one that _Tp does not satisfy. If such an interface is found, the concept
// check fails in such a way as to hopefully tell the user which interface is
// not satisfied and why.
template <class _Tp,
          class _Interface,
          class UnsatisfiedInterface = _CUDA_VSTD::__type<__unsatisfied_interface<_Interface, _Tp>>>
_LIBCUDACXX_CONCEPT __satisfies = __has_overrides<_Tp, UnsatisfiedInterface>;

///
/// __vtable implementation details
///

template <class... _Interfaces>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES __vtable_tuple
    : __rtti_ex<sizeof...(_Interfaces)>
    , __vtable_for<_Interfaces>...
{
  static_assert((_CUDA_VSTD::is_class_v<_Interfaces> && ...), "expected class types");

  template <class _Tp, class _Super>
  _CUDAX_API constexpr __vtable_tuple(__tag<_Tp, _Super> __type) noexcept
      : __rtti_ex<sizeof...(_Interfaces)>{__type, __tag<_Interfaces...>(), this}
#ifdef _CCCL_COMPILER_MSVC
      // workaround for MSVC bug
      , __overrides_for<_Interfaces>::__vtable{__tag<_Tp>(), this}...
#else
      , __vtable_for<_Interfaces>{__tag<_Tp>(), this}...
#endif
  {
    static_assert(_CUDA_VSTD::is_class_v<_Super>, "expected a class type");
  }

  _LIBCUDACXX_TEMPLATE(class _Interface)
  _LIBCUDACXX_REQUIRES(_CUDA_VSTD::__is_included_in_v<_Interface, _Interfaces...>)
  _CCCL_NODISCARD _CUDAX_API constexpr __vptr_for<_Interface> __query_interface(_Interface) const noexcept
  {
    return static_cast<__vptr_for<_Interface>>(this);
  }
};

// The vtable type for type `_Interface` is a `__vtable_tuple` of `_Interface`
// and all of its base interfaces.
template <class _Interface>
using __vtable = __unique_interfaces<_Interface, _CUDA_VSTD::__type_quote<__vtable_tuple>>;

// __vtable_for_v<_Interface, _Tp> is an instance of `__vtable<_Interface>` that
// contains the overrides for `_Tp`.
template <class _Interface, class _Tp>
inline constexpr __vtable<_Interface> __vtable_for_v{__tag<_Tp, _Interface>()};

///
/// __interface_of
///
template <class _Super>
struct __make_interface_fn
{
  static_assert(_CUDA_VSTD::is_class_v<_Super>, "expected a class type");
  template <class... _Interfaces>
  using __call _CCCL_NODEBUG_ALIAS = detail::__inherit<__rebind_interface<_Interfaces, _Super>...>;
};

// Given an interface `_I<>`, let `_Bs<>...` be the list of types consisting
// of all of `_I<>`'s unique bases. Then `__interface_of<_I<>>` is the
// type `__inherit<_I<_I<>>, _Bs<_I<>>...>`. That is, it transforms the
// unspecialized interfaces into ones specialized for `_I<>` and then
// makes a type that inherits publicly from all of them.
template <class _Interface>
using __interface_of = __unique_interfaces<_Interface, __make_interface_fn<_Interface>>;

#if defined(__cpp_concepts)
template <class _Interface, int = 0>
struct __basic_any_base : __interface_of<_Interface>
{
private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  static constexpr size_t __size_  = __buffer_size(_Interface::size);
  static constexpr size_t __align_ = __buffer_align(_Interface::align);

  __tagged_ptr<__vptr_for<_Interface>> __vptr_{};
  alignas(__align_) _CUDA_VSTD_NOVERSION::byte __buffer_[__size_];
};
#else
// Without concepts, we need a base class to correctly implement movability
// and copyability.
template <class _Interface, int = extension_of<_Interface, imovable<>> + extension_of<_Interface, icopyable<>>>
struct __basic_any_base;

template <class _Interface>
struct __basic_any_base<_Interface, 2> : __interface_of<_Interface> // copyable interfaces
{
  _CUDAX_API __basic_any_base() = default;

  _CUDAX_API __basic_any_base(__basic_any_base&& __other) noexcept
  {
    static_cast<basic_any<_Interface>*>(this)->__convert_from(static_cast<basic_any<_Interface>&&>(__other));
  }

  _CUDAX_API __basic_any_base(__basic_any_base const& __other)
  {
    static_cast<basic_any<_Interface>*>(this)->__convert_from(static_cast<basic_any<_Interface> const&>(__other));
  }

  _CUDAX_API __basic_any_base& operator=(__basic_any_base&& __other) noexcept
  {
    static_cast<basic_any<_Interface>*>(this)->__assign_from(static_cast<basic_any<_Interface>&&>(__other));
    return *this;
  }

  _CUDAX_API __basic_any_base& operator=(__basic_any_base const& __other)
  {
    static_cast<basic_any<_Interface>*>(this)->__assign_from(static_cast<basic_any<_Interface> const&>(__other));
    return *this;
  }

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  static constexpr size_t __size_  = __buffer_size(_Interface::size);
  static constexpr size_t __align_ = __buffer_align(_Interface::align);

  __tagged_ptr<__vptr_for<_Interface>> __vptr_{};
  alignas(__align_) _CUDA_VSTD_NOVERSION::byte __buffer_[__size_];
};

template <class _Interface>
struct __basic_any_base<_Interface, 1> : __basic_any_base<_Interface, 2> // move-only interfaces
{
  _CUDAX_API __basic_any_base()                                       = default;
  _CUDAX_API __basic_any_base(__basic_any_base&&) noexcept            = default;
  __basic_any_base(__basic_any_base const&)                           = delete;
  _CUDAX_API __basic_any_base& operator=(__basic_any_base&&) noexcept = default;
  __basic_any_base& operator=(__basic_any_base const&)                = delete;
};

template <class _Interface>
struct __basic_any_base<_Interface, 0> : __basic_any_base<_Interface, 2> // immovable interfaces
{
  _CUDAX_API __basic_any_base()                            = default;
  __basic_any_base(__basic_any_base&&) noexcept            = delete;
  __basic_any_base(__basic_any_base const&)                = delete;
  __basic_any_base& operator=(__basic_any_base&&) noexcept = delete;
  __basic_any_base& operator=(__basic_any_base const&)     = delete;
};
#endif

///
/// basic_any
///
template <class _Interface, class>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any : __basic_any_base<_Interface>
{
private:
  static_assert(_CUDA_VSTD::is_class_v<_Interface>,
                "basic_any requires an interface type, or a pointer or reference to an interface "
                "type.");
  static_assert(!_CUDA_VSTD::is_const_v<_Interface>, "basic_any does not support const-qualified interfaces.");

  using __basic_any_base<_Interface>::__size_;
  using __basic_any_base<_Interface>::__align_;
  using __basic_any_base<_Interface>::__vptr_;
  using __basic_any_base<_Interface>::__buffer_;

public:
  using interface_type = _Interface;

  _CUDAX_API basic_any() = default;

  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) _LIBCUDACXX_AND __satisfies<_Tp, _Interface>)
  _CUDAX_API basic_any(_Tp __value) noexcept(__is_small<_Tp>(__size_, __align_))
  {
    __emplace<_Tp>(_CUDA_VSTD::move(__value));
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::decay_t<_Tp>, class... _Args)
  _LIBCUDACXX_REQUIRES(__list_initializable_from<_Up, _Args...> _LIBCUDACXX_AND __satisfies<_Tp, _Interface>)
  _CUDAX_API explicit basic_any(_CUDA_VSTD::in_place_type_t<_Tp>, _Args&&... __args) noexcept(
    __is_small<_Up>(__size_, __align_) && _CUDA_VSTD::is_nothrow_constructible_v<_Up, _Args...>)
  {
    __emplace<_Up>(static_cast<_Args&&>(__args)...);
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up, class _Vp = _CUDA_VSTD::decay_t<_Tp>, class... _Args)
  _LIBCUDACXX_REQUIRES(__list_initializable_from<_Vp, _CUDA_VSTD::initializer_list<_Up>&, _Args...> _LIBCUDACXX_AND
                         __satisfies<_Tp, _Interface>)
  _CUDAX_API explicit basic_any(
    _CUDA_VSTD::in_place_type_t<_Tp>,
    _CUDA_VSTD::initializer_list<_Up> __il,
    _Args&&... __args) noexcept(__is_small<_Vp>(__size_, __align_)
                                && _CUDA_VSTD::
                                  is_nothrow_constructible_v<_Vp, _CUDA_VSTD::initializer_list<_Up>&, _Args...>)
  {
    __emplace<_Vp>(__il, static_cast<_Args&&>(__args)...);
  }

#if defined(__cpp_concepts)
  _CUDAX_API basic_any(basic_any&& __other) noexcept
    requires(extension_of<_Interface, imovable<>>)
  {
    __convert_from(_CUDA_VSTD::move(__other));
  }

  _CUDAX_API basic_any(basic_any const& __other)
    requires(extension_of<_Interface, icopyable<>>)
  {
    __convert_from(__other);
  }
#else
  _CUDAX_API basic_any(basic_any&& __other)      = default;
  _CUDAX_API basic_any(basic_any const& __other) = default;
#endif

  // Conversions between compatible basic_any objects handled here:
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface>, basic_any>)
  _CUDAX_API basic_any(basic_any<_SrcInterface>&& __src)
  {
    __convert_from(_CUDA_VSTD::move(__src));
  }

  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface> const&, basic_any>)
  _CUDAX_API basic_any(basic_any<_SrcInterface> const& __src)
  {
    __convert_from(__src);
  }

  _CUDAX_API ~basic_any()
  {
    reset();
  }

#if defined(__cpp_concepts)
  _CUDAX_API basic_any& operator=(basic_any&& __other) noexcept
    requires(extension_of<_Interface, imovable<>>)
  {
    return __assign_from(_CUDA_VSTD::move(__other));
  }

  _CUDAX_API basic_any& operator=(basic_any const& __other)
    requires(extension_of<_Interface, icopyable<>>)
  {
    return __assign_from(__other);
  }
#else
  _CUDAX_API basic_any& operator=(basic_any&& __other)      = default;
  _CUDAX_API basic_any& operator=(basic_any const& __other) = default;
#endif

  // Assignment from a compatible basic_any object handled here:
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface>, basic_any>)
  _CUDAX_API basic_any& operator=(basic_any<_SrcInterface>&& __src)
  {
    return __assign_from(_CUDA_VSTD::move(__src));
  }

  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface> const&, basic_any>)
  _CUDAX_API basic_any& operator=(basic_any<_SrcInterface> const& __src)
  {
    return __assign_from(__src);
  }

  // Implicit conversions to reference types here:
  _CCCL_NODISCARD _CUDAX_API operator basic_any<__ireference<_Interface>>() & noexcept
  {
    return basic_any<__ireference<_Interface>>(*this);
  }

  _CCCL_NODISCARD _CUDAX_API operator basic_any<__ireference<_Interface const>>() const& noexcept
  {
    return basic_any<__ireference<_Interface const>>(*this);
  }

  _CUDAX_API void swap(basic_any& __other) noexcept
  {
    /// if both objects refer to heap-allocated object, we can just
    /// swap the pointers. otherwise, do it the slow(er) way.
    if (!__in_situ() && !__other.__in_situ())
    {
      _CUDA_VSTD::swap(__vptr_, __other.__vptr_);
      __swap_ptr_ptr(__buffer_, __other.__buffer_);
    }

    basic_any __tmp;
    __tmp.__convert_from(_CUDA_VSTD::move(*this));
    (*this).__convert_from(_CUDA_VSTD::move(__other));
    __other.__convert_from(_CUDA_VSTD::move(__tmp));
  }

  friend _CUDAX_TRIVIAL_API void swap(basic_any& __lhs, basic_any& __rhs) noexcept
  {
    __lhs.swap(__rhs);
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::decay_t<_Tp>, class... _Args)
  _LIBCUDACXX_REQUIRES(__list_initializable_from<_Up, _Args...>)
  _CUDAX_API _Up& emplace(_Args&&... __args) noexcept(
    __is_small<_Up>(__size_, __align_) && _CUDA_VSTD::is_nothrow_constructible_v<_Up, _Args...>)
  {
    reset();
    return __emplace<_Up>(static_cast<_Args&&>(__args)...);
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up, class _Vp = _CUDA_VSTD::decay_t<_Tp>, class... _Args)
  _LIBCUDACXX_REQUIRES(__list_initializable_from<_Vp, _CUDA_VSTD::initializer_list<_Up>&, _Args...>)
  _CUDAX_API _Vp& emplace(_CUDA_VSTD::initializer_list<_Up> __il, _Args&&... __args) noexcept(
    __is_small<_Vp>(__size_, __align_)
    && _CUDA_VSTD::is_nothrow_constructible_v<_Vp, _CUDA_VSTD::initializer_list<_Up>&, _Args...>)
  {
    reset();
    return __emplace<_Vp>(__il, static_cast<_Args&&>(__args)...);
  }

  _CCCL_NODISCARD _CUDAX_API bool has_value() const noexcept
  {
    return __get_vptr() != nullptr;
  }

  _CUDAX_API void reset()
  {
    if (auto __vptr = __get_vptr())
    {
      _CCCL_ASSERT(__vptr->__query_interface(iunknown())->__cookie_ == 0xDEADBEEF,
                   "query_interface returned a bad pointer to the iunknown vtable");
      __vptr->__query_interface(iunknown())->__dtor_(__buffer_, __in_situ());
      __release();
    }
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref type() const noexcept
  {
    if (auto __vptr = __get_vptr())
    {
      _CCCL_ASSERT(__vptr->__query_interface(iunknown())->__cookie_ == 0xDEADBEEF,
                   "query_interface returned a bad pointer to the iunknown vtable");
      return *__vptr->__query_interface(iunknown())->__object_info_->__object_typeid_;
    }
    return _CCCL_TYPEID(void);
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref interface() const noexcept
  {
    if (auto __vptr = __get_vptr())
    {
      _CCCL_ASSERT(__vptr->__query_interface(iunknown())->__cookie_ == 0xDEADBEEF,
                   "query_interface returned a bad pointer to the iunknown vtable");
      return *__vptr->__query_interface(iunknown())->__interface_typeid_;
    }
    return _CCCL_TYPEID(_Interface);
  }

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Do not document
  _CCCL_NODISCARD _CUDAX_API bool __in_situ() const noexcept
  {
    return __vptr_.__flag();
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;
  template <class, int>
  friend struct __basic_any_base;

  _CUDAX_API void __release()
  {
    __vptr_for<_Interface> __vptr = nullptr;
    __vptr_.__set(__vptr, false);
  }

  template <class _Tp, class... _Args>
  _CUDAX_API _Tp& __emplace(_Args&&... __args) noexcept(
    __is_small<_Tp>(__size_, __align_) && _CUDA_VSTD::is_nothrow_constructible_v<_Tp, _Args...>)
  {
    if constexpr (__is_small<_Tp>(__size_, __align_))
    {
      ::new (__buffer_) _Tp{static_cast<_Args&&>(__args)...};
    }
    else
    {
      ::new (__buffer_) __identity<_Tp*>{new _Tp{static_cast<_Args&&>(__args)...}};
    }

    __vptr_for<_Interface> __vptr = &__vtable_for_v<_Interface, _Tp>;
    __vptr_.__set(__vptr, __is_small<_Tp>(__size_, __align_));
    return *_CUDA_VSTD::launder(static_cast<_Tp*>(__get_optr()));
  }

  // this overload handles moving from basic_any<_SrcInterface> and
  // basic_any<__ireference<_SrcInterface>> (but not basic_any<__ireference<_SrcInterface
  // const>>).
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>, basic_any>)
  _CUDAX_API void
  __convert_from(basic_any<_SrcInterface>&& __from) noexcept(_CUDA_VSTD::is_same_v<_SrcInterface, _Interface>)
  {
    _CCCL_ASSERT(!has_value(), "forgot to clear the destination object first");
    using __src_interface_t = __remove_ireference_t<_SrcInterface>;
    // if the source is an lvalue reference, we need to copy from it.
    if constexpr (__is_lvalue_reference_v<_SrcInterface>)
    {
      __convert_from(__from); // will copy from the source
    }
    else if (auto __to_vptr = __vptr_cast<__src_interface_t, _Interface>(__from.__get_vptr()))
    {
      if (!__from.__in_situ())
      {
        ::new (__buffer_) __identity<void*>(__from.__get_optr());
        __vptr_.__set(__to_vptr, false);
        __from.__release();
      }
      else if constexpr (_CUDA_VSTD::is_same_v<_SrcInterface, _Interface>)
      {
        __from.__move_to(__buffer_);
        __vptr_.__set(__from.__get_vptr(), true);
        __from.reset();
      }
      else
      {
        bool const __small = __from.__move_to(__buffer_, __size_, __align_);
        __vptr_.__set(__to_vptr, __small);
        __from.reset();
      }
    }
  }

  // this overload handles copying from basic_any<_Interface>,
  // basic_any<__ireference<_Interface>>, and basic_any<_Interface&>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const&, basic_any>)
  _CUDAX_API void __convert_from(basic_any<_SrcInterface> const& __from)
  {
    _CCCL_ASSERT(!has_value(), "forgot to clear the destination object first");
    using __src_interface_t = __remove_ireference_t<_CUDA_VSTD::remove_reference_t<_SrcInterface>>;
    if (auto __to_vptr = __vptr_cast<__src_interface_t, _Interface>(__from.__get_vptr()))
    {
      bool const __small = __from.__copy_to(__buffer_, __size_, __align_);
      __vptr_.__set(__to_vptr, __small);
    }
  }

  // Assignment from a compatible basic_any object handled here:
  _LIBCUDACXX_TEMPLATE(class _SrcCvAny)
  _LIBCUDACXX_REQUIRES(__any_castable_to<_SrcCvAny, basic_any>)
  _CUDAX_API basic_any& __assign_from(_SrcCvAny&& __src)
  {
    if (!__ptr_eq(this, &__src))
    {
      reset();
      __convert_from(static_cast<_SrcCvAny&&>(__src));
    }
    return *this;
  }

  _CCCL_NODISCARD _CUDAX_API __vptr_for<_Interface> __get_vptr() const noexcept
  {
    return __vptr_.__get();
  }

  _CCCL_NODISCARD _CUDAX_API void* __get_optr() noexcept
  {
    void* __pv = __buffer_;
    return __in_situ() ? __pv : *static_cast<void**>(__pv);
  }

  _CCCL_DIAG_PUSH
  _CCCL_DIAG_SUPPRESS_MSVC(4702) // warning C4702: unreachable code (srsly where, msvc?)
  _CCCL_NODISCARD _CUDAX_API void const* __get_optr() const noexcept
  {
    void const* __pv = __buffer_;
    return __in_situ() ? __pv : *static_cast<void const* const*>(__pv);
  }
  _CCCL_DIAG_POP

  _CCCL_NODISCARD _CUDAX_API __rtti const* __get_rtti() const noexcept
  {
    return __get_vptr()->__query_interface(iunknown());
  }
};

#if !defined(__cpp_concepts)
///
/// A base class for basic_any<__ireference<_Interface>> that provides a conversion
/// to basic_any<__ireference<_Interface const>>.
///
template <class _Interface>
struct __basic_any_reference_conversion_base
{
  _CCCL_NODISCARD _CUDAX_API operator basic_any<__ireference<_Interface const>>() const noexcept
  {
    return basic_any<__ireference<_Interface const>>(static_cast<basic_any<__ireference<_Interface>> const&>(*this));
  }
};

template <class _Interface>
struct __basic_any_reference_conversion_base<_Interface const>
{};
#endif

_CCCL_DIAG_PUSH
// "operator basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_DIAG_SUPPRESS_NVHPC(conversion_function_not_usable)
// "operator basic_any<...> will not be called for implicit or explicit conversions"
_CCCL_NV_DIAG_SUPPRESS(554)

///
/// basic_any<__ireference<_Interface>>
///
template <class _Interface, class Select>
struct _LIBCUDACXX_DECLSPEC_EMPTY_BASES basic_any<__ireference<_Interface>, Select>
    : __interface_of<__ireference<_Interface>>
#if !defined(__cpp_concepts)
    , __basic_any_reference_conversion_base<_Interface>
#endif
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expecting a class type");
  using interface_type                 = _CUDA_VSTD::remove_const_t<_Interface>;
  static constexpr bool __is_const_ref = _CUDA_VSTD::is_const_v<_Interface>;

  basic_any(basic_any&&)      = delete;
  basic_any(basic_any const&) = delete;

  basic_any& operator=(basic_any&&)      = delete;
  basic_any& operator=(basic_any const&) = delete;

#if defined(__cpp_concepts)
  _CCCL_NODISCARD _CUDAX_API operator basic_any<__ireference<_Interface const>>() const noexcept
    requires(!__is_const_ref)
  {
    return basic_any<__ireference<_Interface const>>(*this);
  }
#endif

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref type() const noexcept
  {
    return *__get_rtti()->__object_info_->__object_typeid_;
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref interface() const noexcept
  {
    return *__get_rtti()->__interface_typeid_;
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API static constexpr bool has_value() noexcept
  {
    return true;
  }

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Do not document
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API static constexpr bool __in_situ() noexcept
  {
    return true;
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  _CUDAX_API basic_any() = default;

  _CUDAX_API explicit basic_any(_CUDA_VSTD::__maybe_const<__is_const_ref, basic_any<interface_type>>& __other) noexcept
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  _CUDAX_API basic_any(__vptr_for<interface_type> __vptr,
                       _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __optr) noexcept
      : __vptr_(__vptr)
      , __optr_(__optr)
  {}

  _CUDAX_TRIVIAL_API void reset() noexcept {}

  _CUDAX_TRIVIAL_API void __release() {}

  _CUDAX_API void __set_ref(__vptr_for<interface_type> __vptr,
                            _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    __vptr_ = __vptr;
    __optr_ = __vptr_ ? __obj : nullptr;
  }

  template <class _VTable>
  _CUDAX_API void __set_ref(_VTable const* __other, _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = typename _VTable::interface;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  template <class... _Interfaces>
  _CUDAX_API void __set_ref(__iset_vptr<_Interfaces...> __other,
                            _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __obj) noexcept
  {
    using _OtherInterface = __iset<_Interfaces...>;
    __vptr_               = __try_vptr_cast<_OtherInterface, interface_type>(__other);
    __optr_               = __vptr_ ? __obj : nullptr;
  }

  template <class _SrcCvAny>
  _CUDAX_API void __convert_from(_SrcCvAny&& __from)
  {
    using __src_interface_t = typename _CUDA_VSTD::remove_reference_t<_SrcCvAny>::interface_type;
    if (!__from.has_value())
    {
      __throw_bad_any_cast();
    }
    auto __to_vptr = __vptr_cast<__src_interface_t, interface_type>(__from.__get_vptr());
    __set_ref(__to_vptr, __from.__get_optr());
  }

  _CUDAX_TRIVIAL_API _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __get_optr() const noexcept
  {
    return __optr_;
  }

  _CUDAX_TRIVIAL_API __vptr_for<interface_type> __get_vptr() const noexcept
  {
    return __vptr_;
  }

  _CUDAX_TRIVIAL_API __rtti const* __get_rtti() const noexcept
  {
    return __vptr_->__query_interface(iunknown());
  }

  __vptr_for<interface_type> __vptr_{};
  _CUDA_VSTD::__maybe_const<__is_const_ref, void>* __optr_{};
};

_CCCL_NV_DIAG_DEFAULT(554)
_CCCL_DIAG_POP

///
/// basic_any<_Interface&>
///
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<_Interface&> : basic_any<__ireference<_Interface>, int>
{
  static_assert(_CUDA_VSTD::is_class_v<_Interface>, "expecting a class type");
  using typename basic_any<__ireference<_Interface>, int>::interface_type;
  using basic_any<__ireference<_Interface>, int>::__is_const_ref;

  _CUDAX_API basic_any(basic_any const& __other) noexcept
  {
    this->__set_ref(__other.__get_vptr(), __other.__get_optr());
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) _LIBCUDACXX_AND __satisfies<_Up, interface_type> _LIBCUDACXX_AND(
    __is_const_ref || !_CUDA_VSTD::is_const_v<_Tp>))
  _CUDAX_API basic_any(_Tp& __obj) noexcept
  {
    __vptr_for<interface_type> const __vptr = &__vtable_for_v<interface_type, _Up>;
    this->__set_ref(__vptr, &__obj);
  }

  _LIBCUDACXX_TEMPLATE(class _Tp)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) )
  basic_any(_Tp const&&) = delete;

  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>) _LIBCUDACXX_AND //
                       (!__is_value_v<_SrcInterface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface>, basic_any>)
  _CUDAX_API basic_any(basic_any<_SrcInterface>&& __src) noexcept
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface>&, basic_any>)
  _CUDAX_API basic_any(basic_any<_SrcInterface>& __src) noexcept
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_SrcInterface, _Interface&>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_SrcInterface> const&, basic_any>)
  _CUDAX_API basic_any(basic_any<_SrcInterface> const& __src) noexcept
  {
    this->__set_ref(__src.__get_vptr(), __src.__get_optr());
  }

  // A temporary value cannot bind to a basic_any reference.
  // TODO: find another way to support APIs that take by reference and want
  // implicit conversion from prvalues.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__is_value_v<_SrcInterface>) //
  basic_any(basic_any<_SrcInterface> const&&) = delete;

  basic_any& operator=(basic_any&&)      = delete;
  basic_any& operator=(basic_any const&) = delete;

  _CUDAX_TRIVIAL_API basic_any<__ireference<_Interface>>&& move() & noexcept
  {
    return _CUDA_VSTD::move(*this);
  }

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  _CUDAX_API basic_any() = default;
};

///
/// basic_any<_Interface*>
///
template <class _Interface>
struct _CCCL_TYPE_VISIBILITY_DEFAULT basic_any<_Interface*>
{
  using interface_type                 = _CUDA_VSTD::remove_const_t<_Interface>;
  static constexpr bool __is_const_ptr = _CUDA_VSTD::is_const_v<_Interface>;

  ///
  /// Constructors
  ///
  _CUDAX_API basic_any() = default;

  _CUDAX_TRIVIAL_API basic_any(_CUDA_VSTD::nullptr_t) {}

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) _LIBCUDACXX_AND __satisfies<_Up, interface_type> _LIBCUDACXX_AND(
    __is_const_ptr || !_CUDA_VSTD::is_const_v<_Tp>))
  _CUDAX_API basic_any(_Tp* __obj) noexcept
  {
    operator=(__obj);
  }

  _CUDAX_API basic_any(basic_any const& __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_OtherInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_OtherInterface*>, basic_any<_Interface*>>)
  _CUDAX_API basic_any(basic_any<_OtherInterface*> const& __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface>&, basic_any<_Interface&>>)
  _CUDAX_API basic_any(basic_any<_OtherInterface>* __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface> const&, basic_any<_Interface&>>)
  _CUDAX_API basic_any(basic_any<_OtherInterface> const* __other) noexcept
  {
    __convert_from(__other);
  }

  _LIBCUDACXX_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _LIBCUDACXX_REQUIRES(__is_interface<_OtherInterface<_Super>> _LIBCUDACXX_AND
                         _CUDA_VSTD::derived_from<basic_any<_Super>, _OtherInterface<_Super>> _LIBCUDACXX_AND
                           _CUDA_VSTD::same_as<__normalized_interface_of<basic_any<_Super>*>, _Interface*>)
  _CUDAX_API explicit basic_any(_OtherInterface<_Super>* __self) noexcept
  {
    __convert_from(basic_any_from(__self));
  }

  _LIBCUDACXX_TEMPLATE(template <class...> class _OtherInterface, class _Super)
  _LIBCUDACXX_REQUIRES(__is_interface<_OtherInterface<_Super>> _LIBCUDACXX_AND
                         _CUDA_VSTD::derived_from<basic_any<_Super>, _OtherInterface<_Super>> _LIBCUDACXX_AND
                           _CUDA_VSTD::same_as<__normalized_interface_of<basic_any<_Super> const*>, _Interface*>)
  _CUDAX_API explicit basic_any(_OtherInterface<_Super> const* __self) noexcept
  {
    __convert_from(basic_any_from(__self));
  }

  ///
  /// Assignment operators
  ///
  _CUDAX_API basic_any& operator=(_CUDA_VSTD::nullptr_t) noexcept
  {
    reset();
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _Tp, class _Up = _CUDA_VSTD::remove_const_t<_Tp>)
  _LIBCUDACXX_REQUIRES((!__is_basic_any<_Tp>) _LIBCUDACXX_AND //
                         __satisfies<_Up, interface_type> _LIBCUDACXX_AND //
                       (__is_const_ptr || !_CUDA_VSTD::is_const_v<_Tp>))
  _CUDAX_API basic_any& operator=(_Tp* __obj) noexcept
  {
    __vptr_for<interface_type> __vptr = &__vtable_for_v<interface_type, _Up>;
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *this;
  }

  _CUDAX_API basic_any& operator=(basic_any const& __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES((!_CUDA_VSTD::same_as<_OtherInterface, _Interface>)
                         _LIBCUDACXX_AND __any_convertible_to<basic_any<_OtherInterface*>, basic_any<_Interface*>>)
  _CUDAX_API basic_any& operator=(basic_any<_OtherInterface*> const& __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface>&, basic_any<_Interface&>>)
  _CUDAX_API basic_any& operator=(basic_any<_OtherInterface>* __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  _LIBCUDACXX_TEMPLATE(class _OtherInterface)
  _LIBCUDACXX_REQUIRES(__any_convertible_to<basic_any<_OtherInterface> const&, basic_any<_Interface&>>)
  _CUDAX_API basic_any& operator=(basic_any<_OtherInterface> const* __other) noexcept
  {
    __convert_from(__other);
    return *this;
  }

  ///
  /// emplace
  ///
  _LIBCUDACXX_TEMPLATE(
    class _Tp, class _Up = _CUDA_VSTD::remove_pointer_t<_Tp>, class _Vp = _CUDA_VSTD::remove_const_t<_Up>)
  _LIBCUDACXX_REQUIRES(__satisfies<_Vp, _Interface> _LIBCUDACXX_AND(__is_const_ptr || !_CUDA_VSTD::is_const_v<_Up>))
  _CUDAX_API _CUDA_VSTD::__maybe_const<__is_const_ptr, _Vp>*& emplace(_CUDA_VSTD::type_identity_t<_Up>* __obj) noexcept
  {
    __vptr_for<interface_type> __vptr = &__vtable_for_v<interface_type, _Vp>;
    __ref_.__set_ref(__obj ? __vptr : nullptr, __obj);
    return *static_cast<_CUDA_VSTD::__maybe_const<__is_const_ptr, _Vp>**>(static_cast<void*>(&__ref_.__optr_));
  }

#if defined(__cpp_three_way_comparison)
  _CCCL_NODISCARD _CUDAX_API bool operator==(basic_any const& __other) const noexcept
  {
    using __void_ptr_t = _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__get_optr()) == *static_cast<__void_ptr_t>(__other.__get_optr());
  }
#else
  _CCCL_NODISCARD_FRIEND _CUDAX_API bool operator==(basic_any const& __lhs, basic_any const& __rhs) noexcept
  {
    using __void_ptr_t = _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const*;
    return *static_cast<__void_ptr_t>(__lhs.__get_optr()) == *static_cast<__void_ptr_t>(__rhs.__get_optr());
  }

  _CCCL_NODISCARD_FRIEND _CUDAX_TRIVIAL_API bool operator!=(basic_any const& __lhs, basic_any const& __rhs) noexcept
  {
    return !(__lhs == __rhs);
  }
#endif

  using __any_ref_t _CCCL_NODEBUG_ALIAS =
    _CUDA_VSTD::__maybe_const<__is_const_ptr, basic_any<__ireference<_Interface>>>;

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API auto operator->() const noexcept -> __any_ref_t*
  {
    return &__ref_;
  }

  _CCCL_NODISCARD _CUDAX_TRIVIAL_API auto operator*() const noexcept -> __any_ref_t&
  {
    return __ref_;
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref type() const noexcept
  {
    return __ref_.__vptr_ != nullptr
           ? (__is_const_ptr ? *__get_rtti()->__object_info_->__const_pointer_typeid_
                             : *__get_rtti()->__object_info_->__pointer_typeid_)
           : _CCCL_TYPEID(void);
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__type_info_ref interface() const noexcept
  {
    return __ref_.__vptr_ != nullptr ? *__get_rtti()->__interface_typeid_ : _CCCL_TYPEID(interface_type);
  }

  _CCCL_NODISCARD _CUDAX_API bool has_value() const noexcept
  {
    return __ref_.__vptr_ != nullptr;
  }

  _CUDAX_API void reset() noexcept
  {
    __vptr_for<interface_type> __vptr = nullptr;
    __ref_.__set_ref(__vptr, nullptr);
  }

  _CCCL_NODISCARD _CUDAX_API explicit operator bool() const noexcept
  {
    return __ref_.__vptr_ != nullptr;
  }

#if !defined(DOXYGEN_SHOULD_SKIP_THIS) // Do not document
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API static constexpr bool __in_situ() noexcept
  {
    return true;
  }
#endif // DOXYGEN_SHOULD_SKIP_THIS

private:
  template <class, class>
  friend struct basic_any;
  friend struct __basic_any_access;

  template <class _SrcCvAny>
  _CUDAX_API void __convert_from(_SrcCvAny* __other) noexcept
  {
    __other ? __ref_.__set_ref(__other->__get_vptr(), __other->__get_optr()) : reset();
  }

  template <class _OtherInterface>
  _CUDAX_API void __convert_from(basic_any<_OtherInterface*> const& __other) noexcept
  {
    using __other_interface_t = _CUDA_VSTD::remove_const_t<_OtherInterface>;
    auto __to_vptr            = __try_vptr_cast<__other_interface_t, interface_type>(__other.__get_vptr());
    auto __to_optr            = __to_vptr ? *__other.__get_optr() : nullptr;
    __ref_.__set_ref(__to_vptr, __to_optr);
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__maybe_const<__is_const_ptr, void>** __get_optr() noexcept
  {
    return &__ref_.__optr_;
  }

  _CCCL_NODISCARD _CUDAX_API _CUDA_VSTD::__maybe_const<__is_const_ptr, void>* const* __get_optr() const noexcept
  {
    return &__ref_.__optr_;
  }

  _CCCL_NODISCARD _CUDAX_API __vptr_for<interface_type> __get_vptr() const noexcept
  {
    return __ref_.__vptr_;
  }

  _CCCL_NODISCARD _CUDAX_API __rtti const* __get_rtti() const noexcept
  {
    return __ref_.__vptr_ ? __ref_.__vptr_->__query_interface(iunknown()) : nullptr;
  }

  mutable basic_any<__ireference<_Interface>> __ref_;
};

_LIBCUDACXX_TEMPLATE(template <class...> class _Interface, class _Super)
_LIBCUDACXX_REQUIRES(__is_interface<_Interface<_Super>>)
_CUDAX_PUBLIC_API basic_any(_Interface<_Super>*) //
  -> basic_any<__normalized_interface_of<basic_any<_Super>*>>;

_LIBCUDACXX_TEMPLATE(template <class...> class _Interface, class _Super)
_LIBCUDACXX_REQUIRES(__is_interface<_Interface<_Super>>)
_CUDAX_PUBLIC_API basic_any(_Interface<_Super> const*) //
  -> basic_any<__normalized_interface_of<basic_any<_Super> const*>>;

///
/// __valid_any_cast
///
template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast = true;

template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast<_Interface*, _Tp> = false;

template <class _Interface, class _Tp>
inline constexpr bool __valid_any_cast<_Interface*, _Tp*> =
  !_CUDA_VSTD::is_const_v<_Interface> || _CUDA_VSTD::is_const_v<_Tp>;

///
/// __dynamic_any_cast_fn
///
template <class _DstInterface>
struct __dynamic_any_cast_fn
{
  /// @throws bad_any_cast when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface>&& __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(_CUDA_VSTD::move(__src), __dst);
    return __dst;
  }

  /// @throws bad_any_cast when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>&, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface>& __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }

  /// @throws bad_any_cast when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const&, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface> const& __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }

  /// @returns \c nullptr when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface>*, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface>* __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }

  /// @returns \c nullptr when \c __src cannot be dynamically cast to a
  /// \c basic_any<_DstInterface>.
  _LIBCUDACXX_TEMPLATE(class _SrcInterface)
  _LIBCUDACXX_REQUIRES(__any_castable_to<basic_any<_SrcInterface> const*, basic_any<_DstInterface>>)
  _CCCL_NODISCARD _CUDAX_API auto operator()(basic_any<_SrcInterface> const* __src) const -> basic_any<_DstInterface>
  {
    auto __dst = __basic_any_access::__make<_DstInterface>();
    __basic_any_access::__cast_to(__src, __dst);
    return __dst;
  }
};

///
/// dynamic_any_cast
///
/// @throws bad_any_cast when \c from cannot be dynamically cast to a
/// \c basic_any<_DstInterface>
///
template <class _DstInterface>
inline constexpr __dynamic_any_cast_fn<_DstInterface> dynamic_any_cast{};

///
/// any_cast
///
_LIBCUDACXX_TEMPLATE(class _Tp, class _Interface)
_LIBCUDACXX_REQUIRES(__satisfies<_Tp, _Interface> || _CUDA_VSTD::is_void_v<_Tp>)
_CCCL_NODISCARD _CUDAX_API _Tp* any_cast(basic_any<_Interface>* __self) noexcept
{
  static_assert(__valid_any_cast<_Interface, _Tp>);
  if (__self && (_CUDA_VSTD::is_void_v<_Tp> || __self->type() == _CCCL_TYPEID(_Tp)))
  {
    return static_cast<_Tp*>(__basic_any_access::__get_optr(*__self));
  }
  return nullptr;
}

_LIBCUDACXX_TEMPLATE(class _Tp, class _Interface)
_LIBCUDACXX_REQUIRES(__satisfies<_Tp, _Interface> || _CUDA_VSTD::is_void_v<_Tp>)
_CCCL_NODISCARD _CUDAX_API _Tp const* any_cast(basic_any<_Interface> const* __self) noexcept
{
  static_assert(__valid_any_cast<_Interface, _Tp>);
  if (__self && (_CUDA_VSTD::is_void_v<_Tp> || __self->type() == _CCCL_TYPEID(_Tp)))
  {
    return static_cast<_Tp const*>(__basic_any_access::__get_optr(*__self));
  }
  return nullptr;
}

///
/// interface_cast
///
/// given a `basic_any<X<>>` object `o`, `interface_cast<Y>(o)` return a
/// reference to the (empty) sub-object of type `Y<X<>>`, from which
/// `basic_any<X<>>` inherits, where `Y<>` is an interface that `X<>` extends.
///
template <class _Interface>
struct __interface_cast_fn;

template <template <class...> class _Interface>
struct __interface_cast_fn<_Interface<>>
{
  template <class _Super>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API _Interface<_Super>&& operator()(_Interface<_Super>&& __self) const noexcept
  {
    return _CUDA_VSTD::move(__self);
  }

  template <class _Super>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API _Interface<_Super>& operator()(_Interface<_Super>& __self) const noexcept
  {
    return __self;
  }

  template <class _Super>
  _CCCL_NODISCARD _CUDAX_TRIVIAL_API _Interface<_Super> const& operator()(_Interface<_Super> const& __self) noexcept
  {
    return __self;
  }
};

_LIBCUDACXX_TEMPLATE(template <class...> class _Interface, class Object)
_LIBCUDACXX_REQUIRES(
  __is_interface<_Interface<>> _LIBCUDACXX_AND _CUDA_VSTD::__is_callable_v<__interface_cast_fn<_Interface<>>, Object>)
_CCCL_NODISCARD _CUDAX_TRIVIAL_API decltype(auto) interface_cast(Object&& __obj) noexcept
{
  return __interface_cast_fn<_Interface<>>{}(_CUDA_VSTD::forward<Object>(__obj));
}

_LIBCUDACXX_TEMPLATE(class _Interface, class Object)
_LIBCUDACXX_REQUIRES(
  __is_interface<_Interface> _LIBCUDACXX_AND _CUDA_VSTD::__is_callable_v<__interface_cast_fn<_Interface>, Object>)
_CCCL_NODISCARD _CUDAX_TRIVIAL_API decltype(auto) interface_cast(Object&& __obj) noexcept
{
  return __interface_cast_fn<_Interface>{}(_CUDA_VSTD::forward<Object>(__obj));
}
} // namespace cuda::experimental

_CCCL_NV_DIAG_DEFAULT(20012)
_CCCL_DIAG_POP

#endif // __CUDAX_DETAIL_BASIC_ANY_H
