//===----------------------------------------------------------------------===//
//
// Part of libcu++, the C++ Standard Library for your entire system,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef _CUDA___UTILITY_BASIC_ANY_H
#define _CUDA___UTILITY_BASIC_ANY_H

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

//! \file basic_any.h
//!
//! \brief This file provides the implementation of the `__basic_any` class
//! template.
//!
//! The `__basic_any` class template generates a type-erasing wrapper for types
//! described by a user-provided _interface_. An interface is a class template
//! that has member and/or friend functions that dispatch to the appropriate
//! pseudo-virtual functions, and a nested `overrides` type alias that maps the
//! members of a given type to the pseudo-virtuals. Objects that implement the
//! interface will be stored in-situ in the `__basic_any` object if possible;
//! otherwise, they will be stored on the heap.
//!
//! A simple interface looks like this:
//!
//! \code{.cpp}
//! template <class...>
//! struct icat : cuda::__basic_interface<icat>
//! {
//!   void meow() const
//!   {
//!     // dispatch to the icat::meow override:
//!     cuda::__virtcall<&icat::meow>(this);
//!   }
//!
//!   template <class Cat>
//!   using overrides = cuda::__overrides_for<Cat, &Cat::meow>;
//! };
//!
//! // `any_cat` is a type-erasing wrapper that can store any object whose type
//! // satisfies the `icat` interface.
//! using any_cat = cuda::__basic_any<icat<>>;
//! \endcode
//!
//! The `any_cat` type can now be used to store any object that implements the
//! `icat` interface. For example:
//!
//! \code{.cpp}
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
//! \endcode
//!
//!
//! ## The `__basic_any` type
//!
//! For an interface `I`, the `__basic_any<I>` has the following member functions
//! in addition to those provided by `I`:
//!
//! \code{.cpp}
//! void swap(__basic_any<I>& other) noexcept;
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
//! \endcode
//!
//! These members behave as they do for `std::any`. The `interface()` member
//! function returns the `std::type_info` object for the _dynamic_ interface of
//! the stored object, which could be different from `I` if the `__basic_any`
//! object was constructed from a `__basic_any<J>`, where `J` extends `I` (see
//! the next section).
//!
//! ## Interface Extension
//!
//! An interface can extend other interfaces. We can make `any_cat` copyable by
//! extending the `cuda::__icopyable` interface:
//!
//! \code{.cpp}
//! template <class...>
//! struct icat
//!   : cuda::__basic_interface<icat, cuda::__extends<cuda::__icopyable<>>>
//! {
//!   ... as before ...
//! \endcode
//!
//! The `cuda::extends` template is variadic to allow extending multiple
//! interfaces. Interface extension works like virtual inheritance in C++; the
//! same interface can be inherited multiple times, either directly or
//! indirectly, without creating any ambiguity.
//!
//! Conversion from a "derived" `__basic_any` (e.g., `any_cat`) to a "base"
//! `__basic_any` (e.g., `__basic_any<__icopyable<>>`) is supported.
//!
//! ## Pointers and References
//!
//! The `__basic_any` class template can hold pointers and references to objects
//! that implement the interface. For example:
//!
//! \code{.cpp}
//! // `any_cat_ptr` is a type-erasing wrapper that can store a pointer to any
//! // (non-const) object whose type implements the `icat` interface.
//! using any_cat_ptr = cuda::__basic_any<icat<>*>; // note the pointer type!
//!
//! // store a pointer to a siamese cat in a __basic_any object
//! siamese fluffy;
//! any_cat_ptr pcat1 = &fluffy;
//! pcat1->meow(); // prints "meow"
//!
//! // a __basic_any pointer can also point to a type-erased object:
//! any_cat cat2 = fluffy;
//! any_cat_ptr pcat2 = &cat2;
//! pcat2->meow(); // prints "meow"
//! \endcode
//!
//! If the base interface (in this case, `icat<>`) extends `__icopyable<>` or
//! `__imovable<>`, then you can copy or move out of a dereferenced __basic_any
//! pointer. For example:
//!
//! \code{.cpp}
//! any_cat_ptr pcat = &fluffy;
//! any_cat copycat = *pcat;            // copy from fluffy
//! any_cat movecat = std::move(*pcat); // move from fluffy
//! \endcode
//!
//! Type-erased references are supported as well. For example:
//!
//! \code{.cpp}
//! // `any_cat_ref` is a type-erasing wrapper that can store a (non-owning)
//! // reference to any (non-const) object whose type implements the `icat`
//! // interface.
//! using any_cat_ref = cuda::__basic_any<icat<>&>; // note the reference type!
//!
//! // store a non-const reference to a siamese cat in a __basic_any object:
//! siamese fluffy;
//! any_cat_ref rcat1 = fluffy;
//! rcat1.meow(); // prints "meow"
//! \endcode
//!
//! All the conversion sequences that are valid for pointers and references are
//! also valid for __basic_any pointers and references. For example, a
//! `__basic_any<I*>` can be implicitly converted to a `__basic_any<I const*>`, and
//! a `__basic_any<IDerived&>` is implicitly convertible to a
//! `__basic_any<IBase&>`.
//!
//! \warning **You cannot use `std::move` to move a value out of a __basic_any
//! reference.** Although `any_cat value = std::move(rcat1);` will compile and
//! looks perfectly reasonable, it will not actually move the value out of
//! `rcat1`. Instead, it will copy the value. `std::move(rcat1)` is an rvalue
//! reference of type `any_cat_ref`, which continues to behave like an lvalue.
//! For this reason, __basic_any references all have a `move` member function that
//! can be used to move the value out of the reference: `any_cat value =
//! rcat1.move();`.
//!
//! ## In-Situ Storage
//!
//! Small objects with non-throwing move constructors can be stored in-situ in
//! the `__basic_any` object. The size of the in-situ buffer is determined by the
//! interface's `::size` member, which defaults to `3 * sizeof(void*)` and
//! cannot be smaller than `sizeof(void*)`. The alignment of the in-situ buffer
//! is determined by the interface's `::align` member, which defaults to
//! `alignof(std::max_align_t)` and cannot be smaller than `alignof(void*)`.
//!
//! For convenience, the `cuda::interface` template lets you specify the size
//! and alignment of the in-situ buffer directly:
//!
//! \code{.cpp}
//! template <class...>
//! struct icat
//!   : cuda::__basic_interface<icat,
//!                      cuda::__extends<cuda::__icopyable<>>,
//!                      2 * sizeof(void*),  // in-situ buffer size
//!                      alignof(void*)>     // in-situ buffer alignment
//! {
//!   ... as before ...
//! \endcode
//!
//! An object of type `T` will be stored in-situ in a `__basic_any<I>` object if
//! the following conditions are met:
//!
//! - `I::size >= sizeof(T)`, and
//! - `I::align % alignof(T) == 0`, and
//! - `std::is_nothrow_move_constructible_v<T> == true`
//!
//! If any of these conditions are not met, the object will be stored on the
//! heap. When such a `__basic_any` object is copied, a new `T` object will be
//! dynamically allocated and initialized with the old object. In other words,
//! `__basic_any` objects have value semantics regardless of how the wrapped
//! object is stored. All `__basic_any` objects are nothrow move constructible.
//!
//! `__basic_any` pointer and reference types (`__basic_any<I*>` and
//! `__basic_any<I&>`) ignore the interface's `::size` and `::align` members.
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
//! `cuda::interface` template as described above. The members of the interface
//! stub functions, aka thunks, that both define the interface and dispatch to
//! the type-erased pseudo-virtual overrides. You use the `cuda::__virtcall`
//! function to dispatch to the pseudo-virtuals, using the member pointer of the
//! thunk as a key for dispatching to the correct function.
//!
//! The following example shows how to define an interface that will give us the
//! behavior of `std::function`. The interface itself needs to be parameterized
//! by the function signature, which involves some contortions:
//!
//! \code{.cpp}
//! // We use an outer template to bind the function signature to the interface.
//! template <class Signature>
//! struct _with_signature;
//!
//! template <class Ret, class... Args>
//! struct _with_signature<Ret(Args...)>
//! {
//!   // The actual interface is a variadic class template that inherits from
//!   // `cuda::interface` and contains the thunks.
//!   template <class...>
//!   struct _ifuntion
//!     : cuda::__basic_interface<_ifunction, cuda::__extends<cuda::__icopyable<>>>
//!   {
//!     Ret operator()(Args... args) const
//!     {
//!       return cuda::__virtcall<&_ifunction::operator()>(this, std::forward<Args>(args)...);
//!     }
//!
//!     template <class F>
//!     using overrides = cuda::__overrides_for<F, static_cast<Ret (F::*)(Args...)
//!     const>(&F::operator())>;
//!   };
//! };
//!
//! template <class Signature>
//! using ifunction = _with_signature<Signature>::template _ifunction<>;
//!
//! template <class Signature>
//! using any_function = cuda::__basic_any<ifunction<Signature>>;
//! \endcode
//!
//! With the `any_function` template defined above, we can now store any callable
//! object that has the same signature as the `Signature` template parameter.
//!
//! \code{.cpp}
//! any_function<int(int, int)> fn = std::plus<int>{};
//! assert(fn(1, 2) == 3);
//! \endcode
//!
//! Within a thunk of an interface `I<>`, the `this` pointer has type
//! `I<J>*`, where `J` is the interface used to parameterize the `__basic_any`
//! object. For example, imagine an interface `ialley_cat` that extends `icat`.
//! `__basic_any<ialley_cat<>>` inherits publicly from `icat<ialley_cat<>>`, which
//! will be the type of the `this` pointer in the `icat<...>::meow()` thunk.
//!
//! Sometimes it is necessary to access the full `__basic_any` object from within
//! the thunk. You can do this by passing the `this` pointer to the
//! `cuda::__basic_any_from` function, which will return a pointer to the full
//! `__basic_any` object; a `__basic_any<ialley_cat<>>*` in this case.
//!
//! ## Using Free Functions As Overrides
//!
//! So far, we have only seen how to bind member functions to an interface. But
//! what if we want to bind a free function? The `cuda::__overrides_for` template can
//! bind free functions to an interface as well. Here is an example that uses a
//! free function, `_equal_to<T>`, to define the `__iequality_comparable`
//! interface:
//!
//! \code{.cpp}
//! template <class T>
//! auto _equal_to(T const& lhs, std::type_info const& rhs_type, void const* rhs)
//!   -> decltype(lhs == lhs) // SFINAE out if `lhs == lhs` is ill-formed
//! {
//!   return (rhs_type == typeid(T)) ? (lhs == *static_cast<T const*>(rhs)) : false;
//! }
//!
//! template <class...>
//! struct __iequality_comparable : cuda::__basic_interface<__iequality_comparable>
//! {
//!   friend bool operator==(__iequality_comparable const& lhs, __iequality_comparable const& rhs)
//!   {
//!     // downcast the `rhs` object to the correct specialization of `__basic_any`:
//!     auto const& rhs_any  = cuda::__basic_any_from(rhs);
//!     // get the object pointer from the `rhs` object:
//!     void const* rhs_optr = cuda::__any_cast<void const>(&rhs_any);
//!     // dispatch to the `_equal_to` override:
//!     return cuda::__virtcall<&_equal_to<__iequality_comparable>>(this, rhs_any.type(), rhs_optr);
//!   }
//!
//!   friend bool operator!=(__iequality_comparable const& lhs, __iequality_comparable const& rhs)
//!   {
//!     return !(lhs == rhs);
//!   }
//!
//!   template <class T>
//!   using overrides = cuda::__overrides_for<T, &_equal_to<T>>;
//! };
//! \endcode
//!
//! ## Implementation Details
//!
//! For an interface `I<>`, let `IBases<>...` be the (flattened, uniqued) list
//! of interfaces that `I<>` extends, either directly or indirectly.
//! `__basic_any<I<>>` inherits publicly from `I<I<>>` and `IBases<I<>>...`.
//! Therefore, the members of `__basic_any<I<>>` are the thunks of `I<>`
//! and all the interfaces that `I<>` extends.
//!
//! The `__basic_any<I<>>` type looks roughly like this:
//!
//! \code{.cpp}
//! template <template <class...> class I>
//! struct _CCCL_TYPE_VISIBILITY_DEFAULT __basic_any<I<>> : I<I<>>, IBases<I<>>...
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
//! \endcode
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
//! \code{.cpp}
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
//! \endcode
//!
//! The `__rtti_ex` struct contains a map from interface typeid's to the pointer
//! to the vtable for that interface. This map is used for `dynamic_cast`-like
//! functionality. `__rtti_ex` is derived from a `__rtti` class that has
//! metadata about the type-erased object like its typeid, size, and alignment.
//!
//! `__vtable_for<I<>>` names a struct that contains function pointers --
//! "overrides" -- for the pseudo-virtual functions of `I<>`. It is actually an
//! alias for a nested struct of the `cuda::__overrides_for` template. The actual vtable
//! struct looks like this:
//!
//! \code{.cpp}
//! template <class InterfaceOrObject, auto... Mbrs>
//! struct __overrides_for
//! {
//!   // InterfaceOrObject is either the type of the unspecialized interface or
//!   // the type of an object that implements the interface and that is being
//!   // type-erased.
//!   struct __vtable
//!     : __rtti_base         // described below
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
//!     __rtti_base const* __vptr_map[{count-of-I's-bases}];
//!   };
//! };
//! \endcode
//!
//! Here is the one trick that makes the whole of `__basic_any` work:
//! `__vtable_for<I<>>` is an alias for `I<>::overrides<I<>>::__vtable`. For the
//! `icat` interface defined above, the `icat<>::overrides` alias looks like this:
//!
//! \code{.cpp}
//! template <class T>
//! using overrides = cuda::__overrides_for<T, &T::meow>;
//! \endcode
//!
//! So `__vtable_for<icat<>>` is an alias for `cuda::__overrides_for<icat<>,
//! &icat<>::meow>::__vtable`. `icat<>::meow` is the thunk in the `icat<>`
//! interface.
//!
//! We can now describe how pseudo-virtual dispatch works. Given an object `cat`
//! of type `__basic_any<ialley_cat<>>` that we defined above, a call of
//! `cat.meow()` does the following:
//!
//! - Within the `meow` thunk, `this` has type `icat<ialley_cat<>>*`.
//!   The thunk calls `cuda::__virtcall<&icat::meow>(this)`.
//!
//! - `&icat::meow` is a pointer to the `icat<ialley_cat<>>::meow` thunk. The
//!   `__virtcall` infers from the `this` argument that the unspecialized
//!   interface is `icat<>`. `__virtcall` maps the `&icat<ialley_cat<>>::meow`
//!   member pointer to the `&icat<>::meow` member pointer by building a map
//!   from `icat<ialley_cat<>>::overrides<icat<ialley_cat<>>` to
//!   `icat<>::overrides<icat<>>`.
//!
//! - Having established that the override for `icat<>::meow` is the one that
//!   should be called, `__virtcall` calls
//!   `__basic_any_from({the-this-ptr-from-above})` to obtain a pointer to the
//!   full `__basic_any<ialley_cat<>>` object. From there, it pulls out the vtable
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
//! \code{.cpp}
//! template <auto Mbr, class = decltype(Mbr)>
//! struct __virtual_fn;
//!
//! template <auto Mbr, class R, class C, class... Args>
//! struct __virtual_fn<Mbr, R (C::*)(Args...)>
//! {
//!   using __function_t = R(*)(void*, Args...);
//!   __function_t __fn_;
//! };
//! \endcode
//!
//! When a `__basic_any<icat<>>` object is constructed from an object `fluffy` of
//! type `siamese`, a global constexpr object of type `__vtable<icat<>>` is
//! instantiated and initialized with `__tag<siamese>{}` (where `__tag` is an
//! empty struct). That will cause the base sub-object `__vtable_for<icat<>>` --
//! aka, `icat<>::overrides<icat<>>::__vtable`, aka `cuda::__overrides_for<icat<>,
//! &icat<>::meow>::__vtable` -- to be initialized with an object of type
//! `icat<>::overrides<siamese>`, aka `cuda::__overrides_for<siamese, &siamese::meow>`.
//! And that causes `__virtual_fn<&icat<>::meow>` to be initialized with a
//! pointer to a function that casts its `void*` argument to `siamese*` and
//! dispatches to its `meow` member function. So that's how the vtable gets
//! populated with the correct function pointers. In the `__basic_any`
//! constructor, a pointer to that vtable is saved in its `__vptr_` member.

// IWYU pragma: begin_exports
#include <cuda/__utility/__basic_any/any_cast.h>
#include <cuda/__utility/__basic_any/basic_any_from.h>
#include <cuda/__utility/__basic_any/basic_any_ptr.h>
#include <cuda/__utility/__basic_any/basic_any_ref.h>
#include <cuda/__utility/__basic_any/basic_any_value.h>
#include <cuda/__utility/__basic_any/conversions.h>
#include <cuda/__utility/__basic_any/dynamic_any_cast.h>
#include <cuda/__utility/__basic_any/interfaces.h>
#include <cuda/__utility/__basic_any/iset.h>
#include <cuda/__utility/__basic_any/overrides.h>
#include <cuda/__utility/__basic_any/rtti.h>
#include <cuda/__utility/__basic_any/semiregular.h>
#include <cuda/__utility/__basic_any/storage.h>
#include <cuda/__utility/__basic_any/tagged_ptr.h>
#include <cuda/__utility/__basic_any/virtcall.h>
#include <cuda/__utility/__basic_any/virtual_functions.h>
#include <cuda/__utility/__basic_any/virtual_tables.h>
// IWYU pragma: end_exports

#endif // _CUDA___UTILITY_BASIC_ANY_H
