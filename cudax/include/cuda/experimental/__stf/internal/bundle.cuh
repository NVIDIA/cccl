//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 *
 * @brief Bundles: non-owning groups of logical data usable as a single task dependency
 *
 * A `bundle` ties several logical data together behind one object (e.g. the
 * three arrays of a CSR matrix) so that tasks can depend on the whole group
 * with a single argument, while every constituent ("field") remains an
 * ordinary logical data usable on its own. A bundle owns no data and no
 * dependency-tracking state: it merely holds handle copies, and a bundle
 * dependency expands into one ordinary dependency per field. The lambda of a
 * task (or parallel_for, ...) receives one tuple of per-field views per
 * bundle dependency instead of one view per field.
 *
 * Fields declared `constant` have a read-only ceiling: whole-bundle access
 * modes distribute to each field as the strongest mode the field admits
 * (`rw()` on a bundle with a constant field passes that field's view as
 * read-only), and their views are const-qualified in every spelling.
 */

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

#include <cuda/std/tuple>
#include <cuda/std/type_traits>
#include <cuda/std/utility>

#include <cuda/experimental/__stf/internal/constants.cuh>
#include <cuda/experimental/__stf/utility/core.cuh>

#include <string>
#include <tuple>
#include <utility>

namespace cuda::experimental::stf
{
// Bundles only ever name these inside templates; the definitions come from
// logical_data.cuh / data_interface.cuh at instantiation time. Keeping this
// header lightweight lets backend_ctx.cuh include it without cycles.
template <typename T>
class logical_data;

template <typename T>
class shape_of;

class void_interface;

template <typename T, typename reduce_op, bool initialize>
class task_dep;

/**
 * @brief Trait tag marking a bundle field as constant (read-only ceiling).
 *
 * A constant field only ever admits read access: whole-bundle modes clamp to
 * `read` for it, and its view is const-qualified in every task.
 */
struct constant
{};

/**
 * @brief Describes one field of a bundle: the instance type and optional traits.
 *
 * @tparam T the instance type of the underlying logical data (e.g. `slice<double>`)
 * @tparam Traits optional traits (`constant_t`)
 */
template <typename T, typename... Traits>
struct field
{
  using type                        = T;
  static constexpr bool is_constant = (::cuda::std::is_same_v<Traits, constant> || ... || false);
};

namespace reserved
{
template <typename T>
struct is_field : ::cuda::std::false_type
{};

template <typename T, typename... Traits>
struct is_field<field<T, Traits...>> : ::cuda::std::true_type
{};
} // end namespace reserved

/**
 * @brief A group of ordinary task dependencies submitted as a single argument.
 *
 * Produced by `bundle::read()/rw()/write()`; consumed by the context task
 * constructs, which expand it into its per-field dependencies and regroup the
 * corresponding views into a single tuple argument for the user lambda.
 */
template <typename... LeafDeps>
class bundle_dep
{
public:
  static constexpr size_t arity = sizeof...(LeafDeps);

  explicit bundle_dep(LeafDeps... d)
      : deps(mv(d)...)
  {}

  ::std::tuple<LeafDeps...> deps;
};

namespace reserved
{
template <typename T>
struct is_bundle_dep : ::cuda::std::false_type
{};

template <typename... L>
struct is_bundle_dep<bundle_dep<L...>> : ::cuda::std::true_type
{};

template <typename T>
inline constexpr bool is_bundle_dep_v = is_bundle_dep<::cuda::std::remove_cvref_t<T>>::value;

//! True when at least one argument of a dependency list is a bundle_dep
template <typename... Args>
inline constexpr bool any_bundle_dep_v = (is_bundle_dep_v<Args> || ... || false);

//! Number of user-visible lambda arguments one flat dependency produces:
//! void_interface (token) dependencies are filtered out of the argument list
//! by the constructs and thus produce none.
template <typename T>
struct dep_visible_arity : ::cuda::std::integral_constant<size_t, 1>
{};

template <typename reduce_op, bool initialize>
struct dep_visible_arity<task_dep<void_interface, reduce_op, initialize>> : ::cuda::std::integral_constant<size_t, 0>
{};

//! Number of user-visible lambda arguments one submitted argument produces
template <typename T>
struct visible_slot_arity
    : ::cuda::std::integral_constant<size_t, dep_visible_arity<::cuda::std::remove_cvref_t<T>>::value>
{};

template <typename... L>
struct visible_slot_arity<bundle_dep<L...>>
    : ::cuda::std::integral_constant<size_t, (dep_visible_arity<::cuda::std::remove_cvref_t<L>>::value + ... + 0)>
{};

//! Expand one submitted dependency argument into a tuple of flat dependencies
template <typename T>
auto as_dep_tuple(T t)
{
  if constexpr (is_bundle_dep_v<T>)
  {
    return mv(t.deps);
  }
  else
  {
    return ::std::make_tuple(mv(t));
  }
}
} // end namespace reserved

/**
 * @brief A non-owning group of logical data described by a list of `field`s.
 *
 * Bundles introduce no new ownership domain: they hold ordinary (refcounted)
 * logical data handles. Every field remains a first-class logical data
 * retrievable with `get_field<Idx>()`, so bundle-level and per-field task
 * dependencies interoperate (they meet at the same logical data).
 *
 * @tparam Fields `field<T, Traits...>` descriptors, in canonical order
 */
template <typename... Fields>
class bundle
{
  static_assert((reserved::is_field<Fields>::value && ...), "bundle<...> parameters must be field<...> descriptors");

public:
  static constexpr size_t n_fields = sizeof...(Fields);

  template <size_t Idx>
  using field_at = ::std::tuple_element_t<Idx, ::std::tuple<Fields...>>;

  /** @brief Adopt existing logical data as the bundle's fields (no context needed) */
  explicit bundle(logical_data<typename Fields::type>... h)
      : handles(mv(h)...)
  {}

  /** @brief Create fresh logical data from shapes, then behave like an adopting bundle */
  template <typename ctx_t>
  bundle(ctx_t& ctx, shape_of<typename Fields::type>... shapes)
      : handles(ctx.logical_data(mv(shapes))...)
  {}

  /** @brief Access field `Idx` as its plain, first-class logical data */
  template <size_t Idx>
  auto& get_field()
  {
    return ::std::get<Idx>(handles);
  }

  template <size_t Idx>
  const auto& get_field() const
  {
    return ::std::get<Idx>(handles);
  }

  /** @brief Depend on every field with read access */
  auto read()
  {
    return dep_impl<access_mode::read>(::std::index_sequence_for<Fields...>());
  }

  /**
   * @brief Depend on the bundle with read-write access.
   *
   * Distributes per field as the strongest admitted mode: mutable fields get
   * `rw`, constant fields clamp to `read` (their views stay const-qualified).
   */
  auto rw()
  {
    return dep_impl<access_mode::rw>(::std::index_sequence_for<Fields...>());
  }

  /**
   * @brief Depend on the bundle with write access.
   *
   * Mutable fields get `write` (previous content is discarded, not fetched);
   * constant fields clamp to `read` — their content is still fetched, since a
   * writer typically needs the constant fields to interpret what it writes.
   */
  auto write()
  {
    return dep_impl<access_mode::write>(::std::index_sequence_for<Fields...>());
  }

  /** @brief Name unnamed fields "prefix.<index>" (never clobbers an existing symbol) */
  bundle& set_symbol(const ::std::string& prefix)
  {
    set_symbol_impl(prefix, ::std::index_sequence_for<Fields...>());
    return *this;
  }

private:
  template <access_mode M, size_t Idx>
  auto leaf_dep()
  {
    auto& ld = ::std::get<Idx>(handles);
    if constexpr (M == access_mode::read || field_at<Idx>::is_constant)
    {
      return ld.read();
    }
    else if constexpr (M == access_mode::write)
    {
      return ld.write();
    }
    else
    {
      return ld.rw();
    }
  }

  template <access_mode M, size_t... Is>
  auto dep_impl(::std::index_sequence<Is...>)
  {
    return bundle_dep<decltype(this->template leaf_dep<M, Is>())...>(this->template leaf_dep<M, Is>()...);
  }

  template <size_t... Is>
  void set_symbol_impl(const ::std::string& prefix, ::std::index_sequence<Is...>)
  {
    ((::std::get<Is>(handles).get_symbol().empty()
        ? void(::std::get<Is>(handles).set_symbol(prefix + "." + ::std::to_string(Is)))
        : void()),
     ...);
  }

  ::std::tuple<logical_data<typename Fields::type>...> handles;
};

namespace reserved
{
/**
 * @brief Callable adapter regrouping the flat per-dependency views of an
 * expanded dependency list back into one tuple argument per bundle.
 *
 * The wrapped construct passes `(leading..., view0, view1, ...)` where the
 * trailing `total_flat` arguments are the flat per-dependency views in
 * submission order and the leading arguments are construct-specific (a
 * stream, shape coordinates, a thread hierarchy, ...). Slots of arity 1 are
 * forwarded untouched (preserving references); slots of arity k become a
 * `::cuda::std::tuple` of the k views, by value.
 *
 * This is a named functor (not a lambda) so it can wrap extended device
 * lambdas, and its call operator is SFINAE-constrained to the wrapped
 * function's own applicability so that constructs probing several calling
 * conventions keep working.
 */
template <typename Fun, typename AritySeq>
struct bundle_arg_adapter;

template <typename Fun, size_t... Arities>
struct bundle_arg_adapter<Fun, ::std::index_sequence<Arities...>>
{
  Fun f;

  static constexpr size_t n_slots    = sizeof...(Arities);
  static constexpr size_t total_flat = (Arities + ... + 0);

  static constexpr size_t arity_of(size_t slot)
  {
    constexpr size_t a[] = {Arities...};
    return a[slot];
  }

  static constexpr size_t offset_of(size_t slot)
  {
    constexpr size_t a[] = {Arities...};
    size_t o             = 0;
    for (size_t i = 0; i < slot; ++i)
    {
      o += a[i];
    }
    return o;
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <size_t Off, typename Tup, size_t... Ks>
  _CCCL_HOST_DEVICE static auto make_view_tuple(Tup& tup, ::std::index_sequence<Ks...>)
  {
    return ::cuda::std::tuple<
      ::cuda::std::remove_cvref_t<::cuda::std::tuple_element_t<Off + Ks, ::cuda::std::remove_cvref_t<Tup>>>...>(
      ::cuda::std::get<Off + Ks>(tup)...);
  }

  // One argument-tuple piece per slot, later flattened with tuple_cat: empty
  // for zero-arity slots (token dependencies produce no user argument), a
  // single forwarded reference for arity-1 slots (a reduction accumulator
  // must remain a reference), a single by-value tuple of views for bundles.
  _CCCL_EXEC_CHECK_DISABLE
  template <size_t NLead, size_t Slot, typename Tup>
  _CCCL_HOST_DEVICE static auto slot_piece(Tup& tup)
  {
    constexpr size_t off = NLead + offset_of(Slot);
    if constexpr (arity_of(Slot) == 0)
    {
      return ::cuda::std::tuple<>();
    }
    else if constexpr (arity_of(Slot) == 1)
    {
      return ::cuda::std::forward_as_tuple(
        static_cast<::cuda::std::tuple_element_t<off, ::cuda::std::remove_cvref_t<Tup>>&&>(::cuda::std::get<off>(tup)));
    }
    else
    {
      return ::cuda::std::make_tuple(make_view_tuple<off>(tup, ::std::make_index_sequence<arity_of(Slot)>()));
    }
  }

  // Type-level mirror of slot_piece, used to constrain operator()
  template <size_t NLead, size_t Slot, typename Tup>
  using slot_piece_t = decltype(slot_piece<NLead, Slot>(::cuda::std::declval<Tup&>()));

  template <typename ArgsTup, size_t... Is>
  static constexpr bool apply_invocable(::std::index_sequence<Is...>)
  {
    return ::cuda::std::is_invocable_v<Fun&, ::cuda::std::tuple_element_t<Is, ArgsTup>...>;
  }

  template <typename Tup, size_t NLead, size_t... LeadIs, size_t... SlotIs>
  static constexpr bool invocable_impl(::std::index_sequence<LeadIs...>, ::std::index_sequence<SlotIs...>)
  {
    using args_tuple = decltype(::cuda::std::tuple_cat(
      ::cuda::std::declval<::cuda::std::tuple<::cuda::std::tuple_element_t<LeadIs, Tup>...>>(),
      ::cuda::std::declval<slot_piece_t<NLead, SlotIs, Tup>>()...));
    return apply_invocable<args_tuple>(::std::make_index_sequence<::cuda::std::tuple_size_v<args_tuple>>());
  }

  template <typename... Args>
  static constexpr bool applicable()
  {
    if constexpr (sizeof...(Args) < total_flat)
    {
      return false;
    }
    else
    {
      constexpr size_t n_lead = sizeof...(Args) - total_flat;
      return invocable_impl<::cuda::std::tuple<Args&&...>, n_lead>(
        ::std::make_index_sequence<n_lead>(), ::std::make_index_sequence<n_slots>());
    }
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <typename Tup, size_t... LeadIs, size_t... SlotIs>
  _CCCL_HOST_DEVICE decltype(auto) call(Tup&& tup, ::std::index_sequence<LeadIs...>, ::std::index_sequence<SlotIs...>)
  {
    constexpr size_t n_lead = sizeof...(LeadIs);
    return ::cuda::std::apply(
      f,
      ::cuda::std::tuple_cat(::cuda::std::forward_as_tuple(::cuda::std::get<LeadIs>(mv(tup))...),
                             slot_piece<n_lead, SlotIs>(tup)...));
  }

  _CCCL_EXEC_CHECK_DISABLE
  template <typename... Args, typename = ::cuda::std::enable_if_t<applicable<Args...>()>>
  _CCCL_HOST_DEVICE decltype(auto) operator()(Args&&... args)
  {
    constexpr size_t n_lead = sizeof...(Args) - total_flat;
    return call(::cuda::std::forward_as_tuple(::cuda::std::forward<Args>(args)...),
                ::std::make_index_sequence<n_lead>(),
                ::std::make_index_sequence<n_slots>());
  }
};

//! Classification helper: the executable "kind" of an adapter (extended
//! device lambda, host-device lambda, plain host callable) is that of the
//! user function it wraps. Identity for every other callable.
template <typename F>
struct bundle_inner_fun
{
  using type = F;
};

template <typename F, typename A>
struct bundle_inner_fun<bundle_arg_adapter<F, A>>
{
  using type = typename bundle_inner_fun<F>::type;
};

template <typename F>
using bundle_inner_fun_t = typename bundle_inner_fun<::cuda::std::remove_cvref_t<F>>::type;

/**
 * @brief Wraps a task-like construct scope so that `operator->*` regroups
 * bundle views into single tuple arguments before invoking the user function.
 */
template <typename Scope, typename AritySeq>
class bundle_scope
{
public:
  explicit bundle_scope(Scope s)
      : inner(mv(s))
  {}

  bundle_scope& set_symbol(::std::string s) &
  {
    inner.set_symbol(mv(s));
    return *this;
  }

  bundle_scope&& set_symbol(::std::string s) &&
  {
    inner.set_symbol(mv(s));
    return mv(*this);
  }

  Scope& get_scope()
  {
    return inner;
  }

  template <typename Fun>
  decltype(auto) operator->*(Fun&& f)
  {
    return mv(inner)->*bundle_arg_adapter<::cuda::std::remove_cvref_t<Fun>, AritySeq>{::cuda::std::forward<Fun>(f)};
  }

private:
  Scope inner;
};

/**
 * @brief Expand a dependency list that contains bundle_dep arguments into
 * flat dependencies, invoke `make_inner` on them, and wrap the resulting
 * scope for view regrouping.
 */
template <typename MakeInner, typename... Args>
auto make_bundle_scope(MakeInner&& make_inner, Args... args)
{
  using arities = ::std::index_sequence<visible_slot_arity<::cuda::std::remove_cvref_t<Args>>::value...>;
  auto flat     = ::std::tuple_cat(as_dep_tuple(mv(args))...);
  auto inner    = ::std::apply(::cuda::std::forward<MakeInner>(make_inner), mv(flat));
  return bundle_scope<decltype(inner), arities>(mv(inner));
}
} // end namespace reserved
} // end namespace cuda::experimental::stf
