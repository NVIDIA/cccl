//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Implementation of the generic event class
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

#include <cuda/experimental/__stf/internal/dot.cuh>
#include <cuda/experimental/__stf/utility/handle.cuh>
#include <cuda/experimental/__stf/utility/memory.cuh>
#include <cuda/experimental/__stf/utility/nvtx.cuh>
#include <cuda/experimental/__stf/utility/scope_guard.cuh>
#include <cuda/experimental/__stf/utility/threads.cuh>
#include <cuda/experimental/__stf/utility/unique_id.cuh>

#include <algorithm>
#include <atomic>
#include <vector>

namespace cuda::experimental::stf
{
class event_impl;
using event = reserved::handle<event_impl>;

namespace reserved
{
using unique_id_t = unique_id<event>;

//!
//! Generates the next unique identifier for asynchronous prerequisite events in DOT.
//!
//! This function provides monotonically increasing unique IDs that are used to:
//! - Establish ordering relationships between events in the task graph
//! - Enable proper dependency tracking in asynchronous execution
//! - Support event comparison and sorting operations
//! - Facilitate debugging and visualization of event dependencies
//!
//! \return A unique integer identifier that is guaranteed to be larger than any previously returned ID
//!
inline int get_next_prereq_unique_id()
{
  return int(unique_id<event>::next_id());
}

using event_vector = small_vector<event, 7>;
static_assert(sizeof(event_vector) == 120);
} // namespace reserved

class backend_ctx_untyped;
class event_list;

/**
 * @brief This is the base event type. It defines an abstract object which
 * represents that an asynchronous event happened.
 *
 * It is possible to insert a dependency between two events provided the
 * dependency follows the chronological order of creation: an event E2 created
 * after an event E1 cannot depend on E1.
 *
 * Inserting a dependency between two events is supposed to be a non-blocking
 * operation, but the actual event implementation is allowed to block
 * temporarily.
 */
class event_impl
{
public:
  /**
   * @brief Deleted copy constructor to prevent copying and ensure unique event identifiers.
   */
  event_impl(const event_impl&) = delete;

  /**
   * @brief Deleted copy assignment operator to prevent copying and ensure unique event identifiers.
   */
  event_impl& operator=(const event_impl&) = delete;

  /**
   * @brief Virtual destructor to support polymorphic deletion.
   */
  virtual ~event_impl() = default;

  /**
   * @brief Default constructor.
   */
  event_impl() = default;

  /**
   * @brief Compares this event with another event for equality based on their unique identifiers.
   * @param e The event to compare with.
   * @return True if the events have the same unique identifier, false otherwise.
   */
  bool operator==(const event_impl& e) const
  {
    return unique_prereq_id == e.unique_prereq_id;
  }

  /**
   * @brief Compares this event with another event for ordering based on their unique identifiers.
   * @param e The event to compare with.
   * @return True if this event's unique identifier is less than the other event's identifier, false otherwise.
   */
  bool operator<(const event_impl& e) const
  {
    return unique_prereq_id < e.unique_prereq_id;
  }

  /**
   * @brief Sets a symbolic name for the event, useful for debugging or tracing.
   * @param s The symbolic name to associate with this event.
   */
  void set_symbol_with_dot(reserved::per_ctx_dot& dot, ::std::string s)
  {
    symbol = mv(s);
    if (dot.is_tracing())
    {
      dot.add_prereq_vertex(symbol, unique_prereq_id);
    }
  }

  /**
   * @brief Sets a symbolic name for the event, useful for debugging or tracing.
   * @param s The symbolic name to associate with this event.
   */
  template <typename context_t>
  void set_symbol(context_t& ctx, ::std::string s)
  {
    set_symbol_with_dot(*ctx.get_dot(), mv(s));
  }

  /**
   * @brief Optionally simplifies the event vector to remove redundant entries.
   * @param unused A vector of events potentially containing redundant entries.
   * @return True if redundant entries were removed and further uniqueness processing is unnecessary, false otherwise.
   * @note This function provides a hook for derived classes to implement optimization strategies.
   */
  virtual bool factorize(backend_ctx_untyped&, reserved::event_vector&)
  {
    return false;
  }

  // stream then depends on the list of events
  virtual void sync_with_stream(backend_ctx_untyped&, event_list&, cudaStream_t) const
  {
    fprintf(stderr, "Unsupported synchronization with stream.\n");
    abort();
  }

  // return stream then depends on the list of events
  virtual event_list from_stream(backend_ctx_untyped&, cudaStream_t) const;

  /**
   * @brief Retrieves the symbolic name of the event.
   * @return The symbolic name of the event. Generates a default name if none was set.
   */
  const ::std::string& get_symbol() const
  {
    if (symbol.empty())
    {
      symbol = "event " + ::std::to_string(unique_prereq_id);
    }
    return symbol;
  }

  /// A unique identifier for the event, used to ensure proper event ordering.
  mutable reserved::unique_id_t unique_prereq_id;

  ::std::atomic<int> outbound_deps = 0;

protected:
  /// The symbolic name associated with the event, mutable to allow lazy initialization.
  mutable ::std::string symbol;
};

/**
 * @brief List of events.
 *
 * Many CUDASTF routines take event lists as asynchronous prerequisites, and return a list of prerequisites.
 */
class event_list
{
  using event_vector = reserved::event_vector;

public:
  event_list()                             = default;
  event_list(const event_list&)            = default;
  event_list(event_list&&)                 = default;
  event_list& operator=(const event_list&) = default;
  event_list& operator=(event_list&&)      = default;

  /// Create a list from a single event
  event_list(event e)
      : payload{mv(e)}
  {}

  event_list(event_vector& payload_)
      : payload(mv(payload_))
      , optimized(false)
  {}

  /// Add an event to an existing list
  void add(event e)
  {
    payload.push_back(mv(e));
    optimized = payload.size() == 1;
  }

  /// Optimize the list to remove redundant entries which are either
  /// identical events, or events which are implicit from other events in the
  /// list.
  void optimize(backend_ctx_untyped& bctx)
  {
    // No need to remove duplicates on a list that was already sanitized,
    // and that has not been modified since
    if (optimized)
    {
      return;
    }

    _CCCL_ASSERT(!payload.empty(), "internal error");

    // nvtx_range r("optimize");

    // This is a list of shared_ptr to events, we want to sort by events
    // ID, not by addresses of the ptr
    ::std::sort(payload.begin(), payload.end(), [](auto& a, auto& b) {
      return *a < *b;
    });

    // All items will have the same (derived) event type as the type of the front element.
    // If the type of the event does not implement a factorize method, a
    // false value is returned (eg. with cudaGraphs)
    bool factorized = payload.front()->factorize(bctx, payload);

    if (!factorized)
    {
      // Note that the list was already sorted above so we can call unique directly
      auto new_end = ::std::unique(payload.begin(), payload.end(), [](auto& a, auto& b) {
        return *a == *b;
      });
      // Erase the "undefined" elements at the end of the container
      payload.erase(new_end, payload.end());
    }

    optimized = true;
  }

  // Introduce a dependency between the event list and the CUDA stream, this
  // relies on the sync_with_stream virtual function of the event list.
  void sync_with_stream(backend_ctx_untyped& bctx, cudaStream_t stream)
  {
    if (payload.size() > 0)
    {
      payload.front()->sync_with_stream(bctx, *this, stream);
    }
  }

  // id_to can be the id of a task or another prereq
  void dot_declare_prereqs(reserved::per_ctx_dot& dot, int id_to, reserved::edge_type style = reserved::edge_type::plain)
  {
    if (!dot.is_tracing_prereqs())
    {
      return;
    }

    for (auto& e : payload)
    {
      dot.add_edge(e->unique_prereq_id, id_to, style);
    }
  }

  // id_from can be the id of a task or another prereq
  void dot_declare_prereqs_from(
    reserved::per_ctx_dot& dot, int id_from, reserved::edge_type style = reserved::edge_type::plain) const
  {
    if (!dot.is_tracing_prereqs())
    {
      return;
    }

    for (auto& e : payload)
    {
      dot.add_edge(id_from, e->unique_prereq_id, style);
    }
  }

  /// Concatenate the content of a list into another list
  ///
  /// This does not depend on the actual event type
  template <typename... Ts>
  void merge(Ts&&... events)
  {
    static_assert(sizeof...(events) > 0);

    // Special case move from a single event list
    if constexpr (sizeof...(Ts) == 1)
    {
      using First = ::std::tuple_element_t<0, ::std::tuple<Ts...>>;
      if constexpr (!::std::is_lvalue_reference_v<First>)
      {
        if (payload.empty())
        {
          // Disposable copy, move from the argument and we're all done
          *this = mv(events...);
          return;
        }
      }
    }

    // This will be the new size of payload with the new events in tow
    const size_t new_size = payload.size() + (... + events.size());

    // Attempt to find enough capacity in one of the added events
    each_in_pack(
      [&](auto&& event) {
        if constexpr (!::std::is_lvalue_reference_v<decltype(event)>)
        {
          if (event.payload.capacity() >= new_size && payload.capacity() < new_size)
          {
            // Awesome, we found enough capacity.
            // Swapping is fine because all elements will be merged and order is not important.
            event.payload.swap(payload);
          }
        }
      },
      ::std::forward<Ts>(events)...);

    if (payload.capacity() < new_size)
    {
      payload.reserve(::std::max(payload.capacity() * 2, new_size));
    }

    each_in_pack(
      [&](auto&& event) {
        if constexpr (::std::is_lvalue_reference_v<decltype(event)>)
        {
          // Simply append copies of elements
          payload.insert(payload.end(), event.begin(), event.end());
        }
        else
        {
          payload.insert(payload.end(),
                         ::std::make_move_iterator(event.payload.begin()),
                         ::std::make_move_iterator(event.payload.end()));
        }
      },
      ::std::forward<Ts>(events)...);

    assert(payload.size() == new_size);
    optimized = payload.size() <= 1;
  }

  template <typename T>
  event_list& operator+=(T&& rhs)
  {
    merge(::std::forward<T>(rhs));
    return *this;
  }

  /// Empty a list
  void clear()
  {
    payload.clear();
    optimized = true;
  }

  /// Get the number of events in the list
  size_t size() const
  {
    return payload.size();
  }

  // Display the content of the event list as a string
  ::std::string to_string() const
  {
    ::std::string result = "LIST OF EVENTS: ";
    if (payload.empty())
    {
      result += "(empty)\n";
      return result;
    }

    for (const auto& e : payload)
    {
      result += ::std::to_string(int(e->unique_prereq_id));
      result += '(';
      result += e->get_symbol();
      result += ") ";
    }
    result += "\n";

    return result;
  }

  auto begin()
  {
    return payload.begin();
  }
  auto end()
  {
    return payload.end();
  }
  auto begin() const
  {
    return payload.begin();
  }
  auto end() const
  {
    return payload.end();
  }

  // Computes the largest prereq id
  int max_prereq_id() const
  {
    int res = 0;
    for (const auto& e : payload)
    {
      res = ::std::max(res, int(e->unique_prereq_id));
    }
    return res;
  }

private:
  // ::std::vector<event> payload;
  event_vector payload;

  /// Indicates if some factorization or removal of duplicated entries was
  /// already performed on that list
  bool optimized = true;
};

_CCCL_DIAG_PUSH
_CCCL_DIAG_SUPPRESS_MSVC(4702) // unreachable code
inline event_list event_impl::from_stream(backend_ctx_untyped&, cudaStream_t) const
{
  fprintf(stderr, "Unsupported synchronization with stream.\n");
  abort();
  return event_list();
}
_CCCL_DIAG_POP

/**
 * @brief Introduce a dependency from all entries of an event list to an event.

 * Makes it so that `to` depends on all of `prereq_in`.
 */
template <typename context_t, typename some_event>
void join(context_t& ctx, some_event& to, event_list& prereq_in)
{
  bool typechecked = false;
  for (auto& item : prereq_in)
  {
    assert(item);
    some_event* from;
    if (!typechecked)
    {
      from = dynamic_cast<some_event*>(item.operator->());
      if (!from)
      {
        fprintf(stderr,
                "Internal error: cannot dynamically cast event \"%s\" from %s to %.*s.\n",
                item->get_symbol().c_str(),
                typeid(decltype(*item)).name(), // Change made here
                static_cast<int>(type_name<some_event>.size()),
                type_name<some_event>.data());
        abort();
      }
      typechecked = true;
    }
    else
    {
      from = static_cast<some_event*>(item.operator->());
    }
    to.insert_dep(ctx.async_resources(), *from);
    from->outbound_deps++;
  }

  // If we are making a DOT output for prereqs, we create a local list (it's easier to get the)
  auto& dot = *ctx.get_dot();
  if (dot.is_tracing_prereqs())
  {
    prereq_in.dot_declare_prereqs(dot, to.unique_prereq_id, reserved::edge_type::prereqs);
  }
}
} // namespace cuda::experimental::stf
