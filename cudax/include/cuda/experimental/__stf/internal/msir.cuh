//===----------------------------------------------------------------------===//
//
// Part of CUDASTF in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2022-2024 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#pragma once

#include <cuda/__cccl_config>

#if defined(_CCCL_IMPLICIT_SYSTEM_HEADER_GCC)
#  pragma GCC system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_CLANG)
#  pragma clang system_header
#elif defined(_CCCL_IMPLICIT_SYSTEM_HEADER_MSVC)
#  pragma system_header
#endif // no system header

/*
 *   We here define the protocol to keep data copies up to date
 *   Task dependencies are supposed to be enforced by the STF model, so this is
 *   intended to implement the required data movements/allocations.
 *
 *   M : msir_state_id::modified
 *   S : msir_state_id::shared
 *   I : msir_state_id::invalid
 *   R : msir_state_id::reduction
 */

#include <cuda/experimental/__stf/internal/async_prereq.cuh>

namespace cuda::experimental::stf
{
namespace reserved
{
enum class msir_state_id
{
  invalid,
  modified,
  shared,
  reduction,
};

inline ::std::string status_to_string(msir_state_id status)
{
  switch (status)
  {
    case msir_state_id::modified:
      return "msir_state_id::modified";
    case msir_state_id::shared:
      return "msir_state_id::shared";
    case msir_state_id::invalid:
      return "msir_state_id::invalid";
    case msir_state_id::reduction:
      return "REDUCTION";
  }

  return "UNKNOWN";
}

inline char status_to_char(msir_state_id status)
{
  switch (status)
  {
    case msir_state_id::modified:
      return 'M';
    case msir_state_id::shared:
      return 'S';
    case msir_state_id::invalid:
      return 'I';
    case msir_state_id::reduction:
      return 'R';
  }

  return 'U';
}

/*
 * Generic interface that does not suppose how memory places are organized.
 * Could be a concept in the future. Passive documentation for now.
 */
/*
template <typename memory_place_interface> class msi_state {
public:
    // Returns the state of a piece of data at a specific place
    int shape(memory_place_interface place, class event_list **msi_prereq);

    // Sets the status for a specific place
    void set_state(memory_place_interface place, int state, class event_list *new_msi_prereq);

    // Update data status when accessing data at the specified place with the specified access type
    // Returns prereq_out
    class event_list *update_state(memory_place_interface place, int access_mode, class event_list *prereq_in);

    // Find a valid copy to move a piece of data to dst_place. Returned value is a possible source place
    // TODO perhaps we should return a list
    // TODO perhaps this should return a pair of (place+prereq)
    memory_place_interface find_source_place(memory_place_interface dst_place);

private:
};
*/

class per_data_instance_msi_state
{
public:
  per_data_instance_msi_state() {}

  ~per_data_instance_msi_state() {}

  msir_state_id get_msir() const
  {
    return msir;
  }

  void set_msir(msir_state_id _msir)
  {
    msir = _msir;
  }

  bool is_allocated() const
  {
    return allocated;
  }

  void set_allocated(bool _allocated)
  {
    allocated = _allocated;
  }

  const event_list& get_read_prereq() const
  {
    return read_prereq;
  }

  const event_list& get_write_prereq() const
  {
    return write_prereq;
  }

  void set_read_prereq(event_list prereq)
  {
    read_prereq = mv(prereq);
  }

  void set_write_prereq(event_list prereq)
  {
    write_prereq = mv(prereq);
  }

  /**
   * @brief Compute the maximum async prereq ID in this list (for both read and write accesses).
   */
  int max_prereq_id() const
  {
    int res = read_prereq.max_prereq_id();
    res     = ::std::max(res, write_prereq.max_prereq_id());
    return res;
  }

  template <typename T>
  void add_read_prereq(backend_ctx_untyped& bctx, T&& prereq)
  {
    read_prereq.merge(::std::forward<T>(prereq));

    if (read_prereq.size() > 16)
    {
      read_prereq.optimize(bctx);
    }
  }
  template <typename T>
  void add_write_prereq(backend_ctx_untyped& bctx, T&& prereq)
  {
    write_prereq.merge(::std::forward<T>(prereq));

    if (write_prereq.size() > 16)
    {
      write_prereq.optimize(bctx);
    }
  }

  void clear_read_prereq()
  {
    read_prereq.clear();
  }
  void clear_write_prereq()
  {
    write_prereq.clear();
  }

  size_t hash() const
  {
    return hash_all(allocated, (int) msir);
  }

private:
  // We need to fulfill these events __and those in read_prereq__ to modify the instance
  event_list write_prereq;

  // We need to fulfill these events to read the instance without modifying it
  event_list read_prereq;

  msir_state_id msir = msir_state_id::invalid; // MSIR = msir_state_id::modified, ...
  bool allocated     = false;
};
} // end namespace reserved

// Overload hash to compute the hash of a per_data_instance_msi_state
// class from the MSI and allocated states.
template <>
struct hash<reserved::per_data_instance_msi_state>
{
  ::std::size_t operator()(reserved::per_data_instance_msi_state const& s) const noexcept
  {
    return s.hash();
  }
};
} // namespace cuda::experimental::stf
