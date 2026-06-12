//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/experimental/__multi_gpu/concepts.h>

#include "concepts_common.cuh"

namespace
{
namespace cudax = ::cuda::experimental;
namespace types = cudax_multi_gpu_concepts;

// nvcc ignores [[maybe_unused]] entirely
_CCCL_BEGIN_NV_DIAG_SUPPRESS(177)

struct rank_convertible_to_int
{
  using native_handle_type = int;
  using group_token_type   = int;

  native_handle_type native_handle() noexcept;

  struct tag
  {
    operator int();
  };

  tag rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void recv_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct size_convertible_to_int
{
  using native_handle_type = int;
  using group_token_type   = int;

  native_handle_type native_handle() noexcept;

  struct tag
  {
    operator int();
  };

  cuda::std::int32_t rank() noexcept;
  tag size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void recv_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct empty
{};

struct no_native_handle_type
{
  using group_token_type = int;

  int native_handle() noexcept;
  ::cuda::std::int32_t rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void recv_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct native_handle_type_not_same : types::synchronous_communicator_model
{
  double native_handle() noexcept;
};

struct native_handle_type_throws : types::synchronous_communicator_model
{
  native_handle_type native_handle();
};

struct no_rank
{
  using native_handle_type = int;
  using group_token_type   = int;

  native_handle_type native_handle() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void recv_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct rank_doesnt_return_int : types::synchronous_communicator_model
{
  struct tag
  {};

  tag rank() noexcept;
};

struct size_doesnt_return_int : types::synchronous_communicator_model
{
  struct tag
  {};

  tag size() noexcept;
};

struct no_group_token_type
{
  using native_handle_type = int;

  native_handle_type native_handle() noexcept;
  ::cuda::std::int32_t rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  int group_token();

  template <class Tp>
  void send_sync(int&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
  template <class Tp>
  void recv_sync(int&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct group_token_doesnt_return_group_token : types::synchronous_communicator_model
{
  struct tag
  {};

  tag group_token() noexcept;
};

struct no_send_sync
{
  using native_handle_type = int;
  using group_token_type   = types::group_token;

  native_handle_type native_handle() noexcept;
  ::cuda::std::int32_t rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void recv_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

struct no_recv_sync
{
  using native_handle_type = int;
  using group_token_type   = types::group_token;

  native_handle_type native_handle() noexcept;
  ::cuda::std::int32_t rank() noexcept;
  ::cuda::std::int32_t size() noexcept;
  group_token_type group_token();

  template <class Tp>
  void send_sync(group_token_type&, Tp*, ::cuda::std::size_t, ::cuda::std::int32_t);
};

_CCCL_END_NV_DIAG_SUPPRESS()

_CCCL_HOST_DEVICE_API constexpr bool test()
{
  static_assert(cudax::synchronous_communicator<types::synchronous_communicator_model>);
  static_assert(cudax::synchronous_communicator<rank_convertible_to_int>);
  static_assert(cudax::synchronous_communicator<size_convertible_to_int>);

  static_assert(!cudax::synchronous_communicator<empty>);
  static_assert(!cudax::synchronous_communicator<no_native_handle_type>);
  static_assert(!cudax::synchronous_communicator<native_handle_type_not_same>);
  static_assert(!cudax::synchronous_communicator<native_handle_type_throws>);
  static_assert(!cudax::synchronous_communicator<no_rank>);
  static_assert(!cudax::synchronous_communicator<rank_doesnt_return_int>);
  static_assert(!cudax::synchronous_communicator<size_doesnt_return_int>);
  static_assert(!cudax::synchronous_communicator<no_group_token_type>);
  static_assert(!cudax::synchronous_communicator<group_token_doesnt_return_group_token>);
  static_assert(!cudax::synchronous_communicator<no_send_sync>);
  static_assert(!cudax::synchronous_communicator<no_recv_sync>);
  return true;
}
} // namespace

int main(int, char**)
{
  test();
  static_assert(test());
  return 0;
}
