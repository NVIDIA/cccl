//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cuda/std/__cccl/cuda_toolkit.h>

#if _CCCL_CTK_AT_LEAST(12, 2)

#  include <cuda/experimental/graph.cuh>
#  include <cuda/experimental/launch.cuh>
#  include <cuda/experimental/stream.cuh>

#  include <testing.cuh>
#  include <utility.cuh>

namespace
{
namespace test
{
// ─── helpers ───────────────────────────────────────────────────────────────

// RAII wrapper around a pinned-memory allocation of N elements of type T.
template <typename T>
struct pinned_array
{
  _malloc_pinned mem;
  std::size_t n;

  explicit pinned_array(std::size_t __n, T __init = T{})
      : mem(__n * sizeof(T))
      , n(__n)
  {
    for (std::size_t i = 0; i < n; ++i)
    {
      get()[i] = __init;
    }
  }

  pinned_array(const pinned_array&)            = delete;
  pinned_array& operator=(const pinned_array&) = delete;

  T* get() const noexcept
  {
    return mem.get_as<T>();
  }
  T& operator[](std::size_t i) const noexcept
  {
    return get()[i];
  }
};

// ─── kernels used in conditional tests ─────────────────────────────────────

#  if _CCCL_CTK_AT_LEAST(12, 4)
// Body kernel for while-loop conditional test: decrements a counter and
// stops the loop when it reaches zero.
struct count_down_and_stop
{
  __device__ void operator()(cudax::conditional_handle handle, int* counter) const noexcept
  {
    --(*counter);
    if (*counter <= 0)
    {
      handle.disable();
    }
  }
};
#  endif // _CCCL_CTK_AT_LEAST(12, 4)
} // namespace test
} // namespace

// ────────────────────────────────────────────────────────────────────────────
// fill_bytes
// ────────────────────────────────────────────────────────────────────────────

C2H_TEST("graph fill_bytes sets every byte to the requested value", "[graph][fill_bytes]")
{
  cudax::stream s{cuda::device_ref{0}};

  constexpr std::size_t N = 64;
  test::pinned_array<int> mem{N, static_cast<int>(0xDEADBEEF)};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Zero-fill via a graph memset node.
  cudax::fill_bytes(pb, ::cuda::std::span{mem.get(), N}, ::cuda::std::uint8_t{0});

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  for (std::size_t i = 0; i < N; ++i)
  {
    CUDAX_REQUIRE(mem[i] == 0);
  }
}

C2H_TEST("graph fill_bytes with non-zero value", "[graph][fill_bytes]")
{
  cudax::stream s{cuda::device_ref{0}};

  constexpr std::size_t N = 8;
  test::pinned_array<unsigned char> mem{N};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::fill_bytes(pb, ::cuda::std::span{mem.get(), N}, ::cuda::std::uint8_t{0xAB});

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  for (std::size_t i = 0; i < N; ++i)
  {
    CUDAX_REQUIRE(mem[i] == static_cast<unsigned char>(0xAB));
  }
}

// ────────────────────────────────────────────────────────────────────────────
// copy_bytes
// ────────────────────────────────────────────────────────────────────────────

C2H_TEST("graph copy_bytes copies data from source to destination", "[graph][copy_bytes]")
{
  cudax::stream s{cuda::device_ref{0}};

  constexpr std::size_t N = 32;
  test::pinned_array<int> src{N};
  test::pinned_array<int> dst{N, -1};

  for (std::size_t i = 0; i < N; ++i)
  {
    src[i] = static_cast<int>(i * 7);
  }

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cudax::copy_bytes(pb, ::cuda::std::span{src.get(), N}, ::cuda::std::span{dst.get(), N});

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  for (std::size_t i = 0; i < N; ++i)
  {
    CUDAX_REQUIRE(dst[i] == src[i]);
  }
}

C2H_TEST("graph copy_bytes can be chained after fill_bytes", "[graph][fill_bytes][copy_bytes]")
{
  cudax::stream s{cuda::device_ref{0}};

  constexpr std::size_t N = 16;
  test::pinned_array<unsigned char> src{N};
  test::pinned_array<unsigned char> dst{N};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Fill source with 0xFF, then copy to destination.
  cudax::fill_bytes(pb, ::cuda::std::span{src.get(), N}, ::cuda::std::uint8_t{0xFF});
  cudax::copy_bytes(pb, ::cuda::std::span{src.get(), N}, ::cuda::std::span{dst.get(), N});

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  for (std::size_t i = 0; i < N; ++i)
  {
    CUDAX_REQUIRE(dst[i] == static_cast<unsigned char>(0xFF));
  }
}

// ────────────────────────────────────────────────────────────────────────────
// host_launch
// ────────────────────────────────────────────────────────────────────────────

C2H_TEST("graph host_launch executes a lambda callback", "[graph][host_launch]")
{
  cudax::stream s{cuda::device_ref{0}};
  // pinned so the host-side increment is visible immediately after sync
  test::pinned<int> counter{0};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Capture the pointer by value so the callback remains valid after graph build.
  int* ptr = counter.get();
  cudax::host_launch(pb, [ptr]() {
    *ptr = 42;
  });

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(*counter == 42);
}

C2H_TEST("graph host_launch with arguments", "[graph][host_launch]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned<int> a{10};
  test::pinned<int> b{20};
  test::pinned<int> result{0};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  int* pa  = a.get();
  int* pb2 = b.get();
  int* pr  = result.get();
  cudax::host_launch(
    pb,
    [](int* x, int* y, int* r) {
      *r = *x + *y;
    },
    pa,
    pb2,
    pr);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(*result == 30);
}

C2H_TEST("graph host_launch can be chained with kernel nodes", "[graph][host_launch]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Kernel sets value to 42.
  int* ptr = mem.get();
  cudax::launch(pb, test::one_thread_dims, test::assign_42{}, ptr);

  // Host callback increments it.
  cudax::host_launch(pb, [ptr]() {
    *ptr += 1;
  });

  // Kernel verifies the final value is 43.
  cudax::launch(pb, test::one_thread_dims, test::verify_n<43>{}, ptr);

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 43);
}

C2H_TEST("graph host_launch can be launched multiple times", "[graph][host_launch]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};
  int* ptr = mem.get();

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Host callback increments the value each time.
  cudax::host_launch(pb, [ptr]() {
    *ptr += 1;
  });

  auto exec = g.instantiate();

  // Launch 5 times — each launch should increment by 1.
  for (int i = 0; i < 5; ++i)
  {
    exec.launch(s);
    s.sync();
    CUDAX_REQUIRE(mem[0] == i + 1);
  }
}

C2H_TEST("graph host_launch data is cleaned up when graph is destroyed", "[graph][host_launch]")
{
  // Use a shared_ptr as a witness: the weak_ptr expires when all copies are gone.
  auto witness              = ::std::make_shared<int>(42);
  ::std::weak_ptr<int> weak = witness;

  {
    cudax::graph_builder g;
    cudax::path_builder pb = cudax::start_path(g);

    // The lambda captures a copy of the shared_ptr, which gets stored in the graph's user object.
    cudax::host_launch(pb, [witness]() {
      (void) witness;
    });

    // Release our copy — the graph's user object should keep the shared_ptr alive.
    witness.reset();
    CUDAX_REQUIRE(!weak.expired());
  }
  // graph_builder destroyed — user object destructor should have deleted the callback data,
  // releasing the last shared_ptr copy.
  CUDAX_REQUIRE(weak.expired());
}

// ────────────────────────────────────────────────────────────────────────────
// event record / wait
// ────────────────────────────────────────────────────────────────────────────

C2H_TEST("graph record_event and wait(event_ref) impose ordering across independent paths",
         "[graph][event_record][event_wait]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};

  cuda::event ev{cuda::device_ref{0}};

  cudax::graph_builder g;

  // Path A: assign 42, then record the event.
  int* val                   = mem.get();
  cudax::path_builder path_a = cudax::start_path(g);
  cudax::launch(path_a, test::one_thread_dims, test::assign_42{}, val);
  path_a.record_event(ev);

  // Path B (independent start): wait on the event, then verify value is 42.
  cudax::path_builder path_b = cudax::start_path(g); // no deps from path_a
  path_b.wait(ev);
  cudax::launch(path_b, test::one_thread_dims, test::verify_42{}, val);

  // Drain both paths.
  path_a.wait(path_b);
  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 42);
}

C2H_TEST("graph record_event node has the correct node type", "[graph][event_record]")
{
  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cuda::event ev{cuda::device_ref{0}};
  auto node = pb.record_event(ev);

  CUDAX_REQUIRE(node.type() == cudax::graph_node_type::event_record);
}

C2H_TEST("graph wait(event_ref) node has the correct node type", "[graph][event_wait]")
{
  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  cuda::event ev{cuda::device_ref{0}};
  auto node = pb.wait(ev);

  CUDAX_REQUIRE(node.type() == cudax::graph_node_type::wait_event);
}

// ────────────────────────────────────────────────────────────────────────────
// child graph
// ────────────────────────────────────────────────────────────────────────────

C2H_TEST("graph insert_child_graph embeds a subgraph", "[graph][child_graph]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};
  int* val = mem.get();

  // Build the child graph: kernel that assigns 42.
  cudax::graph_builder child_g;
  {
    cudax::path_builder child_pb = cudax::start_path(child_g);
    cudax::launch(child_pb, test::one_thread_dims, test::assign_42{}, val);
  }

  // Build the parent graph: embed the child, then verify.
  cudax::graph_builder parent_g;
  cudax::path_builder pb = cudax::start_path(parent_g);

  cudax::insert_child_graph(pb, child_g);
  cudax::launch(pb, test::one_thread_dims, test::verify_42{}, val);

  auto exec = parent_g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 42);
}

#  if _CCCL_CTK_AT_LEAST(12, 9)
C2H_TEST("graph insert_child_graph with ownership transfer", "[graph][child_graph]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};
  int* val = mem.get();

  cudax::graph_builder child_g;
  {
    cudax::path_builder child_pb = cudax::start_path(child_g);
    cudax::launch(child_pb, test::one_thread_dims, test::assign_42{}, val);
  }

  cudax::graph_builder parent_g;
  cudax::path_builder pb = cudax::start_path(parent_g);

  // Move the child graph into the parent — child_g is null afterwards.
  cudax::insert_child_graph(pb, std::move(child_g));
  CUDAX_REQUIRE(child_g.get() == nullptr);

  cudax::launch(pb, test::one_thread_dims, test::verify_42{}, val);

  auto exec = parent_g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 42);
}
#  endif // _CCCL_CTK_AT_LEAST(12, 9)

C2H_TEST("graph insert_child_graph node has the correct node type", "[graph][child_graph]")
{
  cudax::graph_builder child_g;
  {
    cudax::path_builder child_pb = cudax::start_path(child_g);
    cudax::launch(child_pb, test::one_thread_dims, test::empty_kernel{});
  }

  cudax::graph_builder parent_g;
  cudax::path_builder pb = cudax::start_path(parent_g);

  auto node = cudax::insert_child_graph(pb, child_g);

  CUDAX_REQUIRE(node.type() == cudax::graph_node_type::graph);
}

// ────────────────────────────────────────────────────────────────────────────
// conditional nodes (if / while)
// ────────────────────────────────────────────────────────────────────────────

#  if _CCCL_CTK_AT_LEAST(12, 4)

C2H_TEST("graph make_if_node body executes when handle is non-zero", "[graph][conditional][if_node]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};
  int* val = mem.get();

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Default value 1 → body executes.
  auto [cond_node, body_graph, handle] = cudax::make_if_node(pb, /*__default_val=*/true);

  // Populate the body graph: assign 42 to val.
  {
    cudax::path_builder body_pb = cudax::start_path(body_graph);
    cudax::launch(body_pb, test::one_thread_dims, test::assign_42{}, val);
  }

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 42);
}

C2H_TEST("graph make_if_node body is skipped when handle is zero", "[graph][conditional][if_node]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};
  int* val = mem.get();

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Default value 0 → body is skipped.
  auto [cond_node, body_graph, handle] = cudax::make_if_node(pb, /*__default_val=*/false);

  {
    cudax::path_builder body_pb = cudax::start_path(body_graph);
    cudax::launch(body_pb, test::one_thread_dims, test::assign_42{}, val);
  }

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  // val should remain 0 because the body was skipped.
  CUDAX_REQUIRE(mem[0] == 0);
}

C2H_TEST("graph make_while_node body executes the expected number of times", "[graph][conditional][while_node]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1, 5}; // will be decremented to 0

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // Default value 1 → loop runs as long as counter > 0.
  auto [while_node, body_graph, handle] = cudax::make_while_node(pb);

  // Body: decrement counter and stop when done.
  {
    cudax::path_builder body_pb = cudax::start_path(body_graph);
    cudax::launch(body_pb, test::one_thread_dims, test::count_down_and_stop{}, handle, mem.get());
  }

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 0);
}

C2H_TEST("graph make_if_node with pre-constructed handle", "[graph][conditional][if_node]")
{
  cudax::stream s{cuda::device_ref{0}};
  test::pinned_array<int> mem{1};
  int* val = mem.get();

  cudax::graph_builder g;
  cudax::path_builder pb = cudax::start_path(g);

  // User constructs handle directly.
  cudax::conditional_handle my_handle{g, true};
  auto [cond_node, body_graph, handle] = cudax::make_if_node(pb, my_handle);

  {
    cudax::path_builder body_pb = cudax::start_path(body_graph);
    cudax::launch(body_pb, test::one_thread_dims, test::assign_42{}, val);
  }

  auto exec = g.instantiate();
  exec.launch(s);
  s.sync();

  CUDAX_REQUIRE(mem[0] == 42);
}

#  endif // _CCCL_CTK_AT_LEAST(12, 4)

#endif // _CCCL_CTK_AT_LEAST(12, 2)
