//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

/**
 * @file
 * @brief Concepts for sharded structures: `sharded_view`, `owning_sharded`,
 *        per-shard environments, self-binding, and per-call environments.
 *
 * Umbrella over the three concept headers — `concepts/view.cuh` (the view
 * tier), `concepts/env.cuh` (the environment tiers) and
 * `concepts/guards.cuh` (the entry guards every algorithm shares).
 *
 * The design separates three things with three different lifecycles:
 *
 * 1. **The view** (`sharded_view`): plain data describing what the structure
 *    is — for each shard, a contiguous element range, its *region* in the
 *    global index space (offset + size), and an equality-comparable *place*
 *    identity saying where the bytes live. A view owns no elements and holds
 *    no execution resources (span/mdspan semantics: no capacity, no
 *    allocation, no growth). Views are what interop, inspection and
 *    transport consume.
 * 2. **The per-shard environments** (`sharded_env`, `sharded_env_range`):
 *    standard queryable environments supplying, for shard `i`, the stream to
 *    order work on (`cuda::get_stream`, mandatory) and — for algorithms that
 *    need scratch — a memory resource (`cuda::mr::get_memory_resource`).
 *    Environments are either passed explicitly alongside a view, or derived
 *    from structures that carry their own binding (`self_bound` /
 *    `default_envs`, in the spirit of `std::execution`'s `get_env`).
 * 3. **The per-call environment** ("call env"): resources of the scope where
 *    any cross-shard step runs — a result/join stream (its presence selects
 *    the asynchronous contract), a host-accessible staging resource, and the
 *    synchronization policy (`get_sync_policy`).
 *
 * Semantic guarantees of `sharded_view` (checked by `validate()`, not
 * expressible in the concept): shard regions are pairwise disjoint, ordered
 * by global offset, and tile `[0, total extent)` exactly; empty shards are
 * permitted. Algorithms such as scan and adjacent_difference rely on these.
 *
 * Deliberate v1 simplifications (recorded for review): descriptor and
 * structure access are member/field-structural (`.data`, `.size`,
 * `.global_offset`, `.place` on descriptors; `.num_shards()`, `.shard(i)` on
 * structures) rather than customization-point objects. Foreign structures
 * participate by exposing this shape — `basic_shard_view` is the ready-made
 * portable descriptor value type — or through a thin wrapper. Lifting the
 * access layer to CPOs is a mechanical follow-up if a foreign type ever
 * cannot provide the shape.
 *
 * Descriptor `.data` is pointer-only in v1: views double as the storage and
 * transport currency, where addresses are load-bearing (aliasing validation,
 * contiguity, ABI). The planned relaxation is a wider sibling concept for
 * ALGORITHM ARGUMENTS whose `.data` may be any random-access iterator (both
 * input and output positions, constrained per parameter by readability /
 * writability), so per-shard CUB can consume fancy iterators — additive, and
 * the compute paths already use pure iterator arithmetic in anticipation.
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

#include <cuda/experimental/__sharded/concepts/env.cuh>
#include <cuda/experimental/__sharded/concepts/guards.cuh>
#include <cuda/experimental/__sharded/concepts/view.cuh>
