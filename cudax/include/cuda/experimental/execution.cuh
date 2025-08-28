//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2025 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#ifndef __CUDAX_EXECUTION__
#define __CUDAX_EXECUTION__

// IWYU pragma: begin_exports
#include <cuda/experimental/__execution/apply_sender.cuh>
#include <cuda/experimental/__execution/bulk.cuh>
#include <cuda/experimental/__execution/completion_behavior.cuh>
#include <cuda/experimental/__execution/completion_signatures.cuh>
#include <cuda/experimental/__execution/conditional.cuh>
#include <cuda/experimental/__execution/continues_on.cuh>
#include <cuda/experimental/__execution/cpos.cuh>
#include <cuda/experimental/__execution/domain.cuh>
#include <cuda/experimental/__execution/env.cuh>
#include <cuda/experimental/__execution/get_completion_signatures.cuh>
#include <cuda/experimental/__execution/inline_scheduler.cuh>
#include <cuda/experimental/__execution/just.cuh>
#include <cuda/experimental/__execution/just_from.cuh>
#include <cuda/experimental/__execution/let_value.cuh>
#include <cuda/experimental/__execution/on.cuh>
#include <cuda/experimental/__execution/policy.cuh>
#include <cuda/experimental/__execution/queries.cuh>
#include <cuda/experimental/__execution/read_env.cuh>
#include <cuda/experimental/__execution/run_loop.cuh>
#include <cuda/experimental/__execution/schedule_from.cuh>
#include <cuda/experimental/__execution/sequence.cuh>
#include <cuda/experimental/__execution/start_detached.cuh>
#include <cuda/experimental/__execution/starts_on.cuh>
#include <cuda/experimental/__execution/stop_token.cuh>
#include <cuda/experimental/__execution/stream_context.cuh>
#include <cuda/experimental/__execution/sync_wait.cuh>
#include <cuda/experimental/__execution/then.cuh>
#include <cuda/experimental/__execution/thread_context.cuh>
#include <cuda/experimental/__execution/transform_completion_signatures.cuh>
#include <cuda/experimental/__execution/transform_sender.cuh>
#include <cuda/experimental/__execution/visit.cuh>
#include <cuda/experimental/__execution/when_all.cuh>
#include <cuda/experimental/__execution/write_attrs.cuh>
#include <cuda/experimental/__execution/write_env.cuh>
// IWYU pragma: end_exports

#endif // __CUDAX_EXECUTION__
