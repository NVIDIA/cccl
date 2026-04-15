//===----------------------------------------------------------------------===//
//
// Part of CUDA Experimental in CUDA C++ Core Libraries,
// under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
// SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES.
//
//===----------------------------------------------------------------------===//

#include <cstdint>

#include "copy_common.cuh"

/***********************************************************************************************************************
 * LLM Prefill GPU Tensors (large batch)
 **********************************************************************************************************************/

// src: (19456,2880):(2880,1), BF16, ~107 MB
// dst: (19456,2880):(2880,1)
TEST_CASE("copy d2d llm prefill hidden_states", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int16_t>(19456, 2880);
}

// src: (19456,2560):(2560,1), BF16, ~95 MB
// dst: (19456,2560):(2560,1)
TEST_CASE("copy d2d llm prefill qkv_fused", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int16_t>(19456, 2560);
}

// src: (19456,2048):(2048,1), BF16, ~76 MB
// dst: (19456,2048):(2048,1)
TEST_CASE("copy d2d llm prefill q_output", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int16_t>(19456, 2048);
}

// src: (19456,128):(128,1), BF16, ~4.75 MB
// dst: (19456,128):(128,1)
TEST_CASE("copy d2d llm prefill router_logits", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int16_t>(19456, 128);
}

// src: (19456,4):(4,1), int32, ~304 KB
// dst: (19456,4):(4,1)
TEST_CASE("copy d2d llm prefill expert_indices", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int32_t>(19456, 4);
}

// src: (19456,4):(4,1), float32, ~304 KB
// dst: (19456,4):(4,1)
TEST_CASE("copy d2d llm prefill expert_weights", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<float>(19456, 4);
}

// src: (156608):(1), int32, ~612 KB
// dst: (156608):(1)
TEST_CASE("copy d2d llm prefill expanded_indices_int32", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int32_t>(156608);
}

// src: (156608):(1), int64, ~1.2 MB
// dst: (156608):(1)
TEST_CASE("copy d2d llm prefill expanded_indices_int64", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int64_t>(156608);
}

// src: (39152,2880):(2880,1), BF16, ~215 MB
// dst: (39152,2880):(2880,1)
TEST_CASE("copy d2d llm prefill moe_hidden_states", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int16_t>(39152, 2880);
}

// src: (39152,2880):(2880,1), FP8 E4M3, ~107.5 MB
// dst: (39152,2880):(2880,1)
TEST_CASE("copy d2d llm prefill quantized_activations", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int8_t>(39152, 2880);
}

// src: (20,201088):(201088,1), float32, ~15.3 MB
// dst: (20,201088):(201088,1)
TEST_CASE("copy d2d llm prefill logits", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<float>(20, 201088);
}

// src: (20):(1), int32, 80 B
// dst: (20):(1)
TEST_CASE("copy d2d llm prefill sampled_tokens_int32", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int32_t>(20);
}

// src: (20):(1), int64, 160 B
// dst: (20):(1)
TEST_CASE("copy d2d llm prefill sampled_tokens_int64", "[copy][d2d][llm][prefill]")
{
  test_copy_iota<int64_t>(20);
}

/***********************************************************************************************************************
 * LLM Decode Tensors (small batch)
 **********************************************************************************************************************/

// src: (1792,2880):(2880,1), BF16, ~9.8 MB
// dst: (1792,2880):(2880,1)
TEST_CASE("copy d2d llm decode hidden_states", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int16_t>(1792, 2880);
}

// src: (1792,2560):(2560,1), BF16, ~8.75 MB
// dst: (1792,2560):(2560,1)
TEST_CASE("copy d2d llm decode qkv_fused", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int16_t>(1792, 2560);
}

// src: (1792,2048):(2048,1), BF16, 7 MB
// dst: (1792,2048):(2048,1)
TEST_CASE("copy d2d llm decode q_output", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int16_t>(1792, 2048);
}

// src: (1792,128):(128,1), BF16, 448 KB
// dst: (1792,128):(128,1)
TEST_CASE("copy d2d llm decode router_logits", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int16_t>(1792, 128);
}

// src: (1792,4):(4,1), int32, 28 KB
// dst: (1792,4):(4,1)
TEST_CASE("copy d2d llm decode expert_indices", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int32_t>(1792, 4);
}

// src: (1792,4):(4,1), float32, 28 KB
// dst: (1792,4):(4,1)
TEST_CASE("copy d2d llm decode expert_weights", "[copy][d2d][llm][decode]")
{
  test_copy_iota<float>(1792, 4);
}

// src: (14336):(1), int32, 56 KB
// dst: (14336):(1)
TEST_CASE("copy d2d llm decode expanded_indices_int32", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int32_t>(14336);
}

// src: (14336):(1), int64, 112 KB
// dst: (14336):(1)
TEST_CASE("copy d2d llm decode expanded_indices_int64", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int64_t>(14336);
}

// src: (3584,2880):(2880,1), BF16, ~19.7 MB
// dst: (3584,2880):(2880,1)
TEST_CASE("copy d2d llm decode moe_hidden_states", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int16_t>(3584, 2880);
}

// src: (3584,2880):(2880,1), FP8 E4M3, ~9.8 MB
// dst: (3584,2880):(2880,1)
TEST_CASE("copy d2d llm decode quantized_activations", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int8_t>(3584, 2880);
}

// src: (1792,201088):(201088,1), float32, ~1.34 GB
// dst: (1792,201088):(201088,1)
TEST_CASE("copy d2d llm decode logits", "[copy][d2d][llm][decode]")
{
  test_copy_iota<float>(1792, 201088);
}

// src: (1792):(1), int32, 7 KB
// dst: (1792):(1)
TEST_CASE("copy d2d llm decode sampled_tokens_int32", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int32_t>(1792);
}

// src: (1792):(1), int64, 14 KB
// dst: (1792):(1)
TEST_CASE("copy d2d llm decode sampled_tokens_int64", "[copy][d2d][llm][decode]")
{
  test_copy_iota<int64_t>(1792);
}

/***********************************************************************************************************************
 * LLM Static / Weight Tensors
 **********************************************************************************************************************/

// src: (2880):(1), BF16, ~5.6 KB
// dst: (2880):(1)
TEST_CASE("copy d2d llm weight rmsnorm", "[copy][d2d][llm][weight]")
{
  test_copy_iota<int16_t>(2880);
}

// src: (2560):(1), BF16, 5 KB
// dst: (2560):(1)
TEST_CASE("copy d2d llm bias qkv", "[copy][d2d][llm][weight]")
{
  test_copy_iota<int16_t>(2560);
}

// src: (131072,32):(32,1), float2 (8 bytes, using int64_t as surrogate), 32 MB
// dst: (131072,32):(32,1)
TEST_CASE("copy d2d llm rope_cache", "[copy][d2d][llm][weight]")
{
  test_copy_iota<int64_t>(131072, 32);
}
