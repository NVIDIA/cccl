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

#include "copy_bytes_common.cuh"
#include <cute/layout.hpp>

/***********************************************************************************************************************
 * LLM Prefill GPU Tensors (large batch)
 **********************************************************************************************************************/

// Hidden states: (19456, 2880):(2880, 1), BF16, ~107 MB
TEST_CASE("llm prefill hidden_states", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 19456 * 2880;
  auto layout         = make_layout(make_shape(19456, 2880), make_stride(2880, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// QKV fused: (19456, 2560):(2560, 1), BF16, ~95 MB
TEST_CASE("llm prefill qkv_fused", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 19456 * 2560;
  auto layout         = make_layout(make_shape(19456, 2560), make_stride(2560, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Q output: (19456, 2048):(2048, 1), BF16, ~76 MB
TEST_CASE("llm prefill q_output", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 19456 * 2048;
  auto layout         = make_layout(make_shape(19456, 2048), make_stride(2048, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Router logits: (19456, 128):(128, 1), BF16, ~4.75 MB
TEST_CASE("llm prefill router_logits", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 19456 * 128;
  auto layout         = make_layout(make_shape(19456, 128), make_stride(128, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Expert indices: (19456, 4):(4, 1), int32, ~304 KB
TEST_CASE("llm prefill expert_indices", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 19456 * 4;
  auto layout         = make_layout(make_shape(19456, 4), make_stride(4, 1));
  test_impl<int32_t>(alloc, 0, layout);
}

// Expert weights: (19456, 4):(4, 1), float32, ~304 KB
TEST_CASE("llm prefill expert_weights", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 19456 * 4;
  auto layout         = make_layout(make_shape(19456, 4), make_stride(4, 1));
  test_impl<float>(alloc, 0, layout);
}

// Expanded indices: (156608,):(1), int32, ~612 KB
TEST_CASE("llm prefill expanded_indices_int32", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 156608;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int32_t>(alloc, 0, layout);
}

// Expanded indices: (156608,):(1), int64, ~1.2 MB
TEST_CASE("llm prefill expanded_indices_int64", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 156608;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int64_t>(alloc, 0, layout);
}

// MoE hidden states: (39152, 2880):(2880, 1), BF16, ~215 MB
TEST_CASE("llm prefill moe_hidden_states", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 39152 * 2880;
  auto layout         = make_layout(make_shape(39152, 2880), make_stride(2880, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Quantized activations: (39152, 2880):(2880, 1), FP8 E4M3, ~107.5 MB
TEST_CASE("llm prefill quantized_activations", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 39152 * 2880;
  auto layout         = make_layout(make_shape(39152, 2880), make_stride(2880, 1));
  test_impl<int8_t>(alloc, 0, layout);
}

// Logits: (20, 201088):(201088, 1), float32, ~15.3 MB
TEST_CASE("llm prefill logits", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 20 * 201088;
  auto layout         = make_layout(make_shape(20, 201088), make_stride(201088, 1));
  test_impl<float>(alloc, 0, layout);
}

// Sampled tokens: (20,):(1), int32, 80 B
TEST_CASE("llm prefill sampled_tokens_int32", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 20;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int32_t>(alloc, 0, layout);
}

// Sampled tokens: (20,):(1), int64, 160 B
TEST_CASE("llm prefill sampled_tokens_int64", "[llm][prefill]")
{
  using namespace cute;
  constexpr int alloc = 20;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int64_t>(alloc, 0, layout);
}

/***********************************************************************************************************************
 * LLM Decode Tensors (small batch)
 **********************************************************************************************************************/

// Hidden states: (1792, 2880):(2880, 1), BF16, ~9.8 MB
TEST_CASE("llm decode hidden_states", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 2880;
  auto layout         = make_layout(make_shape(1792, 2880), make_stride(2880, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// QKV fused: (1792, 2560):(2560, 1), BF16, ~8.75 MB
TEST_CASE("llm decode qkv_fused", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 2560;
  auto layout         = make_layout(make_shape(1792, 2560), make_stride(2560, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Q output: (1792, 2048):(2048, 1), BF16, 7 MB
TEST_CASE("llm decode q_output", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 2048;
  auto layout         = make_layout(make_shape(1792, 2048), make_stride(2048, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Router logits: (1792, 128):(128, 1), BF16, 448 KB
TEST_CASE("llm decode router_logits", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 128;
  auto layout         = make_layout(make_shape(1792, 128), make_stride(128, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Expert indices: (1792, 4):(4, 1), int32, 28 KB
TEST_CASE("llm decode expert_indices", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 4;
  auto layout         = make_layout(make_shape(1792, 4), make_stride(4, 1));
  test_impl<int32_t>(alloc, 0, layout);
}

// Expert weights: (1792, 4):(4, 1), float32, 28 KB
TEST_CASE("llm decode expert_weights", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 4;
  auto layout         = make_layout(make_shape(1792, 4), make_stride(4, 1));
  test_impl<float>(alloc, 0, layout);
}

// Expanded indices: (14336,):(1), int32, 56 KB
TEST_CASE("llm decode expanded_indices_int32", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 14336;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int32_t>(alloc, 0, layout);
}

// Expanded indices: (14336,):(1), int64, 112 KB
TEST_CASE("llm decode expanded_indices_int64", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 14336;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int64_t>(alloc, 0, layout);
}

// MoE hidden states: (3584, 2880):(2880, 1), BF16, ~19.7 MB
TEST_CASE("llm decode moe_hidden_states", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 3584 * 2880;
  auto layout         = make_layout(make_shape(3584, 2880), make_stride(2880, 1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Quantized activations: (3584, 2880):(2880, 1), FP8 E4M3, ~9.8 MB
TEST_CASE("llm decode quantized_activations", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 3584 * 2880;
  auto layout         = make_layout(make_shape(3584, 2880), make_stride(2880, 1));
  test_impl<int8_t>(alloc, 0, layout);
}

// Logits: (1792, 201088):(201088, 1), float32, ~1.34 GB
TEST_CASE("llm decode logits", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792 * 201088;
  auto layout         = make_layout(make_shape(1792, 201088), make_stride(201088, 1));
  test_impl<float>(alloc, 0, layout);
}

// Sampled tokens: (1792,):(1), int32, 7 KB
TEST_CASE("llm decode sampled_tokens_int32", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int32_t>(alloc, 0, layout);
}

// Sampled tokens: (1792,):(1), int64, 14 KB
TEST_CASE("llm decode sampled_tokens_int64", "[llm][decode]")
{
  using namespace cute;
  constexpr int alloc = 1792;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int64_t>(alloc, 0, layout);
}

/***********************************************************************************************************************
 * LLM Static / Weight Tensors
 **********************************************************************************************************************/

// Weight (RMSNorm): (2880,):(1), BF16, ~5.6 KB
TEST_CASE("llm weight rmsnorm", "[llm][weight]")
{
  using namespace cute;
  constexpr int alloc = 2880;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int16_t>(alloc, 0, layout);
}

// Bias (QKV): (2560,):(1), BF16, 5 KB
TEST_CASE("llm bias qkv", "[llm][weight]")
{
  using namespace cute;
  constexpr int alloc = 2560;
  auto layout         = make_layout(make_shape(alloc), make_stride(1));
  test_impl<int16_t>(alloc, 0, layout);
}

// RoPE cos/sin cache: (131072, 32):(32, 1), float2 (8 bytes, using int64_t as surrogate), 32 MB
TEST_CASE("llm rope_cache", "[llm][weight]")
{
  using namespace cute;
  constexpr int alloc = 131072 * 32;
  auto layout         = make_layout(make_shape(131072, 32), make_stride(32, 1));
  test_impl<int64_t>(alloc, 0, layout);
}
