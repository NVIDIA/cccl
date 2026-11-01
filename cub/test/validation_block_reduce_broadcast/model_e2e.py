# SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Full pinned Qwen3 checkpoint generation with the CUB normalization consumer."""

import argparse
import json
import time
from collections import Counter
from pathlib import Path

import torch
from common import digest, load_consumer, provenance
from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers.models.qwen3 import modeling_qwen3
from transformers.models.qwen3.modeling_qwen3 import Qwen3Attention

parser = argparse.ArgumentParser()
parser.add_argument("--source", type=Path, required=True)
parser.add_argument("--variant", choices=["baseline", "patch"], required=True)
parser.add_argument("--build", type=Path, required=True)
parser.add_argument("--model", type=Path, required=True)
parser.add_argument("--reference", type=Path, required=True)
parser.add_argument("--write-reference", action="store_true")
parser.add_argument("--output", type=Path, required=True)
args = parser.parse_args()
assert not args.write_reference or args.variant == "baseline"
torch.set_num_threads(8)
torch.manual_seed(11504)
torch.backends.cuda.matmul.allow_tf32 = False
consumer = load_consumer(args.build, args.variant)
calls = [0] * 28
shapes: Counter[tuple[int, int]] = Counter()


def attention(
    module: Qwen3Attention,
    query: torch.Tensor,
    key: torch.Tensor,
    value: torch.Tensor,
    attention_mask: torch.Tensor | None,
    scaling: float,
    dropout: float = 0.0,
    **unused: object,
) -> tuple[torch.Tensor, torch.Tensor]:
    assert dropout == 0 and not module.training
    key_states = modeling_qwen3.repeat_kv(key, module.num_key_value_groups)
    value_states = modeling_qwen3.repeat_kv(value, module.num_key_value_groups)
    scores = torch.matmul(query, key_states.transpose(2, 3)) * scaling
    if attention_mask is not None:
        scores = scores + attention_mask[:, :, :, : key_states.shape[-2]]
    probabilities = consumer.normalize(
        scores.float().contiguous(), False, args.variant == "patch"
    ).to(query.dtype)
    calls[module.layer_idx] += 1
    shapes[(scores.numel() // scores.shape[-1], scores.shape[-1])] += 1
    output = torch.matmul(probabilities, value_states).transpose(1, 2).contiguous()
    return output, probabilities


# Retain the eager backend's causal-mask construction and replace its normalization consumer.
modeling_qwen3.eager_attention_forward = attention
model = (
    AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        attn_implementation="eager",
        local_files_only=True,
    )
    .eval()
    .cuda()
)
assert model.config.num_hidden_layers == 28
tokenizer = AutoTokenizer.from_pretrained(args.model, local_files_only=True)
prompt = tokenizer.encode(
    "Explain how a deterministic parallel reduction combines values. ",
    add_special_tokens=False,
)
weights = [parameter.data_ptr() for parameter in model.parameters()]
references = (
    {}
    if args.write_reference
    else torch.load(args.reference, map_location="cpu", weights_only=True)
)
record = provenance(args.source, consumer) | {"variant": args.variant, "cases": []}
cases = []


@torch.inference_mode()
def generate(input_ids: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    current = input_ids
    cache = None
    tokens = []
    logits = None
    for step in range(32):
        output = model(current, past_key_values=cache, use_cache=True)
        if step == 0:
            logits = output.logits[:, -1].clone()
        current = output.logits[:, -1].argmax(dim=-1, keepdim=True)
        tokens.append(current)
        cache = output.past_key_values
    assert logits is not None
    return torch.cat(tokens, dim=-1), logits


for input_tokens in [128, 512]:
    input_ids = torch.tensor(
        (prompt * (input_tokens // len(prompt) + 1))[:input_tokens], device="cuda"
    ).unsqueeze(0)
    for _ in range(2):
        generated, logits = generate(input_ids)
    if args.write_reference:
        references[input_tokens] = {"tokens": generated.cpu(), "logits": logits.cpu()}
    expected = references[input_tokens]
    calls[:] = [0] * 28
    shapes.clear()
    samples = []
    for _ in range(5):
        torch.cuda.synchronize()
        started = time.perf_counter()
        generated, logits = generate(input_ids)
        torch.cuda.synchronize()
        samples.append((time.perf_counter() - started) * 1000)
        torch.testing.assert_close(generated.cpu(), expected["tokens"], rtol=0, atol=0)
        torch.testing.assert_close(logits.cpu(), expected["logits"], rtol=0, atol=0)
    assert calls == [160] * 28
    assert weights == [parameter.data_ptr() for parameter in model.parameters()]
    cases.append(
        {
            "input_tokens": input_tokens,
            "output_tokens": 32,
            "reference_exact": True,
            "input_sha256": digest(input_ids),
            "tokens_sha256": digest(generated),
            "logits_sha256": digest(logits),
            "calls_per_layer": calls.copy(),
            "shapes": [
                {"rows": rows, "columns": columns, "calls": count}
                for (rows, columns), count in sorted(shapes.items())
            ],
            "weight_addresses_stable": True,
            "weight_addresses": weights,
            "samples_ms": samples,
        }
    )
if args.write_reference:
    torch.save(references, args.reference)
record["cases"] = cases
args.output.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record), flush=True)
