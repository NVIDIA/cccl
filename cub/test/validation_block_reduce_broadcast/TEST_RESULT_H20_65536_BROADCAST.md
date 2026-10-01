<!-- SPDX-FileCopyrightText: Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. All rights reserved. -->
<!-- SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception -->

# H20 65,536-row follow-up

The same frozen candidate and control reproduce stable gains at 65,536 rows:
integer normalization improves 1.0503×, 1.0247× and 1.0371× at widths 128/256/512;
retained normalization improves 1.0350× at width 128. Their whole-quartet intervals
exceed the predeclared 1% useful-gain threshold. Retained widths 256/512 do not
meet that threshold; all eight default controls remain within their bound.
These consumer results do not establish a complete-model speedup.

## Frozen source and replacement host

Control `0909f9ee05c632af18fba74a366b657370a12aee` and candidate
`76db7e3f3ab0d665fc87282b0160a321e51e698e` remain clean. All six independently
built libraries match the original hashes. The replacement H20 GPU1 is
`GPU-a603caaa-2262-235a-315b-3152c90bfe66`, driver 580.65.06, CUDA 12.8.93,
Torch 2.7.0+cu128. The private runtime reuses the preinstalled Torch/CUDA.
The earlier 8,192-row measurements use the original host and remain separate.

Of the initial 16 quartets, nine observed foreign GPU1 work and were invalidated
in full: integer 1/3/4/5/6 and retained 1/3/5/7. All 64 original workers exited
naturally. The nine replacement quartets use distinct IDs 9/11/12/13/14 and
9/11/13/15, retaining each original APPA/PAAP order. All 36 replacement workers
exited naturally with code zero; before/after and 0.5-second snapshots observe
no foreign GPU1 work. GPU0 ran an independently coordinated full-model screen.
Every case below has eight valid quartets. The invalid groups and all raw data
remain in the report; no partial quartet or favorable worker was substituted.

The same 20 CUDA-event samples, 100 replays per sample, 10,000 whole-quartet
bootstrap draws, seed 11504 and 1% thresholds apply. Source/diff/library/harness/
common hashes, worker PID and exact input/output equality are checked. Latency
columns are medians of all process medians in microseconds.

| Workload | Kind/width | Quartets | A | P | A/P | 95% quartet CI | Result |
| :--- | :--- | ---: | ---: | ---: | ---: | :--- | :--- |
| integer-65536 | first-thread/32 | 8 | 46.063 | 46.071 | 0.9999× | 0.9992–1.0007 | within default bound |
| integer-65536 | first-thread/128 | 8 | 41.570 | 41.579 | 0.9991× | 0.9979–1.0002 | within default bound |
| integer-65536 | first-thread/256 | 8 | 88.667 | 88.666 | 1.0000× | 0.9999–1.0000 | within default bound |
| integer-65536 | first-thread/512 | 8 | 187.919 | 187.930 | 0.9999× | 0.9998–1.0001 | within default bound |
| integer-65536 | normalization/32 | 8 | 47.200 | 47.384 | 0.9973× | 0.9962–0.9984 | no stable >1% gain |
| integer-65536 | normalization/128 | 8 | 68.617 | 65.306 | 1.0503× | 1.0494–1.0511 | stable >1% gain |
| integer-65536 | normalization/256 | 8 | 135.516 | 132.288 | 1.0247× | 1.0243–1.0250 | stable >1% gain |
| integer-65536 | normalization/512 | 8 | 284.124 | 273.994 | 1.0371× | 1.0369–1.0374 | stable >1% gain |
| retained-65536 | first-thread/32 | 8 | 50.253 | 50.256 | 0.9999× | 0.9986–1.0013 | within default bound |
| retained-65536 | first-thread/128 | 8 | 45.550 | 45.556 | 1.0000× | 0.9987–1.0013 | within default bound |
| retained-65536 | first-thread/256 | 8 | 108.429 | 108.452 | 1.0000× | 0.9993–1.0006 | within default bound |
| retained-65536 | first-thread/512 | 8 | 233.140 | 233.148 | 0.9999× | 0.9997–1.0001 | within default bound |
| retained-65536 | normalization/32 | 8 | 46.146 | 45.891 | 1.0066× | 1.0048–1.0084 | no stable >1% gain |
| retained-65536 | normalization/128 | 8 | 66.683 | 64.442 | 1.0350× | 1.0338–1.0361 | stable >1% gain |
| retained-65536 | normalization/256 | 8 | 142.292 | 141.351 | 1.0066× | 1.0059–1.0073 | no stable >1% gain |
| retained-65536 | normalization/512 | 8 | 306.627 | 303.690 | 1.0096× | 1.0092–1.0100 | no stable >1% gain |

## Native model diagnosis

The original regular-exp full-model screen passed batch 1 but failed native
batch-16 generation before the candidate arm. At length 128, step 15 changes
native top logits 16.375/16.25 into an exact 16.25/16.25 tie. At length 512,
step 5 changes 18.875/18.75 into 18.75/18.875. Torch-only replay preserves all
native tokens and logits. These failures remain failed.

The installed native persistent softmax folds XOR strides 16/8/4/2/1, while
CUB's shuffle reduction folds adjacent lanes first. A common one-warp consumer
with bit-reversed lane mapping and regular expf follows native order. It matches
all 56 fixed random/BF16-rounded/masked FP32-normalization fixtures bitwise and
all native batch-16 32-token generations and all-step logits at both fixed
lengths. The original-order and fast-exp controls match only 4/56 FP32 fixtures.
This changes the shared validation consumer equally for A and P; it is not a
change to the PR algorithm or a source of attributed performance improvement.

Candidate full-model screens and eight fresh-process quartets per batch remain
pending under the frozen [Torch-order protocol](results/recovery_39817/FORMAL_MODEL_TORCH_ORDER_PROTOCOL.md).
The checkpoint, prompts, batches, lengths, precision and error gates are unchanged.
The one-warp faithful consumer validates complete-model equivalence; it does not
exercise the wider-block optimization. No model speedup is claimed. Original
consumer microbenchmarks remain pending. Task-owned model copies remain until
required model workers finish naturally, then will be removed.

## Evidence

[All original and replacement observations](results/recovery_39817/evidence-index.json),
[the audited table](results/recovery_39817/recovery-65536-final-summary.json),
[the native-order diagnostic](results/recovery_39817/TORCH_ORDER_DIAGNOSIS.md) and
[exact harness snapshots](results/recovery_39817/harness-sources.json) are committed.
Use the same reproduction procedure as the original wide-broadcast report;
recompute this table with both controller status files and the recovery manifest.
