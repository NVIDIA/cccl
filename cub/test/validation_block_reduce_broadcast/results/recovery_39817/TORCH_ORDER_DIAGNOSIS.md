# Native normalization order diagnosis

All prototypes use the same baseline source. These are accuracy probes, not timing evidence.

| Common consumer | Fixed cases | FP32 bitwise to native | BF16 bitwise to native |
| :--- | ---: | ---: | ---: |
| original-warp | 56 | 4 | 46 |
| torch-order | 56 | 56 | 56 |
| torch-fast | 56 | 4 | 29 |

Torch-only full-model replay is bitwise native. Original full-block regular-exp normalization fails batch 16 at token 15 (128-token prompt) and token 5 (512-token prompt). Native and A logits for the first case are 16.375/16.25 and 16.25/16.25; the second reverses 18.875/18.75 into 18.75/18.875.

The one-warp bit-reversed lane mapping with regular expf matches all native batch-16 tokens and teacher-forced logits at both fixed lengths. This restores the common consumer for baseline and candidate validation; the model source, prompt, precision and accuracy gates remain fixed. It does not establish a wide-block or model performance gain.

The independent FP64 normalization is evaluated inside the BF16 network. Its own greedy path may differ from native at a BF16 decision boundary; it supplies the normalization error reference rather than an alternate native-token gate.
