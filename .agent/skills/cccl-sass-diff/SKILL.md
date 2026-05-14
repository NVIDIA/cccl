---
description: "Compare CUDA SASS or PTX between two CCCL builds (commits, branches, working-copy vs HEAD) to detect non-trivial codegen changes — filters addresses, symbols, metadata, and pure register renaming. Triggers: \"check for SASS changes\", \"compare SASS\", \"any codegen impact\", \"PTX diff\"."
---

# cccl-sass-diff

Detect meaningful changes in generated CUDA machine code between two versions. Filter trivial noise so only
behavior- or performance-affecting changes surface.

## Inputs

Ask via `cccl-clarify` if unclear:

- Target/library being built.
- SM architectures (detect, offer, confirm).
- Baseline + candidate refs.
- SASS (`cuobjdump -sass`) or PTX (`cuobjdump -ptx`).

## Workflow

1. Build both versions with the same arches and flags.
2. Dump disassembly to `/tmp/claude/<sessionid>/{baseline,candidate}.sass`.
3. Normalize both identically: strip addresses, build IDs, paths, timestamps, whitespace; drop empty/comment lines.
4. `diff -u` the normalized listings.
5. Classify — ignore register renames with identical opcodes/operands, label renumbering, formatting-only
   differences.
6. Report top 5 non-trivial regions: kernel name, change type (opcode, memory-access size, register count delta,
   control-flow), normalized line numbers, plain-language interpretation. Or: "only noise detected".

Save raw + normalized dumps and the diff command under `/tmp/claude/<sessionid>/`. Unsure on impact → surface and
ask.
