---
name: sass-diff
description: Use when asked to check for SASS (or PTX) changes between commits, branches, or a local changeset; guides normalization, comparison, and reporting of CUDA disassembly diffs.
---

# SASS Diffs

Use this when asked to check for SASS changes between commits, branches or a local changeset.

## Goal

Detect relevant changes in generated CUDA machine code (i.e. SASS) while filtering noise from addresses, symbols, metadata, etc.
Any non-trivial change must be detected.

## Inputs to establish

* Compilation target under test
* The CUDA SM architectures to compile for. Try to detect this from the code and offer the user a list of suggestions.
  The user must confirm or provide this list.
* Baseline source (e.g. the previous commit/branch or the current commit without the changes in the working copy).
* Comparison source (e.g.  the current commit/branch or the current commit with the changes in the working copy).
* Whether a SASS (default) or PTX diff is requested.

## Disassembly listing generation

* Compile both, the baseline and comparison source, with the same compiler flags and options.
  When not specified otherwise, lookup the options from `compile_commands.json`
  or the current build system (i.e. CMake files).
  Make sure the CUDA SM architectures (`CMAKE_CUDA_ARCHITECTURES`) are set to the user-provided or approved list.
* Dump the disassembly from the binaries produced in the previous set using `cuobjdump -sass` or `cuobjdump -ptx`.

## Comparison rules (what matters)

Ignore as trivial:

* Register renaming with identical instruction sequence and operands.
* Pure label renumbering or reordering of identical basic blocks.
* Formatting-only differences or reordered symbol tables.
* Changes to symbol names (global function names)

## Reporting

* If any non-trivial change was detected, report the top 5 regions where a non-trivial change was detected,
  including the name of the kernel they appeared in.
* Provide a short summary of the diff type,
  including opcode changes, memory access size/cache policy changes, control-flow changes, register-count changes,
  spills/local memory, shared memory, and occupancy-relevant resource deltas.
* Explicitly state if only noise was detected.
* If you are not sure if the differences are impactful, show it and ask the user for guidance.
* Keep the disassembly dumps available and tell the user where they can find them.
