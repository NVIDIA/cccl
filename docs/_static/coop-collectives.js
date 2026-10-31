// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from python/cuda_coop/docs/fern/fern/components/CooperativeReductionScan.tsx
// at cccl-mirror trentn/dev/cuda-coop 5dba3d36b6eaae48b967d6fa48f9d15e98136000.
// API choices follow the current common and Numba-CUDA-MLIR group planners.

(() => {
  "use strict";

  const threads = 8;
  const choice = (value, label) => ({ value, label });
  const scope_choices = [choice("block", "Block"), choice("warp", "Physical warp"), choice("logical_warp", "Threads within a warp")];
  const block_reduce_algorithms = [
    { id: "raking_commutative_only", label: "Raking, commutative", tag: "CUB · root result" },
    { id: "raking", label: "Raking", tag: "CUB · root result" },
    { id: "warp_reductions", label: "Warp reductions", tag: "CUB · warp partials" },
  ];
  function group_width(scope) {
    // Scale physical and logical warps to four and two teaching threads.
    // Blocks span all eight teaching threads.
    return scope === "logical_warp" ? 2 : scope === "warp" ? 4 : threads;
  }

  function input_values(count, operator) {
    const maximum_values = [3, 1, 4, 1, 5, 9, 2, 6];
    return Array.from({ length: count }, (_, index) => operator === "sum" ? index + 1 : maximum_values[index % maximum_values.length]);
  }

  function combine(left, right, operator) {
    return operator === "sum" ? left + right : Math.max(left, right);
  }

  function fold(values, operator, initial = null) {
    return values.reduce((value, next) => value === null ? next : combine(value, next, operator), initial);
  }

  function token(id, value, row, index, detail, color = index % threads, from = undefined) {
    return { id, label: value === null ? "?" : String(value), value: value === null ? "undefined" : value,
      row, index, color, detail, from, muted: value === null };
  }

  function groups(items) {
    return Array.from({ length: threads }, (_, thread) => ({ start: thread * items, count: items, label: `T${thread}` }));
  }

  function source_tokens(values, items) {
    return values.map((value, index) => token(`v${index}`, value, "input", index,
      `Input ${index}: T${Math.floor(index / items)}, slot ${index % items}, value ${value}.`, Math.floor(index / items)));
  }

  function reduce_algorithms(state) {
    if (state.scope !== "block") {
      return [{ id: "warp", label: "Warp reduction", tag: "CUB · root result" }];
    }
    return state.operator === "custom_max" ? block_reduce_algorithms.filter(option => option.id !== "raking_commutative_only") : block_reduce_algorithms;
  }

  function build_reduce(state) {
    // Keep every thread visible. A valid prefix limits contributions, not
    // participation. Only the group root owns the result.
    const items = Number(state.items);
    const width = group_width(state.scope);
    const valid = state.valid === "half" ? width / 2 : width;
    const values = input_values(threads * items, state.operator);
    const partials = Array.from({ length: threads }, (_, thread) => fold(values.slice(thread * items, (thread + 1) * items), state.operator));
    const totals = Array.from({ length: threads / width }, (_, group) => fold(partials.slice(group * width, group * width + valid), state.operator));
    const initial = source_tokens(values, items);
    const local = partials.map((value, thread) => token(`p${thread}`, value, "local", thread,
      `T${thread} combines its ${items} input item${items === 1 ? "" : "s"} into ${value}.${thread % width >= valid ? " This thread lies outside the valid prefix." : ""}`,
      thread, { row: "input", index: thread * items }));
    const intermediate = [];
    // Draw illustrative combinations within each group. These segments
    // explain the algorithm choices without modeling GPU instructions.
    for (let start = 0; start < threads; start += width) {
      const contributors = Array.from({ length: valid }, (_, index) => start + index);
      let segments;
      if (state.algorithm === "raking_commutative_only" && valid > 1) {
        const half = Math.ceil(valid / 2);
        segments = contributors.slice(0, half).map((thread, index) => [thread, contributors[index + half]].filter(value => value !== undefined));
      } else {
        const segment_size = state.algorithm === "raking" ? 2 : Math.min(4, valid);
        segments = Array.from({ length: Math.ceil(valid / segment_size) }, (_, index) => contributors.slice(index * segment_size, (index + 1) * segment_size));
      }
      for (const segment of segments) {
        const value = fold(segment.map(thread => partials[thread]), state.operator);
        intermediate.push(token(`combine${segment[0]}`, value, "combined", segment[0],
          `Combine T${segment.join(", T")} partials to obtain ${value}.`, segment[0], { row: "local", index: segment[0] }));
      }
    }
    const results = partials.map((_, thread) => {
      const defined = thread % width === 0;
      const value = defined ? totals[Math.floor(thread / width)] : null;
      return token(`result${thread}`, value, "output", thread,
        defined ? `T${thread} receives group aggregate ${value}.` : `T${thread}: no defined result outside group rank zero. Do not read this return value.`,
        Math.floor(thread / width) * width, { row: "combined", index: Math.floor(thread / width) * width });
    });
    const algorithm = reduce_algorithms(state).find(value => value.id === state.algorithm);
    const notes = [
      "Every member participates, including ranks outside a valid prefix. Only the selected prefix contributes to the aggregate.",
      "Only rank zero of each group has a defined return value; ? marks every other return.",
      "The partial-combine rows show an illustrative legal reduction tree, not a CUB instruction trace. Floating-point results can depend on combination order.",
    ];
    if (state.operator === "custom_max") notes.push("Custom max uses a device callback through cuda.coop.numba_mlir; the common API accepts built-in operator names.");
    return {
      detail: `${algorithm.label}: ${state.scope.replaceAll("_", " ")} groups combine ${valid * items} values using ${state.operator === "sum" ? "sum" : "maximum"}.`,
      rows: [
        { id: "input", label: "Input registers · blocked items", count: values.length, groups: groups(items) },
        { id: "local", label: "Local fold · one partial per thread", count: threads, groups: groups(1) },
        { id: "combined", label: "Cooperative combine · illustrative partials", count: threads },
        { id: "output", label: "Return values · ? is undefined", count: threads, groups: groups(1) },
      ],
      phases: [
        { label: "Inputs", description: "Each thread begins with its own values; the input payload is unchanged by reduction.", tokens: initial },
        { label: "Local fold", description: "Each thread combines its items before the group combines per-thread contributions.", tokens: local },
        { label: "Group combine", description: "Independent partials are combined within each participating group. No contribution crosses a group boundary.", tokens: intermediate },
        { label: "Result ownership", description: notes[1], tokens: results },
      ],
      notes,
      summary: `Group aggregates: [${totals.join(", ")}]. Only each group's rank zero may use its returned aggregate.`,
      caption: "Eight teaching threads represent the ownership pattern. Displayed physical warps have four lanes, logical warps two; CUDA physical warps have 32 lanes. Values and prefixes are mathematical examples, not performance measurements.",
    };
  }

  window.CoopExplorer.register("reduce", {
    title: "Follow a cooperative reduction", eyebrow: "Values to an aggregate", defaultAlgorithm: "raking",
    algorithms: reduce_algorithms,
    controls: [
      { id: "scope", label: "Group", value: "block", choices: scope_choices },
      { id: "operator", label: "Operator", value: "sum", choices: [choice("sum", "Sum"), choice("max", "Maximum"), choice("custom_max", "Custom maximum callback")] },
      { id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "valid", label: "Contributing ranks", value: "full", choices: state => [choice("full", "All group members"), ...(state.items === "1" ? [choice("half", "First half (valid_items)")] : [])] },
    ],
    build: build_reduce,
  });

})();
