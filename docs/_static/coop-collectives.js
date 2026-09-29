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
  const reduce_scopes = [...scope_choices, choice("thread", "One thread"), choice("mapped_warps", "Warps within a block"), choice("cluster", "Two-block cluster")];
  const block_reduce_algorithms = [
    { id: "raking_commutative_only", label: "Raking, commutative", tag: "CUB · root result" },
    { id: "raking", label: "Raking", tag: "CUB · root result" },
    { id: "warp_reductions", label: "Warp reductions", tag: "CUB · warp partials" },
  ];
  function group_width(scope) {
    return scope === "thread" ? 1 : scope === "logical_warp" ? 2 : scope === "warp" ? 4 : threads;
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
    const direct_cub = state.operator === "custom_max" || state.valid === "half";
    const automatic = { id: "group", label: "Group reduction", tag: "Built-in · hierarchy-aware ownership" };
    if (state.scope !== "block") {
      return direct_cub ? [{ id: "warp", label: "Warp reduction", tag: "CUB · root result" }] : [automatic];
    }
    if (state.ownership === "broadcast") return [automatic];
    return state.operator === "custom_max" ? block_reduce_algorithms.filter(option => option.id !== "raking_commutative_only") : direct_cub ? block_reduce_algorithms : [automatic, ...block_reduce_algorithms];
  }

  function build_reduce(state) {
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
      const defined = state.ownership === "broadcast" || thread % width === 0;
      const value = defined ? totals[Math.floor(thread / width)] : null;
      return token(`result${thread}`, value, "output", thread,
        defined ? `T${thread} receives group aggregate ${value}.` : `T${thread}: no defined result with broadcast=False. Do not read this return value.`,
        Math.floor(thread / width) * width, { row: "combined", index: Math.floor(thread / width) * width });
    });
    const algorithm = reduce_algorithms(state).find(value => value.id === state.algorithm);
    const notes = [
      "Every member participates, including ranks outside a valid prefix. Only the selected prefix contributes to the aggregate.",
      state.ownership === "broadcast" ? "The built-in group path broadcasts one aggregate to every member." : "With broadcast=False, only rank zero of each group has a defined return value; ? marks every other return.",
      "The partial-combine rows show an illustrative legal reduction tree, not a CUB instruction trace. Floating-point results can depend on combination order.",
    ];
    if (state.operator === "custom_max") notes.push("Custom max uses a device callback through cuda.coop.numba_mlir, with broadcast=False; the common API accepts built-in operator names.");
    if (state.scope === "cluster") notes.push("Cluster reduction is implemented for compute capability 9.0 or newer and requires a cluster launch. The picture uses two teaching blocks; grid reduction is unsupported.");
    if (state.scope === "mapped_warps") notes.push("this_block().group_by(2) selects groups of two physical warps (64 threads in executable code); it does not select two individual threads.");
    return {
      detail: `${algorithm.label}: ${state.scope.replaceAll("_", " ")} groups combine ${valid * items} values using ${state.operator === "sum" ? "sum" : "maximum"}.`,
      rows: [
        { id: "input", label: "Input registers · blocked items", count: values.length, groups: groups(items) },
        { id: "local", label: "Local fold · one partial per thread", count: threads, groups: groups(1) },
        { id: "combined", label: state.scope === "cluster" ? "Combine block partials inside a cluster" : "Cooperative combine · illustrative partials", count: threads },
        { id: "output", label: "Return values · ? is undefined", count: threads, groups: groups(1) },
      ],
      phases: [
        { label: "Inputs", description: "Each thread begins with its own values; the input payload is unchanged by reduction.", tokens: initial },
        { label: "Local fold", description: "Each thread combines its items before the group combines per-thread contributions.", tokens: local },
        { label: "Group combine", description: "Independent partials are combined within each participating group. No contribution crosses a group boundary.", tokens: intermediate },
        { label: "Result ownership", description: notes[1], tokens: results },
      ],
      notes,
      summary: `Group aggregates: [${totals.join(", ")}]. ${state.ownership === "broadcast" ? "Every group member receives its aggregate." : "Only each group's rank zero may use its returned aggregate."}`,
      caption: "Eight teaching threads represent the ownership pattern. Displayed physical warps have four lanes, logical warps two; CUDA physical warps have 32 lanes. Cluster mode groups two illustrative blocks of four threads. Values and prefixes are mathematical examples, not performance measurements.",
    };
  }

  window.CoopExplorer.register("reduce", {
    title: "Follow a cooperative reduction", eyebrow: "Values to an aggregate", defaultAlgorithm: "raking",
    algorithms: reduce_algorithms,
    controls: [
      { id: "scope", label: "Group", value: "block", choices: reduce_scopes },
      { id: "operator", label: "Operator", value: "sum", choices: state => [choice("sum", "Sum"), choice("max", "Maximum"), ...(["block", "warp", "logical_warp"].includes(state.scope) ? [choice("custom_max", "Custom maximum callback")] : [])] },
      { id: "items", label: "Items per thread", value: "2", choices: state => state.operator === "custom_max" && state.scope !== "block" ? ["1"] : ["1", "2", "4"] },
      { id: "valid", label: "Contributing ranks", value: "full", choices: state => [choice("full", "All group members"), ...(state.items === "1" && ["block", "warp", "logical_warp"].includes(state.scope) ? [choice("half", "First half (valid_items)")] : [])] },
      { id: "ownership", label: "Return ownership", value: "root", choices: state => [choice("root", "Rank zero only"), ...(state.operator !== "custom_max" && state.valid === "full" ? [choice("broadcast", "All members (broadcast)")] : [])] },
    ],
    build: build_reduce,
  });

})();
