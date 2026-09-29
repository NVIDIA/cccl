// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from python/cuda_coop/docs/fern/fern/components/CooperativeReductionScan.tsx
// at cccl-mirror trentn/dev/cuda-coop 5dba3d36b6eaae48b967d6fa48f9d15e98136000.
// API choices follow the shared planners and backend-specific callback contracts.

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
  const block_scan_algorithms = [
    { id: "raking", label: "Raking", tag: "Shared segments and prefixes" },
    { id: "raking_memoize", label: "Raking, memoize", tag: "Retain local partials in registers" },
    { id: "warp_scans", label: "Warp scans", tag: "Local warp scans and warp prefixes" },
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
    if (state.operator === "custom_max") notes.push("Custom max uses a device callback through cuda.coop.numba_mlir, with broadcast=False. CUTLASS supports built-in operators; the common API accepts built-in operator names.");
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

  function prefix_choices(state) {
    const generic_exclusive = state.variant === "exclusive_scan";
    const choices = generic_exclusive ? [choice("zero", "Initial value 0"), choice("ten", "Initial value 10")] : [choice("none", "No added prefix")];
    if (state.scope === "block") choices.push(choice("callback", "Callback: aggregate + 7"), choice("stateful", "Running state: prefix 10"));
    return choices;
  }

  function build_scan(state) {
    const items = Number(state.items);
    const width = group_width(state.scope);
    const valid = state.valid === "half" ? width / 2 : width;
    const operator = state.variant.endsWith("_sum") ? "sum" : state.operator;
    const inclusive = state.variant.startsWith("inclusive");
    const values = input_values(threads * items, operator);
    const totals = Array.from({ length: threads }, (_, thread) => fold(values.slice(thread * items, (thread + 1) * items), operator));
    const aggregates = Array.from({ length: threads / width }, (_, group) => fold(values.slice(group * width * items, (group * width + valid) * items), operator));
    const seeds = aggregates.map(aggregate => state.prefix === "callback" ? aggregate + 7 : ["ten", "stateful"].includes(state.prefix) ? 10 : state.prefix === "zero" || !inclusive ? 0 : null);
    const local = values.map((_, index) => {
      const thread = Math.floor(index / items);
      const value = fold(values.slice(thread * items, index + 1), operator);
      return token(`v${index}`, value, "local", index, `T${thread} local inclusive prefix through slot ${index % items}: ${value}.`, thread);
    });
    const prefixes = totals.map((_, thread) => {
      const start = Math.floor(thread / width) * width;
      const prefix = fold(totals.slice(start, thread), operator, seeds[Math.floor(thread / width)]);
      const entry = token(`prefix${thread}`, prefix, "prefix", thread,
        prefix === null ? "This is the first thread: no preceding value or added prefix exists." : `T${thread} receives prefix ${prefix} from earlier threads and the selected initial prefix.`, thread,
        { row: "local", index: thread * items + items - 1 });
      if (prefix === null) { entry.label = "∅"; entry.value = "empty prefix"; }
      if (thread % width >= valid) { entry.label = "?"; entry.value = "undefined"; entry.detail = "This rank is outside valid_items; its output is not defined."; entry.muted = true; }
      return entry;
    });
    const output = values.map((_, index) => {
      const thread = Math.floor(index / items);
      const group = Math.floor(thread / width);
      const value = thread % width < valid ? fold(values.slice(group * width * items, index + (inclusive ? 1 : 0)), operator, seeds[group]) : null;
      return token(`v${index}`, value, "output", index,
        value === null ? `T${thread} is outside valid_items; this result is undefined.` : `${state.variant} result for T${thread}, slot ${index % items}: ${value}.`, thread,
        { row: "local", index });
    });
    const rows = [
      { id: "input", label: "Input registers · blocked sequence", count: values.length, groups: groups(items) },
      { id: "local", label: "Local inclusive prefixes · inputs remain unchanged", count: values.length, groups: groups(items) },
      { id: "prefix", label: "Prefix entering each thread · ∅ means no earlier value", count: threads, groups: groups(1) },
      { id: "output", label: `${inclusive ? "Inclusive" : "Exclusive"} result registers`, count: values.length, groups: groups(items) },
    ];
    if (state.aggregate === "emit") {
      rows.push({ id: "aggregate", label: "Optional aggregate_output · input aggregate, without prefix", count: threads, groups: groups(1) });
      for (let thread = 0; thread < threads; ++thread) output.push(token(`aggregate${thread}`, aggregates[Math.floor(thread / width)], "aggregate", thread,
        "The group input aggregate is returned to every lane; initial_value is not included.", Math.floor(thread / width) * width));
    }
    if (state.prefix === "stateful") {
      rows.push({ id: "state", label: "Running state · inspect at block rank zero", count: 1 });
      output.push(token("state", combine(10, aggregates[0], operator), "state", 0,
        `The callback returned the previous prefix 10 and updated its state with tile aggregate ${aggregates[0]}.`, 0));
    }
    const algorithm = state.scope === "block" ? block_scan_algorithms.find(value => value.id === state.algorithm) : { label: "Warp scan" };
    const notes = [
      `${inclusive ? "Inclusive output includes the current item." : "Exclusive output stops before the current item."} Order is blocked: all of T0's items, then T1's, and so on within each group.`,
      "The local and incoming prefixes are mathematical decompositions. Raking stages through shared segments; memoization retains partials in registers; warp_scans propagates totals between warp scans. The figure does not specify exact instructions or storage padding.",
      state.scope === "block" ? "Block scans accept scalar values or ThreadData, and return a separate result without changing the input." : "Physical and logical warp scans accept one scalar per lane. Every lane participates; ranks beyond valid_items have undefined scan outputs.",
    ];
    if (state.prefix === "callback") notes.push(`The Numba-qualified block prefix callback receives input aggregate ${aggregates[0]} and returns ${seeds[0]} (aggregate + 7). That returned value is combined before the scanned sequence.`);
    if (state.prefix === "stateful") notes.push("The Numba-qualified StatefulFunction receives one-item mutable running state as the third positional argument. Here it returns 10 and combines the tile aggregate into the state for a later scan.");
    if (state.operator === "custom_max" && !state.variant.endsWith("_sum")) notes.push("Custom maximum is a device callback passed as scan_op through cuda.coop.numba_mlir. CUTLASS supports built-in scan operators and does not accept callbacks.");
    if (state.aggregate === "emit") notes.push("Both qualified APIs support aggregate_output as a one-item output. It excludes any initial prefix and cannot be combined with a prefix callback.");
    return {
      detail: `${algorithm.label}: ${state.variant.replaceAll("_", " ")} over ${width * items} ordered items per ${state.scope.replaceAll("_", " ")} group.`,
      rows,
      phases: [
        { label: "Inputs", description: "Values are ordered by thread rank and then by local item slot.", tokens: source_tokens(values, items) },
        { label: "Local prefixes", description: "Each thread computes prefixes within its own items; these do not yet include earlier threads.", tokens: local },
        { label: "Propagate prefixes", description: "Each thread receives the aggregate of earlier threads, combined with any initial or callback prefix.", tokens: prefixes },
        { label: "Scan results", description: notes[0], tokens: output },
      ],
      notes,
      summary: `Scan outputs: [${output.filter(entry => entry.row === "output").map(entry => entry.label).join(", ")}]. Input group aggregates: [${aggregates.join(", ")}].`,
    };
  }

  window.CoopExplorer.register("scan", {
    title: "Follow a cooperative scan", eyebrow: "Ordered values to prefixes", defaultAlgorithm: "raking",
    algorithms: state => state.scope === "block" ? block_scan_algorithms : [{ id: "warp", label: "Warp scan", tag: "One scalar per physical or logical lane" }],
    controls: [
      { id: "scope", label: "Group", value: "block", choices: scope_choices },
      { id: "items", label: "Items per thread", value: "2", choices: state => state.scope === "block" ? ["1", "2", "4"] : ["1"] },
      { id: "variant", label: "Operation", value: "exclusive_sum", choices: [choice("exclusive_sum", "Exclusive sum"), choice("inclusive_sum", "Inclusive sum"), choice("exclusive_scan", "Exclusive scan"), choice("inclusive_scan", "Inclusive scan")] },
      { id: "operator", label: "Operator", value: "sum", choices: state => state.variant.endsWith("_sum") ? [choice("sum", "Sum")] : [choice("sum", "Sum"), choice("max", "Maximum"), choice("custom_max", "Custom maximum callback")] },
      { id: "prefix", label: "Prefix", value: "none", choices: prefix_choices },
      { id: "valid", label: "Contributing ranks", value: "full", choices: state => [choice("full", "All group members"), ...(state.scope !== "block" ? [choice("half", "First half (valid_items)")] : [])] },
      { id: "aggregate", label: "Aggregate output", value: "none", choices: state => [choice("none", "Scan results only"), ...(!["callback", "stateful"].includes(state.prefix) ? [choice("emit", "Also return input aggregate")] : [])] },
    ],
    build: build_scan,
  });
})();
