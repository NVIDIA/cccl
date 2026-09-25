// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Ownership models adapted from CooperativeDataMotion.tsx in
// python/cuda_coop/docs/fern/fern/components at cccl-mirror
// trentn/dev/cuda-coop 5dba3d36b6eaae48b967d6fa48f9d15e98136000.

(() => {
  "use strict";

  const threads = 8;
  const warp_threads = 4;
  const item_control = {id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"]};

  function owner(layout, value, items) {
    if (layout === "blocked") return [Math.floor(value / items), value % items];
    if (layout === "striped") return [value % threads, Math.floor(value / threads)];
    const warp_size = warp_threads * items;
    return [Math.floor(value / warp_size) * warp_threads + value % warp_threads, Math.floor((value % warp_size) / warp_threads)];
  }

  function register_index(layout, value, items) {
    const [thread, slot] = owner(layout, value, items);
    return thread * items + slot;
  }

  function thread_row(id, label, items) {
    return {
      id, label, count: threads * items,
      groups: Array.from({length: threads}, (_, thread) => ({start: thread * items, count: items, label: `T${thread}`})),
    };
  }

  function owner_text(layout, value, items) {
    const [thread, slot] = owner(layout, value, items);
    return `T${thread}, slot ${slot}`;
  }

  const stores = [
    {id: "direct", label: "Direct", tag: "Blocked scalar stores", input: "blocked", access: "blocked",
      detail: "Each thread writes its consecutive values directly. Adjacent threads address locations separated by the items-per-thread count at each scalar issue step.",
      note: "No exchange scratch. Larger blocked strides can reduce memory-transaction utilization."},
    {id: "striped", label: "Striped", tag: "Striped registers and stores", input: "striped", access: "striped",
      detail: "Each thread already owns striped values. Adjacent threads write adjacent addresses at each item slot, without exchanging values.",
      note: "No exchange scratch. The input payload must already have the intended striped ownership."},
    {id: "vectorize", label: "Vectorize", tag: "Blocked vector candidates", input: "blocked", access: "blocked",
      detail: "Each thread writes its consecutive values using wider memory accesses when the pointer, type, alignment, and item count permit vectorization.",
      note: "No exchange scratch. The diagram shows candidate bundles, not a guarantee of vector instructions."},
    {id: "transpose", label: "Transpose", tag: "Blocked input, striped stores", input: "blocked", access: "striped", exchange: true,
      detail: "A shared-memory exchange converts blocked input into striped writer registers. Adjacent threads can then write adjacent memory locations.",
      note: "The block exchange uses shared scratch and synchronization. Scratch padding is omitted from the diagram."},
    {id: "warp_transpose", label: "Warp transpose", tag: "Warp-striped writer registers", input: "blocked", access: "warp-striped", exchange: true,
      detail: "Each warp exchanges its blocked values into a warp-striped arrangement, then writes its own contiguous memory tile.",
      note: "All warps have exchange scratch. The real block size must be a multiple of the 32-lane physical warp."},
    {id: "warp_transpose_timesliced", label: "Warp transpose, timesliced", tag: "Warps reuse exchange scratch", input: "blocked", access: "warp-striped", exchange: true, timesliced: true,
      detail: "Warps take turns exchanging through one shared scratch region. The rearranged registers then feed warp-striped stores.",
      note: "One warp of exchange scratch is reused in serialized rounds. The real block must contain complete physical warps."},
  ];

  function build_store(state) {
    const option = stores.find((entry) => entry.id === state.algorithm);
    const items = Number(state.items);
    const count = threads * items;
    const rows = [thread_row("input", `Working copy · ${option.input} input`, items)];
    if (option.exchange) rows.push({id: "scratch", label: option.timesliced ? "Shared scratch · one warp at a time" : "Shared scratch · logical positions, padding omitted", count: option.timesliced ? warp_threads * items : count});
    rows.push(thread_row("writers", `${option.access} writer registers`, items));
    rows.push({id: "memory", label: "Global memory · logical item index", count});

    function tokens_for(locations) {
      return Array.from({length: count}, (_, value) => {
        const warp = Math.floor(value / (warp_threads * items));
        const row = locations[warp];
        let index = value;
        if (row === "input") index = register_index(option.input, value, items);
        if (row === "writers") index = register_index(option.access, value, items);
        if (row === "scratch" && option.timesliced) index = value % (warp_threads * items);
        return {
          id: `v${value}`, label: String(value), value, row, index,
          color: owner(option.input, value, items)[0],
          detail: `Value ${value}: input ${owner_text(option.input, value, items)} → writer ${owner_text(option.access, value, items)} → memory[${value}].`,
        };
      });
    }

    const phases = [{label: "Input registers", description: `The tile begins in ${option.input} ownership. Each displayed thread has ${items} item${items === 1 ? "" : "s"}.`, tokens: tokens_for(["input", "input"])}];
    if (option.timesliced) {
      phases.push(
        {label: "Warp 0 exchange", description: "Displayed warp 0 uses shared scratch. Warp 1 retains its blocked input registers.", tokens: tokens_for(["scratch", "input"])},
        {label: "Warp 1 exchange", description: "Warp 0 has warp-striped writer registers. Warp 1 reuses the same scratch for its exchange.", tokens: tokens_for(["writers", "scratch"])},
      );
    } else if (option.exchange) {
      phases.push({label: "Shared exchange", description: "Values pass through scratch at their logical positions; readers then take them in the writer layout.", tokens: tokens_for(["scratch", "scratch"])});
    }
    phases.push(
      {label: "Writer registers", description: `${option.access} registers supply the memory writes. This is an ownership stage, not a barrier or exact instruction schedule.`, tokens: tokens_for(["writers", "writers"])},
      {label: "Memory", description: "The full tile is stored in consecutive memory locations. Each value reaches the address shown by its label.", tokens: tokens_for(["memory", "memory"])},
    );
    return {
      detail: option.detail, rows, phases,
      notes: [option.note, "Store preserves the caller's payload. These stages follow its internal working copy.", items === 1 ? "With one item per thread, blocked and striped ownership coincide; there is no multi-item vector bundle." : "The values and ownership are illustrative; access patterns do not specify a transaction count."],
      summary: `Memory receives values 0–${count - 1} in order. ${option.exchange ? "The exchange changes which thread writes each value." : "Each thread writes directly from the illustrated input ownership."}`,
      caption: "Eight illustrative threads; warp variants use two four-lane teaching warps. CUDA physical warps have 32 lanes. Timeslicing serializes exchange scratch use, not the subsequent global-memory store instruction stream.",
    };
  }

  const exchanges = [
    {id: "striped_to_blocked", label: "Striped to blocked", tag: "Common API", input: "striped", output: "blocked"},
    {id: "blocked_to_striped", label: "Blocked to striped", tag: "Common API", input: "blocked", output: "striped"},
    {id: "warp_striped_to_blocked", label: "Warp-striped to blocked", tag: "Qualified block API", input: "warp-striped", output: "blocked"},
    {id: "blocked_to_warp_striped", label: "Blocked to warp-striped", tag: "Qualified block API", input: "blocked", output: "warp-striped"},
    {id: "scatter_to_blocked", label: "Scatter to blocked", tag: "Ranked destinations", input: "blocked", output: "blocked", scatter: true},
    {id: "scatter_to_striped", label: "Scatter to striped", tag: "Ranked destinations", input: "blocked", output: "striped", scatter: true},
    {id: "scatter_to_striped_guarded", label: "Guarded scatter", tag: "Negative ranks suppress writes", input: "blocked", output: "striped", scatter: true, guarded: true},
    {id: "scatter_to_striped_flagged", label: "Flagged scatter", tag: "Flags suppress writes", input: "blocked", output: "striped", scatter: true, flagged: true},
  ];

  function build_exchange(state) {
    const option = exchanges.find((entry) => entry.id === state.algorithm);
    const items = Number(state.items);
    const count = threads * items;
    const is_suppressed = (value) => (option.guarded && value % 5 === 0) || (option.flagged && value % 4 === 0);
    const destination = (value) => option.scatter ? (5 * value) % count : value;
    const suppressed = Array.from({length: count}, (_, value) => value).filter(is_suppressed);
    const holes = suppressed.map(destination);
    const rows = [
      thread_row("input", `${option.input} input registers`, items),
      {id: "scratch", label: option.scatter ? "Shared scratch · destination ranks" : "Shared scratch · logical positions, padding omitted", count},
      thread_row("output", `${option.output} result registers`, items),
    ];
    if (suppressed.length) rows.push({id: "suppressed", label: "Suppressed inputs · no destination write", count: suppressed.length});

    function tokens_for(row) {
      const tokens = Array.from({length: count}, (_, value) => {
        const target = destination(value);
        const suppressed_value = is_suppressed(value);
        const location = row !== "input" && suppressed_value ? "suppressed" : row;
        let index = target;
        if (location === "input") index = register_index(option.input, value, items);
        if (location === "output") index = register_index(option.output, target, items);
        if (location === "suppressed") index = suppressed.indexOf(value);
        const route = suppressed_value
          ? option.guarded ? "rank −1 suppresses its write" : `rank ${target}, valid flag 0 suppresses its write`
          : `${option.scatter ? `rank ${target} → ` : ""}result ${owner_text(option.output, target, items)}`;
        return {
          id: `v${value}`, label: String(value), value, row: location, index,
          color: owner(option.input, value, items)[0], muted: location === "suppressed",
          detail: `Value ${value}: input ${owner_text(option.input, value, items)} → ${route}.`,
        };
      });
      if (row === "output") {
        for (const target of holes) tokens.push({
          id: `undefined${target}`, label: "?", row: "output", index: register_index(option.output, target, items), color: 7, muted: true,
          detail: `Logical output ${target}, ${owner_text(option.output, target, items)}: no input wrote this destination. Its value is unspecified and must not be consumed.`,
        });
      }
      return tokens;
    }

    const common = ["striped_to_blocked", "blocked_to_striped"].includes(option.id);
    const detail = option.scatter
      ? `The rank attached to each input selects its logical destination. This example starts with rank[p] = (5 × p) mod ${count}, a permutation of the tile.${option.guarded ? " Every fifth input instead has rank −1." : option.flagged ? " Every fourth input has valid flag 0." : ""}`
      : `The shared-memory exchange converts ${option.input} input to ${option.output} ownership while preserving the logical value sequence.`;
    return {
      detail, rows,
      phases: [
        {label: "Input registers", description: "Each thread contributes its fixed-size input payload. Exchange returns a new payload without modifying this input.", tokens: tokens_for("input")},
        {label: option.scatter ? "Scatter by rank" : "Shared exchange", description: option.scatter ? "Each participating input writes to its rank in shared scratch. The demonstrated ranks have no competing writes." : "Values enter shared scratch at their logical positions; the result layout determines which thread reads each position.", tokens: tokens_for("scratch")},
        {label: "Result registers", description: suppressed.length ? `${count - suppressed.length} destinations are defined. The ${suppressed.length} question-mark slots were not written and have unspecified values.` : `The result uses ${option.output} ownership. Exchange itself performs no global-memory load or store.`, tokens: tokens_for("output")},
      ],
      notes: [
        common ? "Available through cuda.coop.exchange for block and warp groups." : "Use cuda.coop.numba_mlir.exchange or cuda.coop.cutlass.exchange with a block group for this mode; it is not available through the common API.",
        option.scatter ? "Ranks must be signed integer payloads with the same item count. Active destinations must be in range and unique; guarded mode only tests whether a rank is negative." : "Shared scratch and synchronization connect the layouts. The diagram omits padding and uses the default non-timesliced exchange.",
        suppressed.length ? "Suppressed writes do not initialize their output slots. A zero shown as an input is a real value; '?' marks an unspecified result." : "Input and output payloads have the same shape. The mode changes ownership rather than the payload extent.",
      ],
      summary: option.scatter ? `Values scatter to logical positions (5 × p) mod ${count}; the output exposes those positions in ${option.output} registers.${suppressed.length ? " Only written destinations may be read." : " Every destination is written once."}` : `Logical order is unchanged. T0 receives [${Array.from({length: count}, (_, value) => value).filter((value) => owner(option.output, value, items)[0] === 0).join(", ")}] in the result.`,
      caption: "Eight illustrative threads, with four lanes per teaching warp for warp-striped layouts. CUDA physical warps have 32 lanes. Colors identify the original source thread. Select a value to inspect its input and result slots.",
    };
  }

  window.CoopExplorer.register("store", {
    title: "Follow a cooperative store", eyebrow: "Registers to memory", defaultAlgorithm: "transpose",
    algorithms: stores.map(({id, label, tag}) => ({id, label, tag})), controls: [item_control], build: build_store,
  });
  window.CoopExplorer.register("exchange", {
    title: "Follow a cooperative exchange", eyebrow: "Changing register ownership", defaultAlgorithm: "striped_to_blocked",
    algorithms: exchanges.map(({id, label, tag}) => ({id, label, tag})), controls: [item_control], build: build_exchange,
  });
})();
