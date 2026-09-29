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

  window.CoopExplorer.register("store", {
    title: "Follow a cooperative store", eyebrow: "Registers to memory", defaultAlgorithm: "transpose",
    algorithms: stores.map(({id, label, tag}) => ({id, label, tag})), controls: [item_control], build: build_store,
  });
})();
