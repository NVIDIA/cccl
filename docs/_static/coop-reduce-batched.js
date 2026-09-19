// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Show the batch axis and ownership contract, not CUB's instruction schedule.
(() => {
  "use strict";

  const choice = (value, label) => ({ value, label });
  const groups = (count, slots, prefix) => Array.from({ length: count }, (_, index) => ({
    start: index * slots, count: slots, label: `${prefix}${index}`,
  }));
  const token = (id, value, row, index, batch, detail, from) => ({
    id, label: value === null ? "?" : String(value),
    value: value === null ? "undefined" : value, row, index, color: batch,
    muted: value === null, detail, from,
  });

  function build(state) {
    const width = Number(state.width);
    const batches = Number(state.batches);
    const slots = Math.ceil(batches / width);
    const maximum = state.algorithm === "max";
    const records = Array.from({ length: width * batches }, (_, index) => {
      const lane = Math.floor(index / batches);
      const batch = index % batches;
      return { lane, batch, index, value: 1 + 2 * lane + batch };
    });
    const input = records.map(item => token(`input-${item.index}`, item.value,
      "input", item.index, item.batch,
      `Lane ${item.lane}, local slot ${item.batch}: value ${item.value} belongs to batch ${item.batch}. Inputs remain unchanged.`));
    const gathered = records.map(item => token(`item-${item.index}`, item.value,
      "batches", item.batch * width + item.lane, item.batch,
      `Batch ${item.batch} includes this value ${item.value} from lane ${item.lane}. This row groups values for explanation; it is not a required shared-memory transpose.`,
      { row: "input", index: item.index }));
    const totals = Array.from({ length: batches }, (_, batch) => {
      const values = records.filter(item => item.batch === batch).map(item => item.value);
      return maximum ? Math.max(...values) : values.reduce((sum, value) => sum + value, 0);
    });
    const aggregates = totals.map((value, batch) => token(`total-${batch}`, value,
      "totals", batch, batch,
      `Batch ${batch}: ${maximum ? "maximum" : "sum"} across ${width} lanes is ${value}.`));
    const output = Array.from({ length: width * slots }, (_, index) => {
      const lane = Math.floor(index / slots);
      const slot = index % slots;
      const batch = state.layout === "striped" ? lane + slot * width : index;
      if (batch >= batches) {
        return token(`unused-${index}`, null, "output", index, 0,
          `Lane ${lane}, result slot ${slot} has no batch. Its value is unspecified; guard reads and stores by batch index.`);
      }
      return token(`total-${batch}`, totals[batch], "output", index, batch,
        `Lane ${lane}, result slot ${slot} owns batch ${batch}, whose result is ${totals[batch]}.`,
        { row: "totals", index: batch });
    });
    const rows = [
      { id: "input", label: "Input payloads · each local slot is a different batch", count: records.length, groups: groups(width, batches, "L") },
      { id: "batches", label: "Same inputs grouped by batch · one contribution from each lane", count: records.length, groups: groups(batches, width, "B") },
      { id: "totals", label: "One aggregate per batch", count: batches },
      { id: "output", label: `Fresh result payloads · ${state.layout} batch ownership · ? is unspecified`, count: width * slots, groups: groups(width, slots, "L") },
    ];
    return {
      rows,
      phases: [
        { label: "Input slots", description: `Each of ${width} lanes contributes ${batches} items. Slot j belongs to batch j, giving ${batches} independent reductions of ${width} values.`, tokens: input },
        { label: "Batch axis", description: "Read across lanes for each batch. The regrouping is a teaching view of the reduction axis, not an implementation step.", tokens: [...input, ...gathered] },
        { label: "Reduce", description: `Apply ${maximum ? "maximum" : "addition"} independently to each batch. There are ${batches} results in total.`, tokens: [...input, ...gathered, ...aggregates] },
        { label: "Distribute", description: `Each lane receives ${slots} result slots. ${state.layout === "striped" ? "Lane r, slot i owns batch r + i × warp width." : "Lane r, slot i owns batch r × slots per lane + i."} ${width * slots - batches} padded slots have unspecified contents.`, tokens: [...input, ...output] },
      ],
      detail: `${batches} batches × ${width} lanes → ${batches} aggregates, distributed over ${width} fresh payloads of ${slots} slots each.`,
      summary: `Batch results in batch order: [${totals.join(", ")}]. Inputs are preserved.`,
      notes: [
        "All lanes in the selected logical warp participate. Other logical warps need not execute the call.",
        "reduce_batched reduces each local slot across lanes. Ordinary reduce combines all payload items into one group aggregate.",
        "Blocked and striped refer to output batch ownership. Neither option changes which input values belong to a batch.",
        "The batch count is a positive compile-time extent; the output extent is ceil(batches / warp width). Slots beyond the batch count must not be read or stored.",
        "This model explains the API contract. CUB uses a native batched warp collective; these stages do not depict its shuffle instructions or a performance comparison.",
      ],
      caption: "One logical warp of four or eight teaching lanes. Colors identify batches. Select a value to inspect its lane and batch; use arrow keys to move between values.",
    };
  }

  window.CoopExplorer.register("reduce-batched", {
    title: "Follow independent batched reductions", eyebrow: "One slot, one batch", defaultAlgorithm: "sum",
    algorithms: [
      { id: "sum", label: "Sum", tag: 'binary_op="sum"' },
      { id: "max", label: "Maximum", tag: 'binary_op="max"' },
    ],
    controls: [
      { id: "width", label: "Logical warp width", value: "4", choices: ["4", "8"] },
      { id: "batches", label: "Batches per warp", value: "3", choices: ["1", "3", "4", "8", "10"] },
      { id: "layout", label: "Output layout", value: "striped", choices: [choice("striped", "Striped"), choice("blocked", "Blocked")] },
    ],
    build,
  });
})();
