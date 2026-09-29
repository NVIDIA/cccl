// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Illustrative comparison merges; contracts follow the common group planner
// and the BlockMergeSort / WarpMergeSort providers, not an instruction trace.
(() => {
  "use strict";

  const threads = 8;
  const badges = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdef";
  const choice = (value, label) => ({ value, label });
  const group_width = scope => scope === "block" ? threads : scope === "warp" ? 4 : 2;
  const capacity = state => group_width(state.scope) * Number(state.items);

  function merge(left, right, descending) {
    const result = [];
    let a = 0;
    let b = 0;
    while (a < left.length && b < right.length) {
      const take_left = descending ? left[a].key > right[b].key : left[a].key < right[b].key;
      // Right-first ties are one permitted ordering, not a CUB prediction.
      result.push(take_left ? left[a++] : right[b++]);
    }
    return result.concat(left.slice(a), right.slice(b));
  }

  function sort_run(values, descending) {
    if (values.length < 2) return values.slice();
    const middle = Math.floor(values.length / 2);
    return merge(sort_run(values.slice(0, middle), descending), sort_run(values.slice(middle), descending), descending);
  }

  function thread_groups(items) {
    return Array.from({ length: threads }, (_, thread) => ({ start: thread * items, count: items, label: `T${thread}` }));
  }

  function build(state) {
    const items = Number(state.items);
    const width = group_width(state.scope);
    const tile = width * items;
    const valid = state.valid === "all" ? tile : Number(state.valid);
    const descending = state.order === "descending";
    const pairs = state.algorithm === "pairs";
    const count = threads * items;
    const input = Array.from({ length: count }, (_, index) => ({
      key: state.input === "duplicates" ? (index * 5 + 3) % 7 : (index * 5 + 7) % count,
      source: index,
    }));
    const label = entry => pairs ? `${entry.key}:${badges[entry.source]}` : String(entry.key);
    const association = entry => pairs ? `, associated value ${entry.source} (badge ${badges[entry.source]})` : "";
    const initial = input.map((entry, index) => ({
      id: `input${index}`, label: label(entry), value: entry.key, row: "input", index,
      color: Math.floor(index / items), muted: index % tile >= valid,
      detail: `Original input ${index}: T${Math.floor(index / items)}, slot ${index % items}, key ${entry.key}${association(entry)}. ${index % tile < valid ? "Inside this group's valid prefix." : "Outside this group's valid prefix; not an input to the sort."}`,
    }));

    function tokens(values, row, previous, previous_row) {
      return values.map((entry, index) => {
        if (entry === null) return {
          id: `undefined${index}`, label: "?", value: "undefined", row, index, muted: true,
          color: Math.floor(index / items), from: { row: previous_row, index },
          detail: `Group ${Math.floor(index / tile)}, blocked slot ${index % tile}: outside valid_items=${valid}. This slot has no defined sorted result; do not read or store it.`,
        };
        return {
          id: `item${entry.source}`, label: label(entry), value: entry.key, row, index,
          color: Math.floor(entry.source / items),
          from: { row: previous_row, index: previous.indexOf(entry) },
          detail: `Group ${Math.floor(index / tile)}, blocked slot ${index % tile}: key ${entry.key}${association(entry)} from original input ${entry.source}. ${row === "output" ? `Returned to T${Math.floor(index / items)}, slot ${index % items}.` : "Working copy; the original input is unchanged."}`,
        };
      });
    }

    let working = input.map((entry, index) => index % tile < valid ? entry : null);
    const phases = [
      { label: "Inputs", description: "Blocked order follows thread rank, then local item slot. Muted inputs lie outside each group's valid prefix.", tokens: initial },
      { label: "Copy", description: "Copy valid inputs into separate working payloads. The input row stays unchanged throughout the sort.", tokens: initial.concat(tokens(working, "working", input, "input")) },
    ];
    const local = [];
    for (let start = 0; start < count; start += items) {
      const run = sort_run(working.slice(start, start + items).filter(entry => entry !== null), descending);
      local.push(...run, ...Array(items - run.length).fill(null));
    }
    phases.push({
      label: "Local runs", description: `Sort each thread's valid items into an ordered run of up to ${items} keys. Values follow their keys.`,
      tokens: initial.concat(tokens(local, "working", working, "working")),
    });
    working = local;
    for (let run_items = items; run_items < tile; run_items *= 2) {
      const next = [];
      for (let group = 0; group < count; group += tile) {
        for (let start = group; start < group + tile; start += 2 * run_items) {
          const left = working.slice(start, start + run_items).filter(entry => entry !== null);
          const right = working.slice(start + run_items, start + 2 * run_items).filter(entry => entry !== null);
          const run = merge(left, right, descending);
          next.push(...run, ...Array(2 * run_items - run.length).fill(null));
        }
      }
      phases.push({
        label: `Merge ${2 * run_items / items} threads`,
        description: `Merge adjacent runs into up to ${2 * run_items} sorted items. Runs stop at group boundaries and at valid_items=${valid}. Equal keys take the right run first in this illustration.`,
        tokens: initial.concat(tokens(next, "working", working, "working")),
      });
      working = next;
    }
    phases.push({
      label: "Blocked results", description: `Return the sorted prefix to each group's threads in blocked order. ${valid < tile ? "The ? tail is undefined." : "Every output slot is defined."} Inputs remain unchanged.`,
      tokens: initial.concat(tokens(working, "output", working, "working")),
    });
    const outputs = Array.from({ length: count / tile }, (_, group) => {
      const values = working.slice(group * tile, group * tile + valid).map(label);
      return `G${group}: [${values.join(", ")}]`;
    });
    const sentinel = descending ? -1 : count;
    return {
      detail: `${pairs ? "merge_sort_pairs" : "merge_sort_keys"}, ${state.order}: ${count / tile} independent teaching group${count === tile ? "" : "s"}, ${valid} valid items out of ${tile} per group.`,
      rows: [
        { id: "input", label: "Input payloads · remain unchanged", count, groups: thread_groups(items) },
        { id: "working", label: "Working copies · merge inside each group", count, groups: Array.from({ length: count / tile }, (_, group) => ({ start: group * tile, count: tile, label: `G${group}` })) },
        { id: "output", label: "Returned payloads · blocked order · ? is undefined", count, groups: thread_groups(items) },
      ],
      phases,
      notes: [
        pairs ? "A badge identifies the associated original-position value: A=0, B=1, and so on. Select a pair to inspect its numeric value. The badge stays with its key through every merge." : "Colors track the original owner, not the destination thread. Select a key to inspect its original position and current blocked slot.",
        "Every group member participates, including threads with no valid input items. Groups sort independently; sorting each block does not sort a multi-block array.",
        valid < tile ? `This partial tile uses valid_items=${valid} and an illustrative oob_default=${sentinel}. The sentinel must have the key dtype and sort after every valid key, so descending order needs a lower sentinel. ? hides the unspecified tail; it is not a sentinel value.` : "The full-tile call omits valid_items and oob_default. For a partial tile, both arguments must be supplied and uniform within the group.",
        "These are illustrative comparison merges, not a CUB instruction trace. The right-first tie choice demonstrates one permitted result. Merge Sort does not guarantee the relative order of equal keys.",
      ],
      summary: `${outputs.join("; ")}.${valid < tile ? ` Only these ${valid} positions per group may be read or stored.` : ""}`,
      caption: "Eight teaching lanes form one block, two displayed physical warps of four lanes, or four displayed logical warps of two lanes. Physical CUDA warps have 32 lanes; the displayed warp widths and counts are scaled examples. The kernel example below uses actual launch dimensions.",
    };
  }

  window.CoopExplorer.register("merge-sort", {
    title: "Follow a cooperative Merge Sort", eyebrow: "Keys and their associated values", defaultAlgorithm: "pairs",
    algorithms: [
      { id: "keys", label: "Sort keys", tag: "merge_sort_keys" },
      { id: "pairs", label: "Sort key/value pairs", tag: "merge_sort_pairs" },
    ],
    controls: [
      { id: "scope", label: "Group", value: "block", choices: [choice("block", "Block"), choice("warp", "Physical warp"), choice("logical_warp", "Logical warp")] },
      { id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "order", label: "Key order", value: "ascending", choices: ["ascending", "descending"] },
      { id: "input", label: "Input keys", value: "unique", choices: [choice("unique", "Distinct keys"), choice("duplicates", "Repeated keys")] },
      { id: "valid", label: "Valid items per group", value: "all", choices: state => [choice("all", `All ${capacity(state)} items`), ...Array.from({ length: capacity(state) }, (_, index) => choice(String(index), `${index} of ${capacity(state)} items`))] },
    ],
    build,
  });
})();
