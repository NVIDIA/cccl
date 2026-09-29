// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Illustrative candidate filtering, following cub::detail::block_topk_air.
// Small unsigned keys and two-bit digits keep the selection visible; these
// stages are a teaching model, not a trace of the compiled CUB implementation.
(() => {
  "use strict";

  const threads = 8;
  const choice = (value, label) => ({ value, label });
  const mixed = [7, 2, 12, 5, 9, 1, 14, 4, 11, 3, 8, 6, 15, 0, 10, 13];
  const tied = [9, 4, 9, 2, 12, 9, 4, 12, 9, 1, 12, 4, 2, 9, 12, 1];

  function valid_count(state) {
    const capacity = threads * Number(state.items);
    return state.valid === "zero" ? 0 : state.valid === "one" ? 1 :
      state.valid === "half" ? capacity / 2 : state.valid === "partial" ? capacity * 3 / 4 : capacity;
  }

  function groups(items) {
    return Array.from({ length: threads }, (_, thread) => ({ start: thread * items, count: items, label: `T${thread}` }));
  }

  function token(id, value, row, index, color, detail, muted = false, from = undefined) {
    return { id, label: value === null ? "?" : String(value), value: value === null ? "undefined" : value,
      row, index, color, detail, muted, from };
  }

  function build_topk(state) {
    const items = Number(state.items);
    const capacity = threads * items;
    const valid = valid_count(state);
    const k = Number(state.k);
    const count = Math.min(k, valid);
    const pairs = state.payload === "pairs";
    const minimum = state.algorithm === "min";
    const records = Array.from({ length: capacity }, (_, index) => ({
      index, key: state.pattern === "equal" ? 9 : (state.pattern === "ties" ? tied : mixed)[index % 16],
    }));
    const source = records.flatMap(record => {
      const { index, key } = record;
      const owner = Math.floor(index / items);
      const validity = index < valid ? "Valid input." : "Illustrative filler outside valid_items; TopK ignores this slot.";
      const entries = [token(`source-key-${index}`, key, "input", index, owner,
        `Input key ${key} at T${owner}, slot ${index % items} (blocked position ${index}). ${validity}`, index >= valid)];
      if (pairs) entries.push(token(`source-value-${index}`, index, "input-values", index, owner,
        `Input value ${index} is the original position paired with key ${key}. ${validity}`, index >= valid));
      return entries;
    });
    const rows = [{ id: "input", label: "Input keys · blocked layout · unchanged throughout", count: capacity, groups: groups(items) }];
    if (pairs) rows.push({ id: "input-values", label: "Input values · original positions, paired with the keys above", count: capacity, groups: groups(items) });
    rows.push(
      { id: "histogram", label: "Candidate histogram · digit buckets 0, 1, 2, 3", count: 4 },
      { id: "candidates", label: "Selection status at each source position · inspect a key for its status", count: capacity, groups: groups(items) },
      { id: "output", label: `Returned keys · first ${count} blocked positions defined · ? is unspecified`, count: capacity, groups: groups(items) },
    );
    if (pairs) rows.push({ id: "output-values", label: "Returned values · same key/value pairing · ? is unspecified", count: capacity, groups: groups(items) });
    const phases = [{ label: "Inputs", description: `${valid} of ${capacity} input items are valid. All eight threads participate; the input payloads remain unchanged.`, tokens: source }];
    let candidates = records.slice(0, valid);
    const accepted = [];
    let previous = "input";

    if (count > 0 && count < valid) {
      for (const shift of [2, 0]) {
        const histogram = [0, 0, 0, 0];
        for (const record of candidates) histogram[(record.key >> shift) & 3] += 1;
        const order = minimum ? [0, 1, 2, 3] : [3, 2, 1, 0];
        let needed = count - accepted.length;
        let boundary;
        for (const digit of order) {
          if (histogram[digit] >= needed) { boundary = digit; break; }
          needed -= histogram[digit];
        }
        const remaining = [];
        for (const record of candidates) {
          const digit = (record.key >> shift) & 3;
          if (minimum ? digit < boundary : digit > boundary) accepted.push(record);
          else if (digit === boundary) remaining.push(record);
        }
        candidates = remaining;
        const accepted_ids = new Set(accepted.map(record => record.index));
        const candidate_ids = new Set(candidates.map(record => record.index));
        const classification = records.slice(0, valid).map(record => {
          const status = accepted_ids.has(record.index) ? "Guaranteed selection" : candidate_ids.has(record.index) ? "Boundary candidate" : "Excluded from selection";
          return token(`key-${record.index}`, record.key, "candidates", record.index, Math.floor(record.index / items),
            `Key ${record.key} from input position ${record.index}: ${status}. This row annotates the source positions; filtering does not move input data.`,
            !accepted_ids.has(record.index) && !candidate_ids.has(record.index), { row: previous, index: record.index });
        });
        const buckets = histogram.map((size, digit) => token(`bucket-${digit}`, size, "histogram", digit, digit,
          `Digit ${digit} from bits ${shift + 1}:${shift}: ${size} current candidates. ${digit === boundary ? "This bucket contains the selection boundary." : (minimum ? digit < boundary : digit > boundary) ? "These candidates are guaranteed selections." : "These candidates are excluded."}`,
          minimum ? digit > boundary : digit < boundary));
        phases.push({
          label: shift === 2 ? "High digit" : "Low digit",
          description: `Teaching pass over bits ${shift + 1}:${shift}: bucket counts [${histogram.join(", ")}]. ${minimum ? "Smallest" : "Largest"} digits are preferred. ${accepted.length} items are guaranteed selections; choose ${count - accepted.length} of ${candidates.length} candidates in boundary bucket ${boundary}.`,
          tokens: [...source, ...classification, ...buckets],
        });
        previous = "candidates";
        if (accepted.length + candidates.length === count) break;
      }
    }

    // Reverse source order chooses one permitted tie subset and output order.
    // No full sort is needed to implement the illustrative selection.
    const chosen = new Set([...accepted, ...candidates.slice().reverse().slice(0, count - accepted.length)].map(record => record.index));
    const selected = count === valid ? records.slice(0, valid) : records.slice(0, valid).reverse().filter(record => chosen.has(record.index));
    const output = Array.from({ length: capacity }, (_, index) => {
      const record = selected[index];
      const owner = Math.floor(index / items);
      if (!record) {
        const detail = `Output T${owner}, slot ${index % items} (blocked position ${index}) is outside min(k, valid_items) = ${count}. Its contents are unspecified; do not read or store this slot.`;
        const entries = [token(`tail-key-${index}`, null, "output", index, owner, detail, true)];
        if (pairs) entries.push(token(`tail-value-${index}`, null, "output-values", index, owner, detail, true));
        return entries;
      }
      const color = Math.floor(record.index / items);
      const detail = `Output T${owner}, slot ${index % items} receives key ${record.key} from original position ${record.index}. This is one permitted output order; TopK does not sort the selection.`;
      const entries = [token(`key-${record.index}`, record.key, "output", index, color, detail, false, { row: previous, index: record.index })];
      if (pairs) entries.push(token(`value-${record.index}`, record.index, "output-values", index, color,
        `Output value ${record.index} stays paired with original key ${record.key} at blocked output position ${index}.`, false, { row: "input-values", index: record.index }));
      return entries;
    }).flat();
    const boundary_note = count === 0 ? "No output items are defined because k or valid_items is zero." : count === valid ?
      "All valid items are selected. The illustration retains their original blocked positions; no filtering is needed." :
      "The selected items fill a blocked prefix in one permitted order. Equal keys at the boundary have no guaranteed selection or tie order.";
    phases.push({ label: "Defined output", description: boundary_note, tokens: [...source, ...output] });
    return {
      rows, phases,
      detail: `Block TopK ${minimum ? "minimum" : "maximum"} ${pairs ? "pairs" : "keys"}: k = ${k}, valid_items = ${valid}, capacity = ${capacity}. Defined output length = min(${k}, ${valid}) = ${count}.`,
      summary: `Defined keys: [${selected.map(record => record.key).join(", ")}].${pairs ? ` Paired original positions: [${selected.map(record => record.index).join(", ")}].` : ""} ${capacity - count} output slots are unspecified.`,
      notes: [
        "Only the first min(k, valid_items) blocked output positions may be read or stored. Both counts are uniform across the block and lie between zero and capacity, inclusive.",
        "The input payloads are preserved. For pairs, each selected value remains attached to its original key.",
        "The result is unordered. The illustrated order and tied-key choices are examples, with no API guarantee; sorting is a separate operation.",
        "The histogram passes illustrate most-significant-digit candidate filtering for small unsigned keys. CUB uses its own digit width and stopping conditions; these stages are not an instruction trace or a performance comparison.",
      ],
      caption: "One complete one-dimensional block with eight teaching threads. Each thread owns consecutive blocked items. TopK has no Warp or logical-warp group mode in this API. Select a value to inspect it; use arrow keys to move between values.",
    };
  }

  window.CoopExplorer.register("topk", {
    title: "Follow a block TopK selection", eyebrow: "Candidates to an unordered prefix", defaultAlgorithm: "max",
    algorithms: [
      { id: "max", label: "Largest keys", tag: "topk_max_keys / topk_max_pairs" },
      { id: "min", label: "Smallest keys", tag: "topk_min_keys / topk_min_pairs" },
    ],
    controls: [
      { id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "payload", label: "Payload", value: "pairs", choices: [choice("keys", "Keys only"), choice("pairs", "Key/value pairs")] },
      { id: "pattern", label: "Input keys", value: "mixed", choices: [choice("mixed", "Mixed keys"), choice("ties", "Repeated keys"), choice("equal", "All keys equal")] },
      { id: "k", label: "Requested k", value: "4", choices: state => Array.from({ length: threads * Number(state.items) + 1 }, (_, value) => choice(value, String(value))) },
      { id: "valid", label: "Valid input prefix", value: "partial", choices: state => [
        choice("full", `Full tile (${threads * Number(state.items)})`),
        choice("partial", `Partial tile (${threads * Number(state.items) * 3 / 4})`),
        choice("half", `Half tile (${threads * Number(state.items) / 2})`),
        choice("one", "One item"), choice("zero", "Zero items"),
      ] },
    ],
    build: build_topk,
  });
})();
