// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Mathematical blocked-order models of CUB BlockAdjacentDifference and
// BlockDiscontinuity. These pictures do not model instruction scheduling.
(() => {
  "use strict";

  const threads = 8;
  const pattern = [2, 2, 5, 5, 9, 9, 9, 3, 3, 3, 6, 6, 1, 1, 4, 4];
  const choice = (value, label) => ({ value, label });
  const groups = items => Array.from({ length: threads }, (_, thread) => ({ start: thread * items, count: items, label: `T${thread}` }));
  const inputs = items => Array.from({ length: threads * items }, (_, index) => pattern[index % pattern.length]);
  const token = (id, value, row, index, color, detail, from, muted = false) => ({ id, label: value === null ? "none" : String(value), value, row, index, color, detail, from, muted });
  const row = (id, label, count, items) => ({ id, label, count, groups: groups(items) });
  const boundary_choices = [choice("none", "Not supplied"), choice("0", "0"), choice("2", "2"), choice("9", "9")];
  const caption = "Eight teaching threads form one complete block. Each thread owns consecutive int32 items in blocked order. Colors identify the owning thread. Stages show mathematical dependencies, not the CUB instruction sequence or scratch layout. Select a value for details; arrow keys move between values.";

  function input_tokens(values, items) {
    return values.map((value, index) => token(`input-${index}`, value, "input", index, Math.floor(index / items),
      `Input ${index}: ${value}, owned by T${Math.floor(index / items)}, slot ${index % items}. The input remains unchanged.`));
  }

  function build_difference(state) {
    const items = Number(state.items);
    const values = inputs(items);
    const partial = state.valid !== "full";
    const valid = partial ? Number(state.valid) : values.length;
    const left = state.direction === "left";
    const supplied = state.boundary !== "none";
    const boundary = supplied ? Number(state.boundary) : null;
    const boundary_name = left ? "tile_predecessor_item" : "tile_successor_item";
    const preserved = input_tokens(values, items);
    const rows = [row("input", "Input · blocked · preserved", values.length, items)];
    const boundary_tokens = [];
    if (supplied) {
      rows.push({ id: "boundary", label: `${boundary_name} · uniform scalar outside the tile`, count: 1 });
      boundary_tokens.push(token("tile-boundary", boundary, "boundary", 0, 0,
        `${boundary_name} = ${boundary}. All block members supply the same scalar; it represents the item outside this tile.`));
    }
    rows.push(row("neighbor", `${left ? "Left" : "Right"} neighbor · none means copy the current input`, values.length, items));
    rows.push(row("output", "Returned differences · blocked · invalid suffix copied", values.length, items));
    const neighbors = [];
    const outputs = [];
    const links = [];
    const result = [];
    for (let index = 0; index < values.length; ++index) {
      const active = index < valid;
      const neighbor_index = index + (left ? -1 : 1);
      const inside = active && neighbor_index >= 0 && neighbor_index < valid;
      const external = active && !inside && supplied;
      const neighbor = inside ? values[neighbor_index] : external ? boundary : null;
      const origin = inside ? { row: "input", index: neighbor_index } : external ? { row: "boundary", index: 0 } : undefined;
      const value = neighbor === null ? values[index] : values[index] - neighbor;
      result.push(value);
      let explanation;
      if (!active) explanation = `Position ${index} is outside valid_items=${valid}; copy input ${values[index]} unchanged.`;
      else if (neighbor === null) explanation = `Position ${index} has no ${left ? "left" : "right"} neighbor in the valid tile and no boundary scalar; copy input ${values[index]}.`;
      else explanation = `Position ${index}: current ${values[index]} minus ${inside ? `neighbor at ${neighbor_index}` : boundary_name} (${neighbor}) = ${value}.`;
      neighbors.push(token(`neighbor-${index}`, neighbor, "neighbor", index,
        inside ? Math.floor(neighbor_index / items) : Math.floor(index / items), explanation, origin, !active));
      outputs.push(token(`output-${index}`, value, "output", index, Math.floor(index / items), explanation,
        { row: "input", index }, !active));
      if (origin) links.push({ from: origin, to: { row: "neighbor", index } });
    }
    const initial = [...preserved, ...boundary_tokens];
    return {
      caption: `${caption} The small input values keep subtraction within int32 range.`, rows,
      detail: `adjacent_difference(direction="${state.direction}") · ${partial ? `valid_items=${valid}` : "full tile"}${supplied ? ` · ${boundary_name}=${boundary}` : ""}`,
      phases: [
        { label: "Inputs", description: "Read each thread's consecutive items. Neighbor relationships follow the flattened blocked sequence and can cross thread boundaries.", tokens: initial },
        { label: "Read neighbors", description: `For each valid item, read its ${left ? "previous" : "next"} neighbor. A missing neighbor or an invalid item needs no subtraction.`, tokens: [...initial, ...neighbors], links },
        { label: "Return differences", description: "Subtract the neighbor from the current item. Copy an unpaired boundary item and the invalid suffix unchanged into a new payload.", tokens: [...initial, ...neighbors, ...outputs] },
      ],
      summary: `Returned blocked values: [${result.join(", ")}]. ${values.length - valid} invalid suffix item${values.length - valid === 1 ? " is" : "s are"} copied unchanged. All ${values.length} input slots remain unchanged.`,
      notes: [
        "The subtraction order is current minus neighbor for both directions. A thread boundary does not break the sequence; only the tile boundary needs an external value.",
        "With valid_items, only the first valid_items positions take part. The last valid right-difference item is copied unchanged. Right partial tiles do not accept tile_successor_item, so that control is disabled when direction is right and a count is supplied, even for a full-capacity count.",
        "All input slots must be initialized because the invalid suffix is copied. valid_items and any boundary scalar must be uniform across the complete block. The explorer uses the default subtraction; the qualified API also accepts difference_op(current, neighbor).",
      ],
    };
  }

  function build_discontinuity(state) {
    const items = Number(state.items);
    const values = inputs(items);
    const heads = state.algorithm !== "tails";
    const tails = state.algorithm !== "heads";
    const boundaries = { heads: state.predecessor, tails: state.successor };
    const preserved = input_tokens(values, items);
    const rows = [row("input", "Input · full tile · blocked · preserved", values.length, items)];
    const initial = [...preserved];
    const neighbors = [];
    const links = [];
    const flags = {};
    const result = {};
    for (const mode of ["heads", "tails"].filter(mode => mode === "heads" ? heads : tails)) {
      const left = mode === "heads";
      const supplied = boundaries[mode] !== "none";
      const boundary = supplied ? Number(boundaries[mode]) : null;
      const boundary_name = left ? "tile_predecessor_item" : "tile_successor_item";
      if (supplied) {
        rows.push({ id: `${mode}-boundary`, label: `${boundary_name} · uniform scalar outside the tile`, count: 1 });
        initial.push(token(`${mode}-boundary`, boundary, `${mode}-boundary`, 0, 0, `${boundary_name} = ${boundary}.`));
      }
      rows.push(row(`${mode}-neighbor`, `${left ? "Previous" : "Next"} item · none forces the boundary flag to one`, values.length, items));
      flags[mode] = [];
      result[mode] = [];
      values.forEach((value, index) => {
        const neighbor_index = index + (left ? -1 : 1);
        const inside = neighbor_index >= 0 && neighbor_index < values.length;
        const neighbor = inside ? values[neighbor_index] : boundary;
        const origin = inside ? { row: "input", index: neighbor_index } : supplied ? { row: `${mode}-boundary`, index: 0 } : undefined;
        const flag = neighbor === null || value !== neighbor ? 1 : 0;
        const detail = neighbor === null ? `Position ${index}: no tile ${left ? "predecessor" : "successor"}; the ${left ? "first head" : "last tail"} flag is one.` :
          `Position ${index}: ${left ? `${neighbor} != ${value}` : `${value} != ${neighbor}`} is ${flag ? "true" : "false"}, so the int32 ${mode === "heads" ? "head" : "tail"} flag is ${flag}.`;
        neighbors.push(token(`${mode}-neighbor-${index}`, neighbor, `${mode}-neighbor`, index,
          inside ? Math.floor(neighbor_index / items) : Math.floor(index / items), detail, origin));
        if (origin) links.push({ from: origin, to: { row: `${mode}-neighbor`, index } });
        flags[mode].push(token(`${mode}-${index}`, flag, mode, index, Math.floor(index / items), detail, { row: "input", index }));
        result[mode].push(flag);
      });
    }
    if (heads) rows.push(row("heads", "Returned head flags · int32 · blocked", values.length, items));
    if (tails) rows.push(row("tails", "Returned tail flags · int32 · blocked", values.length, items));
    const phases = [
      { label: "Inputs", description: "Every slot is part of a full block tile. Consecutive equal values form runs that may cross thread boundaries.", tokens: initial },
      { label: "Read neighbors", description: "Compare adjacent values in the flattened blocked sequence. A supplied boundary scalar replaces the otherwise missing outside neighbor.", tokens: [...initial, ...neighbors], links },
    ];
    if (heads) phases.push({ label: "Return heads", description: "Flag the start of each run: previous != current. Without a tile predecessor, the first head is one.", tokens: [...initial, ...neighbors, ...flags.heads] });
    if (tails) phases.push({ label: heads ? "Return heads and tails" : "Return tails", description: `Flag the end of each run: current != next. Without a tile successor, the last tail is one.${heads ? " Return the pair (heads, tails)." : ""}`, tokens: [...initial, ...neighbors, ...(flags.heads || []), ...flags.tails] });
    return {
      caption, rows, phases,
      detail: `discontinuity(mode="${state.algorithm}") · full tile · default inequality predicate`,
      summary: [heads ? `Heads: [${result.heads.join(", ")}].` : "", tails ? `Tails: [${result.tails.join(", ")}].` : "", "Flags occupy the same blocked slots as their input items. Inputs remain unchanged."].filter(Boolean).join(" "),
      notes: [
        "A head marks an item's difference from its predecessor; a tail marks its difference from its successor. A run can span several threads. Returning both produces two int32 payloads in (heads, tails) order.",
        "This operation requires a full tile. It has no valid_items argument. Padding becomes input and can change the last valid tail flag; do not treat arbitrary padding as partial-tile support.",
        "The explorer uses !=. A qualified flag_op(previous, current) defines heads, and flag_op(current, next) defines tails; operand order matters for predicates such as <. Boundaries and mode must agree across the block.",
      ],
    };
  }

  window.CoopExplorer.register("adjacent-difference", {
    title: "Follow adjacent differences", eyebrow: "Neighbors across thread and tile boundaries", defaultAlgorithm: "difference",
    algorithms: [{ id: "difference", label: "Adjacent Difference", tag: "Current item minus its neighbor" }],
    controls: [
      { id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "direction", label: "Direction", value: "left", choices: ["left", "right"] },
      { id: "valid", label: "valid_items", value: "full", choices: state => [choice("full", "Not supplied (full tile)"), ...Array.from({ length: threads * Number(state.items) + 1 }, (_, count) => choice(String(count), String(count)))] },
      { id: "boundary", label: "Tile neighbor value", value: "none", choices: state => state.direction === "right" && state.valid !== "full" ? [boundary_choices[0]] : boundary_choices },
    ],
    build: build_difference,
  });

  window.CoopExplorer.register("discontinuity", {
    title: "Find run boundaries", eyebrow: "Heads and tails in blocked order", defaultAlgorithm: "heads_and_tails",
    algorithms: [
      { id: "heads", label: "Heads", tag: "Previous differs from current" },
      { id: "tails", label: "Tails", tag: "Current differs from next" },
      { id: "heads_and_tails", label: "Heads and tails", tag: "Return both int32 payloads" },
    ],
    controls: [
      { id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "predecessor", label: "tile_predecessor_item", value: "none", choices: state => state.algorithm === "tails" ? [boundary_choices[0]] : boundary_choices, hidden: state => state.algorithm === "tails" },
      { id: "successor", label: "tile_successor_item", value: "none", choices: state => state.algorithm === "heads" ? [boundary_choices[0]] : boundary_choices, hidden: state => state.algorithm === "heads" },
    ],
    build: build_discontinuity,
  });
})();
