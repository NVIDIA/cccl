// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from CooperativeDataMotion.tsx at cccl-mirror trentn/dev/cuda-coop
// 5dba3d36b6eaae48b967d6fa48f9d15e98136000; contracts follow this Numba stack.
(() => {
  "use strict";

  const scalar = (state) => ["offset", "rotate"].includes(state.algorithm);
  const algorithms = [
    { id: "up", label: "Up", tag: "Previous item in the blocked tile" },
    { id: "down", label: "Down", tag: "Next item in the blocked tile" },
    { id: "offset", label: "Offset", tag: "Qualified scalar neighbor" },
    { id: "rotate", label: "Rotate", tag: "Qualified scalar wraparound" },
  ];

  function build_shuffle(state) {
    const items = Number(state.items);
    const count = 8 * items;
    const distance = Number(state.distance);
    const is_scalar = scalar(state);
    const groups = Array.from({ length: 8 }, (_, thread) => ({ start: thread * items, count: items, label: `T${thread}` }));
    const source_index = (output) => {
      if (state.algorithm === "up") return output - 1;
      if (state.algorithm === "down") return output + 1;
      if (state.algorithm === "rotate") return (output + distance) % count;
      return output + distance;
    };
    const input = Array.from({ length: count }, (_, value) => ({
      id: `value-${value}`, label: String(value), value, row: "input", index: value,
      color: Math.floor(value / items),
      detail: `Value ${value} starts in T${Math.floor(value / items)}, slot ${value % items}.`,
    }));
    const carried = input.map((token) => {
      const slot = token.index % items;
      const boundary = is_scalar || (state.algorithm === "up" ? slot === items - 1 : slot === 0);
      return boundary ? { ...token, row: "scratch", index: Math.floor(token.index / items), detail: `${token.detail} This value is available to a neighboring thread through shared scratch.` } : token;
    });
    const result = [];
    const links = [];
    const values = [];
    for (let output = 0; output < count; ++output) {
      const source = source_index(output);
      if (source < 0 || source >= count) {
        result.push({ id: `undefined-${output}`, label: "?", row: "output", index: output, muted: true,
          detail: `Output ${output} has no source inside this block. Its value is undefined; do not consume it.` });
        values.push("?");
      } else {
        const token = input[source];
        result.push({ ...token, row: "output", index: output,
          detail: `Output ${output} in T${Math.floor(output / items)}, slot ${output % items}, reads input ${source} (value ${source}).` });
        links.push({ from: { row: "input", index: source }, to: { row: "output", index: output } });
        values.push(source);
      }
    }
    const formula = state.algorithm === "up" ? "output[p] = input[p - 1]" : state.algorithm === "down" ? "output[p] = input[p + 1]" : state.algorithm === "offset" ? `output[t] = input[t + ${distance}]` : `output[t] = input[(t + ${distance}) % 8]`;
    return {
      detail: `${formula}. ${is_scalar ? "Each thread supplies one scalar through the qualified Numba-CUDA-MLIR API." : "The common API shifts the flattened blocked payload by exactly one item, including across thread boundaries."}`,
      rows: [
        { id: "input", label: "Input values · working copies", count, groups },
        { id: "scratch", label: is_scalar ? "Shared scratch · one scalar per thread" : "Shared scratch · boundary value from each thread", count: 8 },
        { id: "output", label: "Result registers · blocked ownership", count, groups },
      ],
      phases: [
        { label: "Input values", description: "All block threads participate. The input payload remains unchanged by the operation.", tokens: input },
        { label: "Neighbor exchange", description: is_scalar ? "Threads publish their scalar values for neighboring threads to read." : "Each thread shares its boundary item; shifts between its own slots remain local.", tokens: carried },
        { label: "Result registers", description: state.algorithm === "rotate" ? "Modulo addressing supplies a value for every thread." : "Only outputs with an in-block source are defined. Question marks are not numeric results.", tokens: result, links },
      ],
      summary: `Blocked result: [${values.join(", ")}].`,
      notes: [
        "Shuffle returns a new result and preserves its input. The diagram follows working copies, not mutations to the caller's payload.",
        is_scalar ? "Offset accepts signed distances. Rotate requires 1 <= distance < block size. Runtime distances may differ between threads; this diagram uses a uniform distance." : "Up and Down require a fixed-size per-thread payload and compile-time distance 1. The boundary output is undefined, not wrapped or copied through.",
        "The group is the complete block. All threads must call the primitive; guard consumption of undefined outputs afterward.",
      ],
      caption: "Teaching model: eight threads. Up/Down shift a blocked tile of 1, 2, or 4 items per thread; Offset/Rotate operate on one scalar per thread. Stages show ownership and dependencies, not instruction timing. Arrow keys inspect neighboring values.",
    };
  }

  window.CoopExplorer.register("shuffle", {
    title: "Follow a cooperative shuffle",
    eyebrow: "Neighbor values within a block",
    defaultAlgorithm: "up",
    algorithms,
    controls: [
      { id: "items", label: "Items per thread", value: "2", choices: (state) => scalar(state) ? ["1"] : ["1", "2", "4"] },
      { id: "distance", label: "Distance", value: "1", choices: (state) => state.algorithm === "offset" ? ["-2", "-1", "0", "1", "2"] : state.algorithm === "rotate" ? ["1", "2", "3", "7"] : ["1"] },
    ],
    build: build_shuffle,
  });
})();
