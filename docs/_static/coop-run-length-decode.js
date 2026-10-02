// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Run Length Decode ownership and table lifetime, not a CUB instruction trace.
(() => {
  "use strict";

  const threads = 4;
  const values = [7, 9, 2, 5, 4, 8, 3, 6];
  const invalid_offset = 4294967295;
  const choice = (value, label) => ({ value, label });
  const run_count = state => threads * Number(state.runs);
  const lengths_for = state => Array.from({ length: run_count(state) }, (_, index) => {
    if (state.input === "empty") return 0;
    if (state.input === "full") return index % 3 + 1;
    const lengths = state.input === "long" ? [3, 17, 2] : [3, 2];
    return lengths[index] || 0;
  });
  const total_for = state => lengths_for(state).reduce((sum, length) => sum + length, 0);
  const required_capacity = state => Number(state.destination_offset) + total_for(state);
  const thread_groups = extent => Array.from({ length: threads }, (_, thread) => ({ start: thread * extent, count: extent, label: `T${thread}` }));

  function build(state) {
    const runs = Number(state.runs);
    const decoded_items = Number(state.items);
    const window_size = threads * decoded_items;
    const lengths = lengths_for(state);
    const total = total_for(state);
    const starts = [];
    let prefix = 0;
    for (const length of lengths) {
      starts.push(prefix);
      prefix += length;
    }
    const bulk = state.algorithm === "bulk";
    const show_relative = state.relative === "yes";
    const offset = Number(state.offset);
    const destination_offset = Number(state.destination_offset);
    const needed = required_capacity(state);
    const capacity = state.capacity === "short" ? Math.max(0, needed - 1) : needed + (state.capacity === "padded" ? 3 : 0);
    // Keep one visible cell for a zero-capacity destination, labeled as outside the array.
    const output_count = bulk ? Math.max(1, capacity) : window_size;
    const prepared = total > 0 && (bulk ? capacity >= needed : offset < total);
    const initial = lengths.flatMap((length, index) => [
      {
        id: `value${index}`, row: "values", index, label: String(values[index]), value: values[index],
        color: Math.floor(index / runs), muted: length === 0,
        detail: `Run ${index}: value ${values[index]}, held by T${Math.floor(index / runs)}, slot ${index % runs}. Input remains unchanged.${length === 0 ? " This run is zero-length padding." : ""}`,
      },
      {
        id: `length${index}`, row: "lengths", index, label: String(length), value: length,
        color: Math.floor(index / runs), muted: length === 0,
        detail: `Run ${index} has length ${length}. Positive lengths precede all zero padding. The block total is ${total}.`,
      },
    ]);
    const table = prepared => lengths.map((length, index) => ({
      id: `table${index}`, row: "table", index,
      label: prepared ? `${values[index]}@${starts[index]}` : String(starts[index]),
      value: prepared ? `${values[index]}@${starts[index]}` : starts[index],
      color: Math.floor(index / runs), muted: length === 0,
      from: prepared ? { row: "values", index } : { row: "lengths", index },
      detail: `Run ${index}: exclusive start ${starts[index]}, value ${values[index]}, length ${length}. ${length ? `It covers decoded indices [${starts[index]}, ${starts[index] + length}).` : "Zero-length padding covers no decoded indices."}${prepared ? " This prepared table stays live through every internal bulk window." : " The exclusive sum of earlier lengths gives this start."}`,
    }));
    function find_run(index) {
      return lengths.findIndex((length, run) => length > 0 && starts[run] <= index && index < starts[run] + length);
    }
    function window_tokens(base) {
      return Array.from({ length: window_size }, (_, index) => {
        const decoded_index = base + index;
        const run = find_run(decoded_index);
        const valid = run >= 0;
        const relative = valid ? decoded_index - starts[run] : invalid_offset;
        const explanation = `T${Math.floor(index / decoded_items)}, slot ${index % decoded_items}: decoded index ${decoded_index}. ${valid ? `Run ${run}, value ${values[run]}, relative offset ${relative}.` : "Outside the stream: the wrapper writes value zero and relative offset UINT32_MAX."}`;
        const output = {
          id: `decoded${index}`, row: "output", index,
          label: valid ? String(values[run]) : "0", value: valid ? values[run] : 0,
          color: valid ? Math.floor(run / runs) : 7, muted: !valid, detail: explanation,
        };
        if (valid) output.from = { row: "table", index: run };
        return [output, ...(show_relative ? [{
          id: `relative${index}`, row: "relative", index,
          label: valid ? String(relative) : "MAX", value: relative,
          color: output.color, muted: !valid, detail: explanation,
          ...(valid ? { from: { row: "table", index: run } } : {}),
        }] : [])];
      }).flat();
    }
    function destination_tokens(written, previous) {
      return Array.from({ length: output_count }, (_, index) => {
        const decoded_index = index - destination_offset;
        const valid = decoded_index >= 0 && decoded_index < written;
        const run = valid ? find_run(decoded_index) : -1;
        const absent = capacity === 0;
        const detail = absent ? "The destination has capacity zero; this outlined teaching cell is outside the array and is never written."
          : valid ? `Destination index ${index}: decoded item ${decoded_index}, run ${run}, value ${values[run]}, within-run offset ${decoded_index - starts[run]}.`
            : `Destination index ${index} keeps its initial value -1; this call has not written it.`;
        const output = {
          id: `${valid ? "written" : "untouched"}${index}`, row: "output", index,
          label: absent ? "X" : valid ? String(values[run]) : "-1", value: valid ? values[run] : -1,
          color: valid ? Math.floor(run / runs) : 7, muted: !valid, detail,
        };
        if (valid && decoded_index >= previous) output.from = { row: "table", index: run };
        return [output, ...(show_relative ? [{
          ...output, id: `relative${output.id}`, row: "relative",
          label: absent ? "X" : valid ? String(decoded_index - starts[run]) : "99",
          value: valid ? decoded_index - starts[run] : 99,
          detail: absent ? detail : valid ? `Relative-offset output at destination index ${index}: ${decoded_index - starts[run]} within run ${run}.` : `Relative-offset output index ${index} keeps its initial value 99.`,
        }] : [])];
      }).flat();
    }
    const original_destination = bulk ? destination_tokens(0, 0) : [];
    const phases = [
      { label: "Run inputs", description: `One block owns ${lengths.length} runs in blocked order. Zero lengths pad the tail; the decoded total is ${total}.`, tokens: initial.concat(original_destination) },
      { label: "Exclusive starts", description: `Check lengths and sum them without overflow. Exclusive starts are [${starts.join(", ")}]; the aggregate total is ${total}.`, tokens: initial.concat(table(false), original_destination) },
      { label: prepared ? "Prepare table" : "Skip decoder", description: prepared ? "Copy run values and starts into CUB's shared table. A cell shows value@start. The input payloads remain unchanged." : "This call needs no CUB run table: the stream is empty or the window is out of range. Keep the validated exclusive starts visible.", tokens: initial.concat(table(prepared), original_destination) },
    ];
    const capacity_error = bulk && capacity < needed;
    if (bulk) {
      phases.push({
        label: "Check capacity",
        description: capacity_error ? `Need ${needed} destination slots but have ${capacity}. The provider traps before either output is written.` : `Capacity ${capacity} covers offset ${destination_offset} plus ${total} decoded items. Both outputs are checked before any write.`,
        tokens: initial.concat(table(false), original_destination),
      });
      // The real provider validates capacity before preparing the CUB table.
      // Keep the scan and capacity stages ahead of the preparation stage as well.
      [phases[2], phases[3]] = [phases[3], phases[2]];
      if (capacity_error) phases.pop();
      else if (total === 0) phases.push({
        label: "Return zero", description: "Return total zero to every block member. Neither output array changes.", tokens: initial.concat(table(false), original_destination),
      });
      else for (let base = 0; base < total; base += window_size) {
        const end = Math.min(total, base + window_size);
        phases.push({
          label: `Window ${base}`, description: `Reuse the same prepared table for decoded indices [${base}, ${end}). Threads own ${decoded_items} blocked items each; write only ${end - base} valid items at destination offset ${destination_offset + base}.`,
          tokens: initial.concat(table(true), destination_tokens(end, base)),
        });
      }
    } else phases.push({
      label: "Decode window", description: `Decode from explicit stream offset ${offset}. Each thread receives ${decoded_items} item${decoded_items === 1 ? "" : "s"}. ${offset >= total ? "Skip CUB for this out-of-range window." : "Mask the final window after CUB decoding."} Every out-of-stream value is zero, and its relative offset is MAX.`,
      tokens: initial.concat(table(prepared), window_tokens(offset)),
    });
    const result_tokens = phases[phases.length - 1].tokens.filter(token => token.row === "output");
    return {
      detail: `${bulk ? "run_length_decode_into" : "run_length_decode"}: ${threads} teaching threads, ${runs} run${runs === 1 ? "" : "s"} per thread, decoded window size ${window_size}, total decoded size ${total}.`,
      rows: [
        { id: "values", label: "Run values · inputs remain unchanged", count: lengths.length, groups: thread_groups(runs) },
        { id: "lengths", label: "Run lengths · positive prefix, then zero padding", count: lengths.length, groups: thread_groups(runs) },
        { id: "table", label: "Exclusive starts, then prepared table · value @ start", count: lengths.length },
        { id: "output", label: bulk ? `Destination · ${capacity} slots · initial contents -1` : "Returned values · blocked order · invalid slots are zero", count: output_count, groups: bulk ? [] : thread_groups(decoded_items) },
        ...(show_relative ? [{ id: "relative", label: bulk ? "Optional relative-offset destination · initial contents 99" : "Optional within-run offsets · MAX marks an invalid slot", count: output_count, groups: bulk ? [] : thread_groups(decoded_items) }] : []),
      ].map(row => ({ ...row, align: "start" })),
      phases,
      notes: [
        "The input run tile and output window can have different per-thread extents. All block members participate, including those holding only zero-length padding.",
        bulk ? "Bulk decoding prepares CUB once inside this call. The table remains live while the provider loops over windows, then scratch can be reused with the normal TempStorage synchronization." : "Each separate window call prepares its own table. decoded_window_offset is an explicit stream index; a previous call does not advance a hidden cursor.",
        bulk ? "Both namespaces return the total to every block member. The Numba-qualified API can also write a relative-offset destination and select uint64 totals and offsets." : "Relative offsets restart at zero for each run. The common API returns decoded values; the Numba-CUDA-MLIR-qualified API also exposes total_decoded_size and relative_offsets. MAX here is UINT32_MAX (4294967295).",
        bulk ? "destination_offset shifts where the full stream is written. The output arrays must not overlap each other or the run inputs. A partial final window writes only valid items; other destination cells keep their original contents." : "An empty stream or a window starting at or beyond the total returns zero values and MAX relative offsets. These tail values are defined by the wrapper.",
        "Negative lengths, interior zero lengths followed by positive runs, and total-size overflow are rejected before CUB decoding. Uniform group controls are a caller precondition.",
      ],
      summary: capacity_error ? "Insufficient capacity: trap before writes; both output arrays stay untouched."
        : `${bulk ? "Destination" : "Window values"}: [${result_tokens.map(token => token.label).join(", ")}]. Total ${total}${bulk ? ", returned to every member" : ", available through the Numba-qualified total output"}.`,
      caption: "Four threads illustrate one complete block. CUB selects the actual scan and decode implementation; these stages show data ownership and prepared-table lifetime, not its compiled instruction sequence. The tested kernels below use 128 threads.",
    };
  }

  window.CoopExplorer.register("run-length-decode", {
    title: "Follow Run Length Decode", eyebrow: "Runs, decoded windows, and prepared tables", defaultAlgorithm: "window",
    algorithms: [
      { id: "window", label: "Decode one window", tag: "run_length_decode" },
      { id: "bulk", label: "Decode full stream", tag: "run_length_decode_into" },
    ],
    controls: [
      { id: "runs", label: "Runs per thread", value: "1", choices: ["1", "2"] },
      { id: "items", label: "Decoded items per thread", value: "1", choices: ["1", "2", "4"] },
      { id: "input", label: "Run lengths", value: "short", choices: [choice("short", "3, 2, then zeros"), choice("long", "3, 17, 2, then zeros"), choice("full", "All runs positive"), choice("empty", "All zero (empty)")] },
      { id: "offset", label: "Decoded window offset", value: "2", hidden: state => state.algorithm === "bulk", choices: state => Array.from({ length: total_for(state) + 3 }, (_, index) => String(index)) },
      { id: "destination_offset", label: "Destination offset", value: "3", hidden: state => state.algorithm === "window", choices: ["0", "3"] },
      { id: "capacity", label: "Destination capacity", value: "padded", hidden: state => state.algorithm === "window", choices: state => [choice("padded", "Enough, with spare slots"), choice("exact", "Exactly enough"), ...(required_capacity(state) ? [choice("short", "One slot too small")] : [])] },
      { id: "relative", label: "Show relative offsets", value: "yes", choices: [choice("yes", "Yes (Numba-qualified API)"), choice("no", "No")] },
    ],
    build,
  });
})();
