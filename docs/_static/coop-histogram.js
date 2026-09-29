// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Teaching stages for CUB BlockHistogram: atomic updates or sort/run counts.
// The sequence explains values and ownership, not GPU scheduling or timing.
(() => {
  "use strict";

  const threads = 8;
  const choice = (value, label) => ({ value, label });
  const groups = width => Array.from({ length: threads }, (_, thread) => ({
    start: thread * width, count: width, label: `T${thread}`,
  }));
  const token = (id, value, row, index, color, detail, muted = false, from = undefined) =>
    ({ id, label: String(value), value, row, index, color, detail, muted, from });

  function build_histogram(state) {
    const items = Number(state.items);
    const bins = Number(state.bins);
    const bins_per_thread = Number(state.bins_per_thread);
    const capacity = threads * items;
    const records = Array.from({ length: capacity }, (_, index) => {
      const sample = state.pattern === "zero" ? 0 : state.pattern === "hot" ?
        (index % 4 === 0 ? index % bins : bins - 1) : (index * 5 + Math.floor(index / 3)) % bins;
      return { index, sample };
    });
    const source = records.map(({ index, sample }) => token(`source-${index}`, sample, "input", index,
      Math.floor(index / items), `T${Math.floor(index / items)}, local slot ${index % items}: sample ${sample} contributes one count to bin ${sample}. This input is preserved.`));
    const rows = [
      { id: "input", label: "Input samples · blocked payloads · preserved throughout", count: capacity, groups: groups(items) },
      { id: "working", label: state.algorithm === "sort" ? "Private working copy · samples may be sorted here" : "Contributions · one sample from each thread in a teaching round", count: capacity },
    ];
    if (state.algorithm === "sort") rows.push({ id: "runs", label: "Equal-sample run lengths · shown at each run head", count: capacity });
    rows.push(
      { id: "bins", label: "Shared bin counts · positions are bin numbers", count: bins },
      { id: "owners", label: "Bin assigned to each returned local slot · · means padding", count: threads * bins_per_thread, groups: groups(bins_per_thread) },
      { id: "output", label: "Returned counts · striped bin ownership · padding is zero", count: threads * bins_per_thread, groups: groups(bins_per_thread) },
    );
    const histogram = Array(bins).fill(0);
    const bin_tokens = () => histogram.map((count, bin) => token(`bin-${bin}`, count, "bins", bin, bin % threads,
      `Bin ${bin} currently contains ${count} sample${count === 1 ? "" : "s"}.`));
    const phases = [
      { label: "Inputs", description: `All ${capacity} input slots contain valid bin indices in [0, ${bins}). There is no partial-input control.`, tokens: source },
      { label: "Initialize", description: `This call starts all ${bins} bin counters at zero. Reusing scratch does not retain counts from an earlier call.`, tokens: [...source, ...bin_tokens()] },
    ];
    let working = [];
    if (state.algorithm === "atomic") {
      for (let item = 0; item < items; item += 1) {
        for (let thread = 0; thread < threads; thread += 1) histogram[records[thread * items + item].sample] += 1;
        working = records.map(({ index, sample }) => token(`work-${index}`, sample, "working", index, Math.floor(index / items),
          `Input position ${index}, sample ${sample}: ${index % items <= item ? "its count has been added" : "its count is still pending"}. Atomic increments to the same bin must not lose updates.`,
          index % items > item, { row: "input", index }));
        phases.push({ label: `Add slot ${item}`, description: `Teaching round ${item + 1}: each thread contributes its local slot ${item}. ${threads * (item + 1)} samples have been counted. Actual atomic updates may occur in a different order.`, tokens: [...source, ...working, ...bin_tokens()] });
      }
    } else {
      working = records.map(({ index, sample }) => token(`work-${index}`, sample, "working", index, Math.floor(index / items),
        `Private copy of input position ${index}; sorting this copy preserves the caller's sample.`, false, { row: "input", index }));
      phases.push({ label: "Private copy", description: "The input-preserving wrapper copies samples before CUB sorts its working items.", tokens: [...source, ...working, ...bin_tokens()] });
      const sorted = records.slice().sort((left, right) => left.sample - right.sample || left.index - right.index);
      working = sorted.map(({ index, sample }, position) => token(`work-${index}`, sample, "working", position, Math.floor(index / items),
        `Working position ${position} holds sample ${sample}, copied from input position ${index}. Equal samples are consecutive; their internal order does not affect counts.`, false, { row: "input", index }));
      phases.push({ label: "Private sort", description: "The sort algorithm groups equal sample values in a private working copy. The input row is unchanged.", tokens: [...source, ...working, ...bin_tokens()] });
      const run_tokens = [];
      for (let begin = 0; begin < sorted.length;) {
        let end = begin + 1;
        while (end < sorted.length && sorted[end].sample === sorted[begin].sample) end += 1;
        histogram[sorted[begin].sample] = end - begin;
        run_tokens.push(token(`run-${begin}`, end - begin, "runs", begin, sorted[begin].sample % threads,
          `The run of sample ${sorted[begin].sample} spans working positions ${begin} through ${end - 1}; its length ${end - begin} is that bin's count.`, false, { row: "working", index: begin }));
        begin = end;
      }
      phases.push({ label: "Count runs", description: "A consecutive run's length gives the count for its bin. Bins with no samples remain zero. CUB uses run-boundary information to compute these counts.", tokens: [...source, ...working, ...run_tokens, ...bin_tokens()] });
    }
    const owners = [];
    const output = [];
    for (let thread = 0; thread < threads; thread += 1) {
      for (let item = 0; item < bins_per_thread; item += 1) {
        const bin = thread + item * threads;
        const position = thread * bins_per_thread + item;
        const padding = bin >= bins;
        const count = padding ? 0 : histogram[bin];
        const detail = padding ? `T${thread}, slot ${item}: projected bin ${bin} is beyond bins = ${bins}; this output slot is a defined zero.` :
          `T${thread}, slot ${item} owns bin ${thread} + ${item} * ${threads} = ${bin}, whose count is ${count}. Use striped Store to write it at destination[${bin}].`;
        owners.push(token(`owner-${position}`, padding ? "·" : bin, "owners", position, thread, detail, padding));
        output.push(token(`output-${position}`, count, "output", position, thread, detail, padding,
          padding ? undefined : { row: "bins", index: bin }));
      }
    }
    phases.push({ label: "Striped result", description: `Member t receives bin t + i * ${threads} in local slot i. ${threads * bins_per_thread - bins} output slots beyond bins are defined zeros. Input samples remain unchanged.`, tokens: [...source, ...bin_tokens(), ...owners, ...output] });
    return {
      rows, phases,
      detail: `${capacity} samples across ${threads} threads; ${bins} bins; ${bins_per_thread} returned counters per thread. Algorithm: ${state.algorithm}.`,
      summary: `Counts in bin order: [${histogram.join(", ")}]. Sum = ${capacity}, one contribution per input sample.`,
      notes: [
        "The bin count and counters per thread are compile-time controls. Eight times counters per thread must cover every bin in this teaching block.",
        "Returned counters use striped ownership: bin = thread rank + local slot * block size. Output padding is zero; it is safe to read but is outside the requested bin range.",
        "Every input slot is counted. Padding an incomplete input tile with sample zero adds unwanted counts to bin zero; Histogram has no valid_items argument.",
        "Each call starts a fresh histogram. To accumulate tiles, add their returned per-bin counts and choose a counter dtype wide enough for the total.",
        "Atomic rounds and sort/run stages explain the algorithms. They are not a trace of GPU scheduling, exact CUB instructions, or relative performance.",
      ],
      caption: "One complete one-dimensional block with eight teaching threads. Common calls accept fixed-size ThreadData; the qualified Numba API also accepts scalars and fixed local arrays. All return fresh counter payloads. Select a value to inspect it; arrow keys move between values.",
    };
  }

  window.CoopExplorer.register("histogram", {
    title: "Count samples into bins", eyebrow: "Fresh counts and striped ownership", defaultAlgorithm: "atomic",
    algorithms: [
      { id: "atomic", label: "Atomic increments", tag: "algorithm=atomic" },
      { id: "sort", label: "Sort and count runs", tag: "algorithm=sort" },
    ],
    controls: [
      { id: "items", label: "Samples per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "bins", label: "Bins", value: "13", choices: ["1", "5", "8", "13", "16"] },
      { id: "bins_per_thread", label: "Counters per thread", value: "2", choices: state => Array.from({ length: 4 - Math.ceil(Number(state.bins) / threads) }, (_, index) => String(index + Math.ceil(Number(state.bins) / threads))) },
      { id: "pattern", label: "Samples", value: "mixed", choices: [choice("mixed", "Mixed bins"), choice("hot", "Mostly one bin"), choice("zero", "All in bin zero")] },
    ],
    build: build_histogram,
  });
})();
