// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Blocked radix semantics follow BlockRadixRank and BlockRadixSort. The
// unsigned teaching keys avoid conflating raw signed bits with ordered bits.
(() => {
  "use strict";

  const threads = 8;
  const choice = (value, label) => ({ value, label });
  const groups = items => Array.from({ length: threads }, (_, thread) => ({ start: thread * items, count: items, label: `T${thread}` }));
  const key_pattern = [35, 18, 33, 2, 35, 17, 1, 18, 3, 34, 16, 2, 19, 32, 1, 17];

  function input_entries(items) {
    return Array.from({ length: threads * items }, (_, origin) => ({
      key: key_pattern[origin % key_pattern.length], origin,
    }));
  }

  function digit(key, begin, end) {
    return (key >>> begin) & ((1 << (end - begin)) - 1);
  }

  function rank_digit(entries, begin, end, descending) {
    const digits = entries.map(entry => digit(entry.key, begin, end));
    const counts = Array(1 << (end - begin)).fill(0);
    for (const value of digits) ++counts[value];
    const prefixes = Array(counts.length).fill(0);
    let prefix = 0;
    for (let index = 0; index < counts.length; ++index) {
      const bin = descending ? counts.length - index - 1 : index;
      prefixes[bin] = prefix;
      prefix += counts[bin];
    }
    const seen = Array(counts.length).fill(0);
    const ranks = digits.map(value => prefixes[value] + seen[value]++);
    return { digits, counts, prefixes, ranks };
  }

  function token(id, value, row, index, color, detail, from) {
    return { id, label: String(value), value, row, index, color, detail, from };
  }

  function input_tokens(entries, items, pairs) {
    const keys = entries.map(entry => token(`input-key-${entry.origin}`, entry.key, "input", entry.origin,
      Math.floor(entry.origin / items),
      `Original key ${entry.key} at blocked position ${entry.origin}: T${Math.floor(entry.origin / items)}, slot ${entry.origin % items}. This input stays unchanged.`));
    if (pairs) keys.push(...entries.map(entry => token(`input-value-${entry.origin}`, entry.origin, "input-values", entry.origin,
      Math.floor(entry.origin / items), `Associated value ${entry.origin} identifies this key's original position. The input values stay unchanged.`)));
    return keys;
  }

  function working_tokens(entries, items, pairs, striped = false) {
    const result = [];
    entries.forEach((entry, position) => {
      const index = striped ? (position % threads) * items + Math.floor(position / threads) : position;
      const owner = `T${Math.floor(index / items)}, slot ${index % items}`;
      const color = Math.floor(entry.origin / items);
      result.push(token(`key-${entry.origin}`, entry.key, "result", index, color,
        `Key ${entry.key}, originally at blocked position ${entry.origin}, now occupies ordered position ${position}: ${owner}.`,
        { row: "input", index: entry.origin }));
      if (pairs) result.push(token(`value-${entry.origin}`, entry.origin, "result-values", index, color,
        `Associated value ${entry.origin} travels with key ${entry.key} to ordered position ${position}: ${owner}.`,
        { row: "input-values", index: entry.origin }));
    });
    return result;
  }

  function digit_tokens(entries, result, begin, end, items) {
    return entries.map((entry, index) => token(`digit-${entry.origin}`, result.digits[index], "digit", index,
      Math.floor(entry.origin / items),
      `Key ${entry.key} (original position ${entry.origin}): bits [${begin}, ${end}) give digit ${result.digits[index]}. Its low eight bits are ${entry.key.toString(2).padStart(8, "0")}.`));
  }

  function rank_tokens(entries, result, items) {
    return entries.map((entry, index) => {
      const value = result.digits[index];
      const earlier = result.ranks[index] - result.prefixes[value];
      return token(`rank-${entry.origin}`, result.ranks[index], "ranks", index, Math.floor(entry.origin / items),
        `Key ${entry.key} (original position ${entry.origin}), digit ${value}: bin prefix ${result.prefixes[value]} + ${earlier} earlier equal-digit item${earlier === 1 ? "" : "s"} = destination ${result.ranks[index]}. The rank stays in its source slot.`);
    });
  }

  function build_sort(state, entries, items, begin, end, descending) {
    const pairs = state.payload === "pairs";
    const striped = state.output === "striped";
    const preserved = input_tokens(entries, items, pairs);
    const rows = [{ id: "input", label: "Input keys · blocked · preserved", count: entries.length, groups: groups(items) }];
    if (pairs) rows.push({ id: "input-values", label: "Input values · original positions · preserved", count: entries.length, groups: groups(items) });
    rows.push(
      { id: "digit", label: "Current digit · one per working key in blocked order", count: entries.length, groups: groups(items) },
      { id: "ranks", label: "Destination for this digit pass · one per working key", count: entries.length, groups: groups(items) },
      { id: "result", label: `Working keys · returned ${striped ? "striped" : "blocked"} after the final pass`, count: entries.length, groups: groups(items) },
    );
    if (pairs) rows.push({ id: "result-values", label: "Working values · travel with their keys", count: entries.length, groups: groups(items) });
    const phases = [{ label: "Inputs", description: "The block starts with keys in blocked order. Each input payload is preserved throughout the call.", tokens: preserved }];
    let ordered = entries.slice();
    for (let bit = begin, pass = 1; bit < end; bit += 4, ++pass) {
      const pass_end = Math.min(bit + 4, end);
      const ranked = rank_digit(ordered, bit, pass_end, descending);
      const working = working_tokens(ordered, items, pairs);
      const digits = digit_tokens(ordered, ranked, bit, pass_end, items);
      const ranks = rank_tokens(ordered, ranked, items);
      phases.push({
        label: `Pass ${pass}: digit`,
        description: `Read bits [${bit}, ${pass_end}) from the working keys. Passes start at begin_bit and advance toward more significant bits.`,
        tokens: [...preserved, ...working, ...digits],
      }, {
        label: `Pass ${pass}: rank`,
        description: `Count ${descending ? "greater" : "smaller"} digits, then earlier equal digits in the current blocked sequence. Each key gets a unique destination.`,
        tokens: [...preserved, ...working, ...digits, ...ranks],
      });
      const next = Array(ordered.length);
      ordered.forEach((entry, index) => { next[ranked.ranks[index]] = entry; });
      ordered = next;
      const final = pass_end === end;
      phases.push({
        label: final ? "Return sorted payload" : `Pass ${pass}: scatter`,
        description: `Move each key${pairs ? " and its associated value" : ""} to its digit rank.${final && striped ? " The final pass returns striped ownership through cuda.coop.numba_mlir; intermediate passes use blocked ownership." : " Results use blocked ownership."} Equal digits retain their previous order.`,
        tokens: [...preserved, ...working_tokens(ordered, items, pairs, final && striped)],
      });
    }
    return {
      rows, phases,
      detail: `${pairs ? "radix_sort_pairs" : "radix_sort_keys"}: ${descending ? "descending" : "ascending"} bits [${begin}, ${end}), ${Math.ceil((end - begin) / 4)} digit pass${end - begin <= 4 ? "" : "es"}.`,
      summary: `Logical sorted keys: [${ordered.map(entry => entry.key).join(", ")}].${pairs ? ` Associated values: [${ordered.map(entry => entry.origin).join(", ")}].` : ""} Returned ownership is ${striped ? "striped (qualified API)" : "blocked"}; inputs remain unchanged.`,
      notes: [
        "Every pass is stable. Ordering the least significant selected digit first lets later passes preserve the order established by earlier digits. Equal complete selected bit fields retain original blocked input order, also in descending mode.",
        "The current provider uses up to four bits per sort pass. This pass width is an implementation choice, not a radix_sort argument. The stages show mathematical ranks and movement, not exact CUB instructions or scratch layout.",
        striped ? `Only cuda.coop.numba_mlir exposes blocked_to_striped=True. Sorted position p returns at thread p % ${threads}, slot p // ${threads}. Store with matching striped ownership.` : `Blocked output puts sorted position p at thread p // ${items}, slot p % ${items}. The common API always returns blocked output.`,
      ],
    };
  }

  function build_rank(entries, items, begin, end, descending) {
    const result = rank_digit(entries, begin, end, descending);
    const preserved = input_tokens(entries, items, false);
    const digits = digit_tokens(entries, result, begin, end, items);
    const bins = result.counts.flatMap((count, bin) => count > 0 ? [bin] : []);
    const bin_groups = bins.map((bin, index) => ({ start: index, count: 1, label: `d${bin}` }));
    const counts = bins.map((bin, index) => token(`count-${bin}`, result.counts[bin], "counts", index, index,
      `Digit ${bin} occurs ${result.counts[bin]} times. Empty bins are omitted from this picture.`));
    const prefixes = bins.map((bin, index) => token(`prefix-${bin}`, result.prefixes[bin], "prefixes", index, index,
      `Digit ${bin} starts at destination ${result.prefixes[bin]}: the number of keys with ${descending ? "greater" : "smaller"} digits.`));
    return {
      detail: `radix_rank: one ${end - begin}-bit digit [${begin}, ${end}), ${descending ? "descending" : "ascending"} destinations without moving keys.`,
      rows: [
        { id: "input", label: "Input keys · blocked · preserved", count: entries.length, groups: groups(items) },
        { id: "digit", label: "Selected digit · stays with each original key", count: entries.length, groups: groups(items) },
        { id: "counts", label: "Counts by digit · occupied bins only", count: bins.length, groups: bin_groups },
        { id: "prefixes", label: "Bin starts · number of keys ordered before this digit", count: bins.length, groups: bin_groups },
        { id: "ranks", label: "Returned int32 ranks · original blocked slots", count: entries.length, groups: groups(items) },
      ],
      phases: [
        { label: "Inputs", description: "Read keys in blocked order: all of T0's items, then T1's, and so on.", tokens: preserved },
        { label: "Extract digit", description: `Extract bits [${begin}, ${end}) from each unsigned key. Bits outside this interval do not affect the result.`, tokens: [...preserved, ...digits] },
        { label: "Count and prefix", description: `Each bin's start counts keys with ${descending ? "greater" : "smaller"} digits. Bins are displayed in increasing digit order for both directions.`, tokens: [...preserved, ...digits, ...counts, ...prefixes] },
        { label: "Return ranks", description: "Add the bin start to the count of earlier equal-digit keys. Return that int32 destination in the original source slot; the input keys do not move.", tokens: [...preserved, ...digits, ...counts, ...prefixes, ...rank_tokens(entries, result, items)] },
      ],
      summary: `Ranks in original blocked input order: [${result.ranks.join(", ")}]. These are destinations, not reordered keys.`,
      notes: [
        "Each destination equals the number of keys with a preceding digit plus the number of earlier input keys with the same digit. Equal digits keep their original blocked order, even when descending=True.",
        "Radix Rank returns one rank per key for one digit. It does not perform the scatter or the sequence of passes that Radix Sort performs. There is no pairs or striped-output rank option.",
        "The count and prefix rows explain the mathematics. The common API returns only ranks. The qualified exclusive_digit_prefix side output is optional and has a separate per-thread bin layout; it is not enabled by this picture.",
      ],
    };
  }

  function build(state) {
    const items = Number(state.items);
    const begin = Number(state.begin);
    const end = Number(state.end);
    const entries = input_entries(items);
    const descending = state.order === "descending";
    const model = state.algorithm === "rank" ? build_rank(entries, items, begin, end, descending) : build_sort(state, entries, items, begin, end, descending);
    model.caption = "Eight teaching threads represent one complete block. Keys are unsigned uint32 examples between 0 and 255; controls cover their low eight bits. Actual integral key types have 32 or 64 bits; Rank selects at most eight bits per call. Colors track original owning threads. Select a value for its key, digit, or destination; arrow keys move between values.";
    return model;
  }

  window.CoopExplorer.register("radix", {
    title: "Follow radix ordering", eyebrow: "Digits, destinations, and stable passes", defaultAlgorithm: "sort",
    algorithms: [
      { id: "sort", label: "Radix Sort", tag: "Stable digit passes and movement" },
      { id: "rank", label: "Radix Rank", tag: "One digit's destinations; keys stay put" },
    ],
    controls: [
      { id: "items", label: "Items per thread", value: "2", choices: ["1", "2", "4"] },
      { id: "payload", label: "Payload", value: "pairs", choices: state => state.algorithm === "rank" ? [choice("keys", "Keys")] : [choice("keys", "Keys"), choice("pairs", "Keys and associated values")], hidden: state => state.algorithm === "rank" },
      { id: "order", label: "Order", value: "ascending", choices: [choice("ascending", "Ascending"), choice("descending", "Descending")] },
      { id: "begin", label: "begin_bit (inclusive)", value: "0", choices: ["0", "1", "2", "3", "4", "5", "6", "7"] },
      { id: "end", label: "end_bit (exclusive)", value: "8", choices: state => Array.from({ length: 8 - Number(state.begin) }, (_, index) => String(Number(state.begin) + index + 1)) },
      { id: "output", label: "Output ownership", value: "blocked", choices: state => [choice("blocked", "Blocked (common API)"), ...(state.algorithm === "rank" ? [] : [choice("striped", "Striped (qualified API)")])], hidden: state => state.algorithm === "rank" },
    ],
    build,
  });
})();
