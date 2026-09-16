// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
// Adapted from python/cuda_coop/docs/fern/fern/components/CooperativeDataMotion.tsx
// at cccl-mirror trentn/dev/cuda-coop 5dba3d36b6eaae48b967d6fa48f9d15e98136000.

(() => {
  "use strict";

  const threads = 8;
  const warp_threads = 4;
  const colors = ["#76b900", "#00a9ce", "#f6b21a", "#d6538f", "#9d80d7", "#e86f18", "#2a9d8f", "#8198a7"];
  const algorithms = {
    direct: {
      label: "Direct", tag: "Blocked scalar loads", access: "blocked", result: "blocked",
      detail: "Each thread loads consecutive items. At each scalar issue step, adjacent threads access addresses separated by the items-per-thread count.",
      scratch: "No exchange scratch.", caution: "Larger per-thread counts can reduce memory-transaction utilization.",
    },
    striped: {
      label: "Striped", tag: "Striped loads and registers", access: "striped", result: "striped",
      detail: "Adjacent threads load adjacent addresses at each issue step. Values remain distributed in stripes across the threads.",
      scratch: "No exchange scratch.", caution: "A following operation that expects blocked ownership needs an explicit conversion.",
    },
    vectorize: {
      label: "Vectorize", tag: "Blocked vector candidates", access: "blocked", result: "blocked",
      detail: "Each thread owns consecutive items that can be grouped into vector loads. The final register ownership matches Direct.",
      scratch: "No exchange scratch.", caution: "Vectorization depends on the type, pointer, alignment, and item count. The diagram does not promise a particular instruction.",
    },
    transpose: {
      label: "Transpose", tag: "Striped loads, blocked registers", access: "striped", result: "blocked", exchange: true,
      detail: "Threads first load striped values. A block-wide shared-memory exchange redistributes them into consecutive per-thread registers.",
      scratch: "Shared scratch for the block exchange; actual storage can include padding.",
      caution: "The exchange adds shared-memory access and synchronization.",
    },
    warp_transpose: {
      label: "Warp transpose", tag: "Warp-striped loads and exchange", access: "warp-striped", result: "blocked", exchange: true,
      detail: "Each warp loads its own tile in stripes, then exchanges values within that warp to produce blocked registers.",
      scratch: "Separate shared scratch for every warp in the block; padding is omitted here.",
      caution: "The real block size must be a multiple of the 32-lane physical warp.",
    },
    warp_transpose_timesliced: {
      label: "Warp transpose, timesliced", tag: "Warps reuse shared scratch", access: "warp-striped", result: "blocked", exchange: true, timesliced: true,
      detail: "All warps load warp-striped values into registers first. Warps then take turns using the same shared scratch for their exchange.",
      scratch: "One warp's shared scratch, reused across exchange rounds; padding is omitted here.",
      caution: "Requires complete physical warps. Less scratch comes with additional serialized exchange rounds.",
    },
  };

  function ownership(layout, value, items) {
    if (layout === "blocked") return [Math.floor(value / items), value % items];
    if (layout === "striped") return [value % threads, Math.floor(value / threads)];
    const warp_size = warp_threads * items;
    return [Math.floor(value / warp_size) * warp_threads + value % warp_threads, Math.floor((value % warp_size) / warp_threads)];
  }

  function svg_element(tag, attributes = {}, text = null) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    if (text !== null) element.textContent = text;
    return element;
  }

  function mount(host, index) {
    const wrapper = host.closest(".coop-visualization");
    const reduced_motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const state = { algorithm: "transpose", items: 2, value: 5, thread: 0, phase: 0, playing: false };
    const prefix = `coop-load-${index}`;
    let phases = [];
    let geometry;
    let tokens = [];
    let timer = null;
    let visible = true;
    const root = document.createElement("section");
    root.className = "coop-load";
    root.setAttribute("aria-label", "Cooperative load explorer");
    root.innerHTML = `
      <div class="coop-load-heading">
        <div><p class="coop-load-eyebrow">Memory to registers</p><h2>Follow a cooperative load</h2></div>
        <div class="coop-load-transport" role="group" aria-label="Animation controls">
          <button type="button" data-action="previous" aria-label="Previous step">Previous</button>
          <button type="button" data-action="play" aria-label="Play animation">Play</button>
          <button type="button" data-action="next" aria-label="Next step">Next</button>
          <button type="button" data-action="reset">Reset</button>
        </div>
      </div>
      <div class="coop-load-options" role="group" aria-label="Load algorithm"></div>
      <div class="coop-load-settings">
        <label for="${prefix}-items">Items per thread <select id="${prefix}-items" data-items><option>1</option><option>2</option><option>4</option></select></label>
        <label for="${prefix}-value">Inspect value <select id="${prefix}-value" data-value></select></label>
        <label for="${prefix}-thread">Inspect result thread <select id="${prefix}-thread" data-thread></select></label>
        <span data-tile-size></span>
      </div>
      <p class="coop-load-detail" data-detail></p>
      <div class="coop-load-phases" role="group" aria-label="Load phases"></div>
      <div class="coop-load-scroll" tabindex="0" role="region" aria-label="Load ownership diagram; scroll horizontally to see all threads">
        <svg class="coop-load-diagram" role="group" aria-labelledby="${prefix}-title" aria-describedby="${prefix}-description"></svg>
      </div>
      <p class="coop-load-status" data-status aria-live="polite" aria-atomic="true"></p>
      <div class="coop-load-inspection">
        <div><h3>Selected value</h3><p data-inspection></p></div>
        <div><h3>Result registers</h3><p data-thread-inspection></p></div>
      </div>
      <div class="coop-load-notes"><p data-scratch></p><p data-caution></p></div>
      <p class="coop-load-caption">Teaching model: eight threads, with four lanes per displayed warp. Physical CUDA warps have 32 lanes. Colors identify each value's original blocked group. The phases illustrate ownership, not instruction timing or transaction counts. Click a value to follow its path; arrow keys move between values.</p>`;

    const query = (selector) => root.querySelector(selector);
    const svg = query("svg.coop-load-diagram");
    const option_grid = query(".coop-load-options");
    for (const [name, option] of Object.entries(algorithms)) {
      const button = document.createElement("button");
      button.type = "button";
      button.dataset.algorithm = name;
      const title = document.createElement("strong");
      title.textContent = option.label;
      const tag = document.createElement("span");
      tag.textContent = option.tag;
      button.append(title, tag);
      option_grid.append(button);
    }
    for (let thread = 0; thread < threads; ++thread) query("[data-thread]").add(new Option(`T${thread}`, thread));

    function read_url() {
      const params = new URL(window.location.href).searchParams;
      const algorithm = params.get("load-algorithm");
      state.algorithm = Object.hasOwn(algorithms, algorithm) ? algorithm : "transpose";
      const items = Number(params.get("load-items"));
      state.items = [1, 2, 4].includes(items) ? items : 2;
      for (const [name, fallback, limit] of [["value", 5, threads * state.items], ["thread", 0, threads]]) {
        const raw = params.get(`load-${name}`);
        const value = raw === null || raw.trim() === "" ? NaN : Number(raw);
        state[name] = Number.isInteger(value) && value >= 0 && value < limit ? value : fallback;
      }
      state.phase = 0;
      state.playing = false;
    }

    function write_url() {
      const url = new URL(window.location.href);
      for (const name of ["algorithm", "items", "value", "thread"]) url.searchParams.set(`load-${name}`, state[name]);
      // A locally opened HTML build can disallow history updates; the explorer still works.
      try { window.history.replaceState(window.history.state, "", url); } catch { /* Keep local controls usable. */ }
    }

    function register_point(layout, value, y) {
      const [thread, slot] = ownership(layout, value, state.items);
      return [geometry.register_x + thread * geometry.thread_step + 8 + slot * 34, y];
    }

    function point(value, location) {
      if (location === "memory") return [geometry.memory_x + value * 34, 56];
      if (location === "read") return register_point(algorithms[state.algorithm].access, value, 178);
      if (location === "scratch") {
        const slot = algorithms[state.algorithm].timesliced ? value % (warp_threads * state.items) : value;
        return [geometry.scratch_x + slot * 34, 306];
      }
      return register_point(algorithms[state.algorithm].result, value, geometry.output_y);
    }

    function add_text(x, y, content, class_name) {
      svg.append(svg_element("text", { x, y, class: class_name }, content));
    }

    function draw_registers(y, title, layout) {
      add_text(20, y - 44, title, "coop-load-axis");
      for (let thread = 0; thread < threads; ++thread) {
        const x = geometry.register_x + thread * geometry.thread_step;
        svg.append(svg_element("rect", { x, y: y - 26, width: geometry.card_width, height: 70, rx: 7, class: "coop-load-card" }));
        add_text(x + 8, y - 10, `T${thread}`, "coop-load-thread-label");
      }
      for (let value = 0; value < threads * state.items; ++value) {
        const [x] = register_point(layout, value, y);
        svg.append(svg_element("rect", { x, y, width: 28, height: 28, rx: 4, class: "coop-load-slot" }));
      }
    }

    function build_scene() {
      const option = algorithms[state.algorithm];
      const total = threads * state.items;
      geometry = { card_width: state.items * 34 + 10, thread_step: state.items * 34 + 24 };
      geometry.width = Math.max(850, threads * geometry.thread_step + 32);
      geometry.register_x = (geometry.width - (threads * geometry.thread_step - 14)) / 2;
      geometry.memory_x = (geometry.width - (total * 34 - 6)) / 2;
      geometry.scratch_x = (geometry.width - ((option.timesliced ? warp_threads * state.items : total) * 34 - 6)) / 2;
      geometry.output_y = option.exchange ? 438 : 306;
      phases = [
        { label: "Memory", positions: ["memory", "memory"], description: "The full input tile starts in consecutive memory locations." },
        { label: "Load into registers", positions: ["read", "read"], description: `All displayed threads have loaded their ${option.access} values into registers.` },
      ];
      if (option.timesliced) {
        phases.push(
          { label: "Warp 0 exchange", positions: ["scratch", "read"], description: "Displayed warp 0 uses shared scratch. Warp 1 keeps its already-loaded values in registers." },
          { label: "Warp 1 exchange", positions: ["output", "scratch"], description: "Warp 0 has completed. Displayed warp 1 reuses the same shared scratch for its exchange." },
        );
      } else if (option.exchange) {
        phases.push({ label: "Shared exchange", positions: ["scratch", "scratch"], description: "Loaded values pass through shared scratch at their logical positions. Storage padding is omitted." });
      }
      phases.push({ label: "Result registers", positions: ["output", "output"], description: `${option.result === "blocked" ? "Consecutive" : "Striped"} per-thread values are ready for the next operation.` });
      query(".coop-load-phases").replaceChildren(...phases.map((phase, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.phase = index;
        button.textContent = `${index + 1}. ${phase.label}`;
        return button;
      }));
      query("[data-value]").replaceChildren(...Array.from({ length: total }, (_, value) => new Option(value, value)));
      svg.replaceChildren();
      svg.setAttribute("viewBox", `0 0 ${geometry.width} ${geometry.output_y + 60}`);
      svg.style.minWidth = `${geometry.width * 0.8}px`;
      svg.append(svg_element("title", { id: `${prefix}-title` }, `${option.label}: memory to register ownership`));
      svg.append(svg_element("desc", { id: `${prefix}-description` }, `${total} values across eight illustrative threads, ${state.items} per thread. The next paragraphs describe the selected value and result thread.`));
      add_text(20, 24, "Global memory · logical item index", "coop-load-axis");
      for (let value = 0; value < total; ++value) {
        const [x, y] = point(value, "memory");
        svg.append(svg_element("rect", { x, y, width: 28, height: 28, rx: 4, class: "coop-load-slot" }));
        add_text(x + 14, y - 6, value, "coop-load-address");
      }
      if (state.algorithm === "vectorize" && state.items > 1) {
        for (let thread = 0; thread < threads; ++thread) {
          svg.append(svg_element("rect", { x: geometry.memory_x + thread * state.items * 34 - 3, y: 52, width: state.items * 34, height: 36, rx: 5, class: "coop-load-vector" }));
        }
      }
      draw_registers(178, `Loaded registers · ${option.access} ownership`, option.access);
      if (option.exchange) {
        add_text(20, 263, option.timesliced ? "Shared scratch · reused by one warp at a time" : "Shared scratch · logical positions (padding omitted)", "coop-load-axis");
        const count = option.timesliced ? warp_threads * state.items : total;
        for (let slot = 0; slot < count; ++slot) {
          const x = geometry.scratch_x + slot * 34;
          svg.append(svg_element("rect", { x, y: 306, width: 28, height: 28, rx: 4, class: "coop-load-slot coop-load-shared-slot" }));
          add_text(x + 14, 298, slot, "coop-load-address");
        }
      }
      draw_registers(geometry.output_y, `Result registers · ${option.result} ownership`, option.result);
      if (option.access === "warp-striped") {
        for (let warp = 0; warp < 2; ++warp) {
          add_text(geometry.register_x + warp * warp_threads * geometry.thread_step + 8, 229, `Displayed warp ${warp} · T${warp * 4}–T${warp * 4 + 3}`, "coop-load-warp-label");
        }
      }
      svg.append(svg_element("path", { class: "coop-load-path", "aria-hidden": "true" }));
      tokens = Array.from({ length: total }, (_, value) => {
        const [owner_thread, owner_slot] = ownership(option.result, value, state.items);
        const [access_thread, access_slot] = ownership(option.access, value, state.items);
        const token = svg_element("g", {
          role: "button", "data-value-index": value, "data-owner-thread": owner_thread,
          "data-owner-slot": owner_slot, "data-access-thread": access_thread, "data-access-slot": access_slot,
          "aria-label": `Value ${value}: loaded by T${access_thread}, slot ${access_slot}; result T${owner_thread}, slot ${owner_slot}`,
          class: "coop-load-token",
        });
        token.style.setProperty("--coop-token-color", colors[Math.floor(value / state.items)]);
        token.append(svg_element("rect", { x: -3, y: -3, width: 34, height: 34, rx: 7, class: "coop-load-token-outline" }));
        token.append(svg_element("rect", { width: 28, height: 28, rx: 4, class: "coop-load-token-body" }));
        token.append(svg_element("text", { x: 14, y: 19, "text-anchor": "middle" }, value));
        svg.append(token);
        return token;
      });
      query("[data-detail]").textContent = option.detail;
      query("[data-scratch]").textContent = option.scratch;
      query("[data-caution]").textContent = state.algorithm === "vectorize" && state.items === 1 ? "One item per thread has no multi-item vector bundle in this illustration. All three ownership layouts coincide." : option.caution;
      query("[data-tile-size]").textContent = `8 threads × ${state.items} = ${total} values`;
      for (const button of option_grid.children) button.setAttribute("aria-pressed", button.dataset.algorithm === state.algorithm);
    }

    function render_frame() {
      const option = algorithms[state.algorithm];
      const phase = phases[state.phase];
      for (const button of query(".coop-load-phases").children) {
        if (Number(button.dataset.phase) === state.phase) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      }
      tokens.forEach((token, value) => {
        const warp = Math.floor(value / (warp_threads * state.items));
        const location = phase.positions[warp];
        const [x, y] = point(value, location);
        token.style.transform = `translate(${x}px, ${y}px)`;
        token.dataset.location = location;
        token.dataset.selected = value === state.value;
        token.setAttribute("aria-pressed", value === state.value);
        token.setAttribute("tabindex", value === state.value ? "0" : "-1");
      });
      const route = ["memory", "read", ...(option.exchange ? ["scratch"] : []), "output"].map((location) => point(state.value, location));
      query(".coop-load-path").setAttribute("d", route.map(([x, y], index) => `${index ? "L" : "M"} ${x + 14} ${y + 14}`).join(" "));
      const [access_thread, access_slot] = ownership(option.access, state.value, state.items);
      const [owner_thread, owner_slot] = ownership(option.result, state.value, state.items);
      query("[data-inspection]").textContent = `Value ${state.value}: memory[${state.value}] → T${access_thread}, slot ${access_slot}${option.exchange ? " → shared scratch" : ""} → T${owner_thread}, slot ${owner_slot}.`;
      const values = Array.from({ length: state.items }, (_, slot) => tokens.findIndex((token) => Number(token.dataset.ownerThread) === state.thread && Number(token.dataset.ownerSlot) === slot));
      query("[data-thread-inspection]").textContent = `T${state.thread} receives [${values.join(", ")}] in slots 0${state.items > 1 ? `–${state.items - 1}` : ""}.`;
      query("[data-status]").setAttribute("aria-live", state.playing ? "off" : "polite");
      query("[data-status]").textContent = `Step ${state.phase + 1} of ${phases.length}. ${phase.description}`;
      query("[data-items]").value = state.items;
      query("[data-value]").value = state.value;
      query("[data-thread]").value = state.thread;
      query("[data-action=play]").textContent = state.playing ? "Pause" : "Play";
      query("[data-action=play]").setAttribute("aria-label", state.playing ? "Pause animation" : "Play animation");
    }

    function sync_timer() {
      window.clearInterval(timer);
      timer = null;
      if (state.playing && visible && !document.hidden) {
        timer = window.setInterval(() => {
          state.phase = (state.phase + 1) % phases.length;
          render_frame();
        }, 1600);
      }
    }

    function choose_value(value, focus = false) {
      state.value = value;
      state.thread = ownership(algorithms[state.algorithm].result, value, state.items)[0];
      render_frame();
      write_url();
      if (focus) tokens[value].focus();
    }

    root.addEventListener("click", (event) => {
      const token = event.target.closest("[data-value-index]");
      if (token) { choose_value(Number(token.dataset.valueIndex)); return; }
      const button = event.target.closest("button");
      if (!button) return;
      if (button.dataset.algorithm) {
        state.algorithm = button.dataset.algorithm;
        state.phase = 0;
        state.playing = false;
        build_scene();
        write_url();
      } else if (button.dataset.phase !== undefined) {
        state.phase = Number(button.dataset.phase);
        state.playing = false;
      } else if (button.dataset.action === "play") {
        state.playing = !state.playing;
      } else {
        state.playing = false;
        if (button.dataset.action === "reset") state.phase = 0;
        else state.phase = (state.phase + (button.dataset.action === "next" ? 1 : phases.length - 1)) % phases.length;
      }
      render_frame();
      sync_timer();
    });
    root.addEventListener("change", (event) => {
      if (event.target.matches("[data-items]")) {
        state.items = Number(event.target.value);
        state.value = Math.min(state.value, threads * state.items - 1);
        state.phase = 0;
        state.playing = false;
        build_scene();
      } else if (event.target.matches("[data-value]")) {
        choose_value(Number(event.target.value));
        return;
      } else if (event.target.matches("[data-thread]")) state.thread = Number(event.target.value);
      render_frame();
      sync_timer();
      write_url();
    });
    root.addEventListener("keydown", (event) => {
      const token = event.target.closest("[data-value-index]");
      if (!token) return;
      const value = Number(token.dataset.valueIndex);
      let next = value;
      if (["ArrowRight", "ArrowDown"].includes(event.key)) next = (value + 1) % tokens.length;
      else if (["ArrowLeft", "ArrowUp"].includes(event.key)) next = (value + tokens.length - 1) % tokens.length;
      else if (event.key === "Home") next = 0;
      else if (event.key === "End") next = tokens.length - 1;
      else if (event.key !== "Enter" && event.key !== " ") return;
      event.preventDefault();
      choose_value(next, true);
    });

    read_url();
    build_scene();
    render_frame();
    host.replaceChildren(root);
    wrapper?.classList.add("coop-load-ready");
    window.addEventListener("popstate", () => { read_url(); build_scene(); render_frame(); sync_timer(); });
    document.addEventListener("visibilitychange", sync_timer);
    reduced_motion.addEventListener("change", () => {
      if (reduced_motion.matches) { state.playing = false; render_frame(); sync_timer(); }
    });
    if (typeof IntersectionObserver === "function") {
      const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync_timer(); });
      observer.observe(host);
    }
  }

  function initialize() {
    document.querySelectorAll("[data-coop-load]").forEach((host, index) => {
      try { mount(host, index); } catch (error) {
        // The adjacent static explanation remains available when enhancement fails.
        console.warn("The cooperative load explorer could not be initialized.", error);
      }
    });
  }
  if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", initialize);
  else initialize();
})();
