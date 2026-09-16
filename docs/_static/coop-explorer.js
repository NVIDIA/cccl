// Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

// Shared controls and SVG rendering, following the Load explorer.
(() => {
  "use strict";

  const colors = ["#76b900", "#00a9ce", "#f6b21a", "#d6538f", "#9d80d7", "#e86f18", "#2a9d8f", "#8198a7"];
  const definitions = new Map();
  const options = (source, state) => (typeof source === "function" ? source(state) : source)
    .map((item) => typeof item === "object" ? { ...item, value: String(item.value) } : { value: String(item), label: String(item) });

  function svg_element(tag, attributes = {}, label) {
    const element = document.createElementNS("http://www.w3.org/2000/svg", tag);
    for (const [name, value] of Object.entries(attributes)) element.setAttribute(name, value);
    if (label !== undefined) element.textContent = label;
    return element;
  }

  function mount(host, name, definition, instance) {
    const reduced_motion = window.matchMedia("(prefers-reduced-motion: reduce)");
    const prefix = `coop-${name}-${instance}`;
    const controls = definition.controls || [];
    const state = {};
    let model;
    let phase = 0;
    let playing = !reduced_motion.matches;
    let visible = true;
    let timer = null;
    let selected = null;
    let geometry;
    let token_elements = new Map();
    const root = document.createElement("section");
    root.className = "coop-load coop-explorer";
    root.setAttribute("aria-label", definition.title);
    root.innerHTML = `
      <div class="coop-load-heading">
        <div><p class="coop-load-eyebrow" data-eyebrow></p><h2 data-title></h2></div>
        <div class="coop-load-transport" role="group" aria-label="Animation controls">
          <button type="button" data-action="previous" aria-label="Previous step">Previous</button>
          <button type="button" data-action="play" aria-label="Pause animation">Pause</button>
          <button type="button" data-action="next" aria-label="Next step">Next</button>
          <button type="button" data-action="reset">Reset</button>
        </div>
      </div>
      <div class="coop-load-options" role="group" aria-label="Algorithm"></div>
      <div class="coop-load-settings"></div>
      <p class="coop-load-detail" data-detail></p>
      <div class="coop-load-phases" role="group" aria-label="Algorithm phases"></div>
      <div class="coop-load-scroll" tabindex="0" role="region" aria-label="Algorithm diagram; scroll horizontally to see all values">
        <svg class="coop-load-diagram" role="group" aria-labelledby="${prefix}-title" aria-describedby="${prefix}-description"></svg>
      </div>
      <p class="coop-load-status" data-status aria-live="off" aria-atomic="true"></p>
      <div class="coop-load-inspection">
        <div><h3>Selected value</h3><p data-inspection></p></div>
        <div><h3>Result</h3><p data-summary></p></div>
      </div>
      <div class="coop-load-notes" data-notes></div>
      <p class="coop-load-caption" data-caption></p>`;
    const query = (selector) => root.querySelector(selector);
    const svg = query("svg");
    query("[data-title]").textContent = definition.title;
    query("[data-eyebrow]").textContent = definition.eyebrow;

    function algorithms() {
      return typeof definition.algorithms === "function" ? definition.algorithms(state) : definition.algorithms;
    }

    function normalize() {
      for (const control of controls) {
        const choices = options(control.choices, state);
        if (!choices.some((choice) => choice.value === state[control.id])) {
          state[control.id] = choices.some((choice) => choice.value === String(control.value)) ? String(control.value) : choices[0].value;
        }
      }
      const choices = algorithms();
      if (!choices.some((choice) => choice.id === state.algorithm)) {
        state.algorithm = choices.some((choice) => choice.id === definition.defaultAlgorithm) ? definition.defaultAlgorithm : choices[0].id;
      }
    }

    function read_url() {
      const params = new URL(window.location.href).searchParams;
      for (const control of controls) state[control.id] = params.get(`${name}-${control.id}`) ?? String(control.value);
      state.algorithm = params.get(`${name}-algorithm`) || definition.defaultAlgorithm;
      normalize();
      phase = 0;
      playing = !reduced_motion.matches;
    }

    function write_url() {
      const url = new URL(window.location.href);
      for (const [key, value] of Object.entries(state)) url.searchParams.set(`${name}-${key}`, value);
      try { window.history.replaceState(window.history.state, "", url); } catch { /* Local file previews can forbid history updates. */ }
    }

    function position(row_id, index) {
      const row = model.rows.find((row) => row.id === row_id);
      const row_index = model.rows.indexOf(row);
      return [(geometry.width - row.count * 42 + 10) / 2 + index * 42, 76 + row_index * 120];
    }

    function rebuild() {
      normalize();
      model = definition.build(state);
      phase = Math.min(phase, model.phases.length - 1);
      query(".coop-load-options").replaceChildren(...algorithms().map((algorithm) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.algorithm = algorithm.id;
        button.setAttribute("aria-pressed", algorithm.id === state.algorithm);
        const title = document.createElement("strong");
        title.textContent = algorithm.label;
        const tag = document.createElement("span");
        tag.textContent = algorithm.tag || "";
        button.append(title, tag);
        return button;
      }));
      query(".coop-load-settings").replaceChildren(...controls.filter((control) => !control.hidden?.(state)).map((control) => {
        const label = document.createElement("label");
        label.htmlFor = `${prefix}-${control.id}`;
        label.textContent = control.label;
        const select = document.createElement("select");
        select.id = label.htmlFor;
        select.dataset.control = control.id;
        select.append(...options(control.choices, state).map((choice) => new Option(choice.label, choice.value)));
        select.value = state[control.id];
        label.append(select);
        return label;
      }));
      query(".coop-load-phases").replaceChildren(...model.phases.map((item, index) => {
        const button = document.createElement("button");
        button.type = "button";
        button.dataset.phase = index;
        button.textContent = `${index + 1}. ${item.label}`;
        return button;
      }));
      query("[data-detail]").textContent = model.detail;
      query("[data-summary]").textContent = model.summary;
      query("[data-notes]").replaceChildren(...(model.notes || []).map((note) => {
        const paragraph = document.createElement("p");
        paragraph.textContent = note;
        return paragraph;
      }));
      query("[data-caption]").textContent = model.caption || "Teaching model: eight threads and four lanes per displayed warp; physical CUDA warps have 32 lanes. Stages show data dependencies and ownership, not instruction timing. Select a value to inspect it; arrow keys move between values.";
      geometry = { width: Math.max(850, Math.max(...model.rows.map((row) => row.count)) * 42 + 40) };
      svg.replaceChildren();
      svg.setAttribute("viewBox", `0 0 ${geometry.width} ${model.rows.length * 120 + 10}`);
      svg.style.minWidth = `${geometry.width * 0.8}px`;
      svg.append(svg_element("title", { id: `${prefix}-title` }, definition.title));
      svg.append(svg_element("desc", { id: `${prefix}-description` }, model.detail));
      model.rows.forEach((row, index) => {
        const y = 32 + index * 120;
        svg.append(svg_element("text", { x: 20, y, class: "coop-load-axis" }, row.label));
        for (const group of row.groups || []) {
          const [x] = position(row.id, group.start);
          svg.append(svg_element("rect", { x: x - 5, y: y + 20, width: group.count * 42 - 1, height: 70, rx: 7, class: "coop-load-card" }));
          svg.append(svg_element("text", { x, y: y + 35, class: "coop-load-thread-label" }, group.label));
        }
        for (let cell = 0; cell < row.count; ++cell) {
          const [x, cell_y] = position(row.id, cell);
          svg.append(svg_element("rect", { x, y: cell_y, width: 32, height: 32, rx: 4, class: "coop-load-slot", "data-row": row.id, "data-index": cell }));
          if (!row.groups?.length) svg.append(svg_element("text", { x: x + 16, y: cell_y - 8, class: "coop-load-address" }, cell));
        }
      });
      svg.append(svg_element("g", { "data-links": "", "aria-hidden": "true" }));
      token_elements = new Map();
      render_frame();
    }

    function render_frame() {
      const frame = model.phases[phase];
      const live_ids = new Set(frame.tokens.map((token) => String(token.id)));
      if (!live_ids.has(selected)) selected = frame.tokens.length ? String(frame.tokens[0].id) : null;
      for (const [id, element] of token_elements) {
        if (!live_ids.has(id)) {
          element.style.opacity = "0";
          element.setAttribute("tabindex", "-1");
          element.setAttribute("aria-hidden", "true");
          element.style.pointerEvents = "none";
        }
      }
      for (const token of frame.tokens) {
        const id = String(token.id);
        let element = token_elements.get(id);
        if (!element) {
          element = svg_element("g", { class: "coop-load-token coop-explorer-token", role: "button", "data-token-id": id });
          element.append(svg_element("rect", { x: -3, y: -3, width: 38, height: 38, rx: 7, class: "coop-load-token-outline" }));
          element.append(svg_element("rect", { width: 32, height: 32, rx: 4, class: "coop-load-token-body" }));
          element.append(svg_element("text", { x: 16, y: 21, "text-anchor": "middle" }));
          const origin = token.from || token;
          const [x, y] = position(origin.row, origin.index);
          element.style.transform = `translate(${x}px, ${y}px)`;
          svg.append(element);
          token_elements.set(id, element);
          // Establish an entry point before transitioning a newly formed partial.
          element.getBoundingClientRect();
        }
        const [x, y] = position(token.row, token.index);
        element.style.transform = `translate(${x}px, ${y}px)`;
        element.style.opacity = token.muted ? "0.45" : "1";
        element.style.pointerEvents = "auto";
        element.style.setProperty("--coop-token-color", colors[(token.color ?? token.index) % colors.length]);
        element.querySelector("text").textContent = token.label;
        element.dataset.row = token.row;
        element.dataset.index = token.index;
        element.dataset.value = token.value ?? token.label;
        element.dataset.selected = id === selected;
        element.setAttribute("tabindex", id === selected ? "0" : "-1");
        element.setAttribute("aria-pressed", id === selected);
        element.setAttribute("aria-label", token.detail || String(token.label));
        element.removeAttribute("aria-hidden");
      }
      const links = query("[data-links]");
      links.replaceChildren(...(frame.links || []).map((link) => {
        const [x1, y1] = position(link.from.row, link.from.index);
        const [x2, y2] = position(link.to.row, link.to.index);
        return svg_element("path", { d: `M ${x1 + 16} ${y1 + 16} L ${x2 + 16} ${y2 + 16}`, class: "coop-load-path" });
      }));
      const token = frame.tokens.find((token) => String(token.id) === selected);
      query("[data-inspection]").textContent = token ? token.detail || String(token.label) : "No defined values at this stage.";
      for (const button of query(".coop-load-phases").children) {
        if (Number(button.dataset.phase) === phase) button.setAttribute("aria-current", "step");
        else button.removeAttribute("aria-current");
      }
      query("[data-status]").setAttribute("aria-live", playing ? "off" : "polite");
      query("[data-status]").textContent = `Step ${phase + 1} of ${model.phases.length}. ${frame.description}`;
      query("[data-action=play]").textContent = playing ? "Pause" : "Play";
      query("[data-action=play]").setAttribute("aria-label", playing ? "Pause animation" : "Play animation");
      root.dataset.phase = phase;
      root.dataset.algorithm = state.algorithm;
    }

    function sync_timer() {
      window.clearInterval(timer);
      if (playing && visible && !document.hidden) {
        timer = window.setInterval(() => { phase = (phase + 1) % model.phases.length; render_frame(); }, 1600);
      }
    }

    root.addEventListener("click", (event) => {
      const token = event.target.closest("[data-token-id]");
      if (token) { selected = token.dataset.tokenId; render_frame(); return; }
      const button = event.target.closest("button");
      if (!button) return;
      if (button.dataset.algorithm) {
        const algorithm = button.dataset.algorithm;
        state.algorithm = algorithm;
        phase = 0;
        rebuild();
        query(`[data-algorithm="${algorithm}"]`)?.focus();
        write_url();
      } else if (button.dataset.phase !== undefined) {
        phase = Number(button.dataset.phase);
        playing = false;
      } else if (button.dataset.action === "play") playing = !playing;
      else {
        playing = false;
        phase = button.dataset.action === "reset" ? 0 : (phase + (button.dataset.action === "next" ? 1 : model.phases.length - 1)) % model.phases.length;
      }
      render_frame();
      sync_timer();
    });
    root.addEventListener("change", (event) => {
      const id = event.target.dataset.control;
      if (!id) return;
      state[id] = event.target.value;
      phase = 0;
      rebuild();
      query(`[data-control="${id}"]`)?.focus();
      write_url();
      sync_timer();
    });
    root.addEventListener("keydown", (event) => {
      const element = event.target.closest("[data-token-id]");
      if (!element) return;
      const ids = model.phases[phase].tokens.map((token) => String(token.id));
      let index = ids.indexOf(element.dataset.tokenId);
      if (["ArrowRight", "ArrowDown"].includes(event.key)) index = (index + 1) % ids.length;
      else if (["ArrowLeft", "ArrowUp"].includes(event.key)) index = (index + ids.length - 1) % ids.length;
      else if (event.key === "Home") index = 0;
      else if (event.key === "End") index = ids.length - 1;
      else if (!["Enter", " "].includes(event.key)) return;
      event.preventDefault();
      selected = ids[index];
      render_frame();
      token_elements.get(selected).focus();
    });

    read_url();
    rebuild();
    host.replaceChildren(root);
    host.closest(".coop-visualization")?.classList.add("coop-load-ready");
    window.addEventListener("popstate", () => { read_url(); rebuild(); sync_timer(); });
    document.addEventListener("visibilitychange", sync_timer);
    reduced_motion.addEventListener("change", () => {
      if (reduced_motion.matches) { playing = false; render_frame(); sync_timer(); }
    });
    if (typeof IntersectionObserver === "function") {
      const observer = new IntersectionObserver(([entry]) => { visible = entry.isIntersecting; sync_timer(); });
      observer.observe(host);
    }
    sync_timer();
  }

  function initialize(name) {
    document.querySelectorAll(`[data-coop-explorer="${name}"]`).forEach((host, index) => {
      if (host.dataset.mounted) return;
      try {
        mount(host, name, definitions.get(name), index);
        host.dataset.mounted = "true";
      } catch (error) {
        console.warn(`The ${name} explorer could not be initialized.`, error);
      }
    });
  }

  window.CoopExplorer = {
    register(name, definition) {
      definitions.set(name, definition);
      if (document.readyState === "loading") document.addEventListener("DOMContentLoaded", () => initialize(name));
      else initialize(name);
    },
  };
})();
