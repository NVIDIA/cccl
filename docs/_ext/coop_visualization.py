# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Embed a cooperative primitive explorer with an RST fallback."""

from docutils import nodes
from sphinx.util.docutils import SphinxDirective

_EXPLORERS = {
    "load": "coop-load.js",
    "store": "coop-movement.js",
    "exchange": "coop-movement.js",
    "shuffle": "coop-shuffle.js",
    "reduce": "coop-collectives.js",
    "scan": "coop-collectives.js",
    "merge-sort": "coop-merge-sort.js",
    "radix": "coop-radix.js",
    "topk": "coop-topk.js",
    "adjacent-difference": "coop-neighbors.js",
    "discontinuity": "coop-neighbors.js",
    "histogram": "coop-histogram.js",
    "run-length-decode": "coop-run-length-decode.js",
    "reduce-batched": "coop-reduce-batched.js",
}

_VISUALIZATION_TITLES = {
    "merge-sort": "Merge Sort",
    "radix": "Radix Rank/Sort",
    "topk": "TopK",
    "adjacent-difference": "Adjacent Difference",
    "discontinuity": "Discontinuity",
    "histogram": "Histogram",
    "run-length-decode": "Run Length Decode",
    "reduce-batched": "Batched Warp Reduction",
}

_API_VISUALIZATIONS = (
    dict.fromkeys(
        ("scan", "exclusive_scan", "inclusive_scan", "exclusive_sum", "inclusive_sum"),
        "scan",
    )
    | {name: name for name in _EXPLORERS}
    | {
        "sum": "reduce",
        "adjacent_difference": "adjacent-difference",
        "run_length_decode": "run-length-decode",
        "run_length_decode_into": "run-length-decode",
        "reduce_batched": "reduce-batched",
    }
    | dict.fromkeys(("merge_sort_keys", "merge_sort_pairs"), "merge-sort")
    | dict.fromkeys(("radix_sort_keys", "radix_sort_pairs", "radix_rank"), "radix")
    | dict.fromkeys(
        ("topk_min_keys", "topk_max_keys", "topk_min_pairs", "topk_max_pairs"), "topk"
    )
)


def add_api_visualization_link(app, what, name, obj, options, lines):
    module, _, function = name.rpartition(".")
    if what != "function" or module not in {"cuda.coop", "cuda.coop.numba_mlir"}:
        return
    visualization = _API_VISUALIZATIONS.get(function)
    if visualization is not None:
        title = _VISUALIZATION_TITLES.get(visualization, visualization.title())
        lines.extend(
            [
                "",
                ".. seealso::",
                "",
                f"   :doc:`{title} visualization "
                f"</python/coop/visualizations/{visualization}>`",
                "",
            ]
        )


class CoopVisualization(SphinxDirective):
    required_arguments = 1
    has_content = True

    def run(self):
        name = self.arguments[0]
        if name not in _EXPLORERS:
            raise self.error(f"Unknown cooperative visualization: {name!r}.")
        self.assert_has_content()
        container = nodes.container(classes=["coop-visualization"])
        container["coop_explorer"] = name
        attribute = (
            "data-coop-load" if name == "load" else f'data-coop-explorer="{name}"'
        )
        container += nodes.raw("", f"<div {attribute}></div>", format="html")
        fallback = nodes.container(classes=["coop-visualization-fallback"])
        self.state.nested_parse(self.content, self.content_offset, fallback)
        container += fallback
        return [container]


def add_visualization_assets(app, pagename, templatename, context, doctree):
    if doctree is None:
        return
    explorers = {
        node["coop_explorer"]
        for node in doctree.findall(nodes.container)
        if "coop_explorer" in node
    }
    if explorers:
        # Assets are relative to Sphinx's output root, including dirhtml builds.
        app.add_css_file("coop-load.css")
        if explorers - {"load"}:
            app.add_js_file("coop-explorer.js", defer="defer")
        for script in sorted({_EXPLORERS[name] for name in explorers}):
            app.add_js_file(script, defer="defer")


def setup(app):
    app.setup_extension("sphinx.ext.autodoc")
    app.add_directive("coop-visualization", CoopVisualization)
    app.connect("autodoc-process-docstring", add_api_visualization_link)
    app.connect("html-page-context", add_visualization_assets)
    return {
        "version": "7",
        "env_version": 6,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
