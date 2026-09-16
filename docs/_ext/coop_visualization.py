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
}


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
    app.add_directive("coop-visualization", CoopVisualization)
    app.connect("html-page-context", add_visualization_assets)
    return {
        "version": "2",
        "env_version": 1,
        "parallel_read_safe": True,
        "parallel_write_safe": True,
    }
