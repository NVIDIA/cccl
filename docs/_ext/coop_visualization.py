# Copyright (c) 2026, NVIDIA CORPORATION & AFFILIATES. ALL RIGHTS RESERVED.
#
# SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

"""Embed a cooperative primitive explorer with an RST fallback."""

from docutils import nodes
from sphinx.util.docutils import SphinxDirective


class CoopVisualization(SphinxDirective):
    required_arguments = 1
    has_content = True

    def run(self):
        if self.arguments[0] != "load":
            raise self.error("The only supported cooperative visualization is 'load'.")
        self.assert_has_content()
        container = nodes.container(classes=["coop-visualization"])
        container += nodes.raw("", "<div data-coop-load></div>", format="html")
        fallback = nodes.container(classes=["coop-visualization-fallback"])
        self.state.nested_parse(self.content, self.content_offset, fallback)
        container += fallback
        return [container]


def add_visualization_assets(app, pagename, templatename, context, doctree):
    if doctree is not None and any(
        "coop-visualization" in node["classes"]
        for node in doctree.findall(nodes.container)
    ):
        # Assets are relative to Sphinx's output root, including dirhtml builds.
        app.add_css_file("coop-load.css")
        app.add_js_file("coop-load.js", defer="defer")


def setup(app):
    app.add_directive("coop-visualization", CoopVisualization)
    app.connect("html-page-context", add_visualization_assets)
    return {"version": "1", "parallel_read_safe": True, "parallel_write_safe": True}
