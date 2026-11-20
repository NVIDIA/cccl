# CCCL Documentation Configuration File
# Generated to replace repo-docs with direct Sphinx usage

import os
import sys
from datetime import datetime

# Add extension directory to path, and docs/ for registry.py
sys.path.insert(0, os.path.abspath("_ext"))
sys.path.insert(0, os.path.abspath(".."))

import registry  # noqa: E402

# -- Project information -----------------------------------------------------

project = "CUDA Core Compute Libraries"
copyright = f"{datetime.now().year}, NVIDIA Corporation"
author = "NVIDIA Corporation"

# Version information
_env_version = os.environ.get("SPHINX_CCCL_VER")
if _env_version:
    release = _env_version
else:
    try:
        with open("../VERSION.md", "r", encoding="utf-8") as f:
            release = f.read().strip()
    except Exception:
        release = "unstable"

version = release

# -- General configuration ---------------------------------------------------

extensions = [
    "sphinx.ext.extlinks",
    "sphinx.ext.mathjax",
    "sphinx.ext.graphviz",
    "sphinx.ext.doctest",
    "myst_parser",  # MyST parser for markdown support
    "breathe",  # For Doxygen integration - has built-in embed:rst support
    # "exhale",  # Disabled - causing build timeouts, API docs handled by breathe
    "sphinx_design",  # For dropdown, card, and other directives
    "sphinx_copybutton",
    "nbsphinx",
    # "rst_processor",  # Disabled - breathe handles embed:rst natively
    "auto_api_generator",  # Automatically generate API reference pages from Doxygen XML
]

# Breathe configuration for Doxygen integration
breathe_projects = {
    "cub": "_build/doxygen/cub/xml",
    "thrust": "_build/doxygen/thrust/xml",
    "libcudacxx": "_build/doxygen/libcudacxx/xml",
    "cudax": "_build/doxygen/cudax/xml",
}

breathe_default_project = "cub"
breathe_default_members = ("members", "undoc-members")
breathe_show_enumvalue_initializer = True
breathe_domain_by_extension = {"cuh": "cpp", "h": "cpp", "hpp": "cpp"}

# Configure cpp domain to handle cub namespace
cpp_index_common_prefix = ["cub::"]
# The NVIDIA theme changes toc_object_entries_show_parents from "domain" to
# "hide" and maximum_signature_line_length from None to 70 before the environment
# is pickled. Set the final values explicitly to avoid invalidating the cache.
toc_object_entries_show_parents = "hide"
maximum_signature_line_length = 70

# Preprocessor definitions for Breathe to handle CCCL macros
cpp_id_attributes = [
    "__device__",
    "__host__",
    "__global__",
    "__forceinline__",
    "_CCCL_HOST_DEVICE",
    "_CCCL_DEVICE",
    "_CCCL_HOST",
    "_CCCL_FORCEINLINE",
    "_CCCL_API",
    "_CCCL_HOST_API",
    "_CCCL_DEVICE_API",
    "_CCCL_NODEBUG_API",
    "_CCCL_NODEBUG_HOST_API",
    "_CCCL_NODEBUG_DEVICE_API",
    "_CCCL_TRIVIAL_API",
    "_CCCL_TRIVIAL_HOST_API",
    "_CCCL_TRIVIAL_DEVICE_API",
]
cpp_paren_attributes = ["__declspec", "__align__"]

# Add support for .rst and .md files
source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}

templates_path = ["../common/_templates"]

# Exclude patterns
exclude_patterns = [
    "_build",
    "_repo",
    "tools",
    "Thumbs.db",
    ".DS_Store",
    "**/.pytest_cache",
    "**/__pycache__",
    "*.pyc",
    "*.pyo",
]

# -- Options for HTML output -------------------------------------------------

html_theme = "nvidia_sphinx_theme"

html_logo = "../common/_static/nvidia-logo.png"

# Each version is published at <site>/cpp/<version>/.
html_baseurl = f"{registry.site_url()}/cpp/{release}/"

html_theme_options = {
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/NVIDIA/cccl",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "navigation_depth": 4,
    "show_toc_level": 2,
    "navbar_start": ["navbar-logo"],
    "navbar_end": ["theme-switcher", "navbar-icon-links"],
    "footer_start": ["copyright"],
    "footer_end": ["sphinx-version"],
    "sidebar_includehidden": True,
    "collapse_navigation": False,
    "switcher": {
        "json_url": f"{registry.site_url()}/cpp/nv-versions.json",
        "version_match": release,
    },
}

# Tell readers that unstable is not a release.
if release == "unstable":
    html_theme_options["announcement"] = (
        "This is the <strong>development</strong> documentation, built from "
        "the latest commit on <code>main</code>."
    )

html_static_path = ["_static", "../common/_static"]

# Images directory
if os.path.exists("img"):
    html_static_path.append("img")

html_js_files = ["deduplicate_toc.js"]

html_title = "CUDA Core Compute Libraries"

# -- Options for extensions --------------------------------------------------

# MyST parser configuration
myst_enable_extensions = [
    "colon_fence",
    "deflist",
    "html_image",
]

# External links configuration
extlinks = {
    "github": ("https://github.com/NVIDIA/cccl/blob/main/%s", "%s"),
}


# Exhale not used - API documentation is handled directly through breathe directives

# Suppress specific warning categories that arise from breathe (Doxygen-to-Sphinx
# bridge) limitations.  These cannot be fixed in our source headers or RST files.
#
# See also _BREATHE_SKIP_SYMBOLS in _ext/auto_api_generator.py for symbols that
# are excluded from page generation entirely due to unparsable declarations.
suppress_warnings = [
    # Breathe walks each Doxygen XML file independently.  When a symbol appears
    # in both a namespace XML and a class/group XML (which is normal for Doxygen),
    # breathe emits the C++ declaration twice, triggering a duplicate-declaration
    # warning.  There is no way to control this from our side without patching
    # breathe's XML traversal.
    "duplicate_declaration.cpp",
    # When breathe expands doxygenfunction/doxygenvariable directives, it writes
    # the resolved C++ signature into RST.  Signatures containing default argument
    # values (e.g. ``= {}``) or complex SFINAE expressions produce RST that the
    # docutils parser cannot handle (mismatched inline-literal markers, unexpected
    # braces, etc.).  The source C++ is valid; the issue is that what breathe
    # emits as RST is not valid RST.
    "docutils",
]

copybutton_prompt_text = ">>> |$ |# "


def setup(app):
    if os.path.exists("_static/custom.css"):
        app.add_css_file("custom.css")
