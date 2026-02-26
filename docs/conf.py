# Configuration file for the Sphinx documentation builder.

import os
import sys

sys.path.insert(0, os.path.abspath(".."))

project = "OmniNav"
author = "Royalvice"
release = "0.1.0"

extensions = [
    "myst_parser",
    "sphinx.ext.autodoc",
    "sphinx.ext.napoleon",
    "sphinx.ext.viewcode",
    "sphinx_copybutton",
    "sphinx_autodoc_typehints",
    "sphinx_design",
]

myst_enable_extensions = ["colon_fence", "deflist", "html_image"]
myst_heading_anchors = 4

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]

source_suffix = {
    ".rst": "restructuredtext",
    ".md": "markdown",
}
master_doc = "index"

html_theme = "pydata_sphinx_theme"
html_logo = "_static/logo.png"
html_favicon = "_static/logo.png"
html_static_path = ["_static"]

html_theme_options = {
    "show_nav_level": 2,
    "navigation_with_keys": False,
    "logo": {
        "text": "OmniNav",
        "image_light": "_static/logo.png",
        "image_dark": "_static/logo.png",
    },
    "navbar_center": ["navbar-nav"],
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/Royalvice/OmniNav",
            "icon": "fa-brands fa-github",
        },
    ],
}

html_context = {
    "display_github": True,
    "github_user": "Royalvice",
    "github_repo": "OmniNav",
    "github_version": "main",
    "conf_py_path": "/docs/",
}

html_css_files = ["css/custom.css"]
html_js_files = ["js/language-toggle.js"]

autodoc_member_order = "bysource"
autodoc_typehints = "description"
autodoc_mock_imports = [
    "genesis",
    "torch",
    "cv2",
    "rclpy",
    "sensor_msgs",
    "std_msgs",
    "geometry_msgs",
    "nav_msgs",
]

language = "en"
