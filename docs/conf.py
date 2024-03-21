# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "overlappogram"
copyright = "2024, J. Marcus Hughes, Dyana Beabout"
author = "J. Marcus Hughes, Dyana Beabout"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = ["autoapi.extension", "sphinx.ext.autodoc", "sphinx.ext.napoleon"]

templates_path = ["_templates"]
exclude_patterns = ["_build", "Thumbs.db", ".DS_Store"]


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "pydata_sphinx_theme"
html_show_sourcelink = False
html_static_path = ["_static"]
html_theme_options = {
    "use_edit_page_button": True,
    "icon_links": [
        {
            "name": "GitHub",
            "url": "https://github.com/jmbhughes/overlappogram",
            "icon": "fa-brands fa-github",
            "type": "fontawesome",
        }
    ],
    "show_nav_level": 1,
    "show_toc_level": 3,
}
html_context = {
    # "github_url": "https://github.com", # or your GitHub Enterprise site
    "github_user": "jmbhughes",
    "github_repo": "overlappogram",
    "github_version": "main",
    "doc_path": "docs/",
}


autoapi_dirs = ["../overlappogram"]
