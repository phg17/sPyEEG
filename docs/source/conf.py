# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = 'SPyEEG'
copyright = '2025, Pierre Guilleminot'
author = 'Pierre Guilleminot'
release = '0.1'

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = [
    'sphinx.ext.duration',
    'sphinx.ext.doctest',
    'sphinx.ext.autodoc',
    'sphinx.ext.autosummary',
    'sphinx.ext.napoleon',
    'sphinx.ext.viewcode',
    'nbsphinx',
    'sphinx.ext.todo',
    'sphinx.ext.intersphinx',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.githubpages',
    'numpydoc',
    'sphinxcontrib.bibtex',
    'sphinx_design',
    'sphinx_contributors',
    'sphinx_rtd_theme',
    #'sphinx_gallery.gen_gallery',
]

bibtex_bibfiles = ['refs.bib']
templates_path = ['_templates']


# Optional: Exclude unwanted files from processing
exclude_patterns = ['**.ipynb_checkpoints', '_build', 'Thumbs.db', '.DS_Store']  # Ignore Jupyter checkpoints



# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = 'alabaster'
html_static_path = ['_static']


# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here.
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

#Autodoc Options
autodoc_default_options = {
    'members': True,
    'undoc-members': True,
    'private-members': True,
    'special-members': '__init__',
    'inherited-members': True,
    'show-inheritance': True,
}

autodoc_typehints = "description"
autosummary_generate = True

# Optional: Configure nbsphinx
nbsphinx_allow_errors = True  # Continue building even if notebooks have errors

# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "sphinx_rtd_theme"
