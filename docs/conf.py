import os
import sys
sys.path.insert(0, os.path.abspath('..'))

project = 'smoothing_spline'
copyright = '2024, Jonathan Taylor'
author = 'Jonathan Taylor'

extensions = [
    'myst_parser',
    'sphinx.ext.autodoc',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx',
    'sphinx.ext.mathjax',
]

templates_path = ['_templates']
exclude_patterns = ['_build', 'Thumbs.db', '.DS_Store']

html_theme = 'alabaster'
html_static_path = ['_static']
