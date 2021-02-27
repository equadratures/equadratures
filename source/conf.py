import os
import sys
project = u' '
copyright = u'2016-2021 by equadratures.org'
author = u'equadratures'
# -- Bryn: Main Vuepress website location ------------------------------------
land_page = "equadratures.org"
outdir = sys.argv[-1]   #Bryn: Assuming last system argument is output directory
version = u'9.0'
release = u'v9.0.1'
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'sphinx.ext.ifconfig',
    'sphinx.ext.viewcode',
    'sphinx.ext.githubpages',
    'sphinxcontrib.slide',
    'sphinxcontrib.youtube'
]
templates_path = ['_docstemplates']
source_suffix = '.txt'
master_doc = 'index'
html_title = 'equadratures'
html_theme = 'pydata_sphinx_theme'
html_logo = 'logo_new.png'
html_favicon = 'eq-logo-favicon.png'
def setup(app):
    app.add_css_file('styles.css')
language = None
exclude_patterns = [u'_docsbuild', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']
htmlhelp_basename = 'EffectiveQuadraturesdoc'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '12pt',
}
man_pages = [
    (master_doc, 'effectivequadratures', u'Effective Quadratures Documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'equadratures', u' ',
     author, 'equadratures', 'equadratures is an open-source python code developed by Effective Quadratures. It is tailored for tackling problems in uncertainty quantification, surrogate-based optimisation, numerical integration, and data-driven dimension reduction. ',
     'Miscellaneous'),
]
intersphinx_mapping = {'https://docs.python.org/': None}
html_theme_options = {
    "external_links": [
      {"url": "https://discourse.equadratures.org/", "name": "Discourse"}
     ],
    'nosidebar': True,
    "github_url": "https://github.com/Effective-Quadratures/equadratures",
    "twitter_url": "https://twitter.com/equadratures",
    'globaltoc_includehidden': "true",
    "search_bar_position": "navbar",
     "show_prev_next": False,
     "search_bar_text": "Search equadratures..."
}
html_sidebars = {
  " /_documentation/tutorials": [],
  "tutorials": [],
  "index": [],
  " /_documentation/modules":[],
  "modules":[]
}
