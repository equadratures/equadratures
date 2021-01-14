import os
import sys
project = u' '
copyright = u'2016-2020 by Effective Quadratures'
author = u'Effective Quadratures'
# -- Bryn: Main Vuepress website location ------------------------------------
land_page = "https://www.effective-quadratures.org/"
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
]
templates_path = ['_docstemplates']
source_suffix = '.txt'
master_doc = '_documentation/index'
html_title = 'equadratures'
html_theme = 'pydata_sphinx_theme'
html_logo = 'logo_new.png'
html_favicon = 'eq-logo-favicon.png'
def setup(app):
    app.add_stylesheet('styles.css')
language = None
exclude_patterns = [u'_docsbuild', 'Thumbs.db', '.DS_Store']
html_static_path = ['_static']
htmlhelp_basename = 'EffectiveQuadraturesdoc'

latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '12pt',
}

latex_documents = [
    (master_doc, 'EffectiveQuadratures.tex', u'Effective Quadratures Documentation',
     u'Pranay Seshadri, Nicholas Wong, Henry Yuchi, Irene Virdis', 'manual'),
]

man_pages = [
    (master_doc, 'effectivequadratures', u'Effective Quadratures Documentation',
     [author], 1)
]

texinfo_documents = [
    (master_doc, 'EffectiveQuadratures', u' ',
     author, 'EffectiveQuadratures', 'One line description of project.',
     'Miscellaneous'),
]

intersphinx_mapping = {'https://docs.python.org/': None}

html_theme_options = {
    'nosidebar': True,
    'sidebar_span': 0,
    "github_url": "https://github.com/pandas-dev/pydata-sphinx-theme",
    "twitter_url": "https://twitter.com/EQuadratures",
    'globaltoc_includehidden': "true",
    "search_bar_position": "navbar",
     "show_prev_next": False,
     "search_bar_text": "Search equadratures..."
}

html_sidebars = {
  "tutorials": [],
  "index": [],
  "modules":[]
}

html_context = {
    "github_user": "pandas-dev",
    "github_repo": "pydata-sphinx-theme",
    "github_version": "master",
    "doc_path": "docs",
}
