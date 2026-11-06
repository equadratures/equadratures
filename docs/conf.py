import os
import sys

# Misc Sphinx settings
#########################
project = u' '
copyright = u'2016-2021 by equadratures'
author = u'equadratures'
land_page = "equadratures.org"
outdir = sys.argv[-1]   #Bryn: Assuming last system argument is output directory
version = u'9.1'
release = u'v9.1.0'
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
    'sphinx.ext.napoleon',
    'sphinxcontrib.mermaid',
    'sphinx_panels',
    'sphinx_copybutton',
    'nbsphinx',
    'sphinx_gallery.load_style',
    'sphinxcontrib.bibtex',
] #TODO - prune unused extensions
master_doc = 'index'
html_title = 'Documentation | equadratures'
html_theme = 'pydata_sphinx_theme'
html_logo = 'equadraturesLockupHorizontalMain.png'
html_favicon = 'eq-logo-favicon.png'
language = None
exclude_patterns = [u'_docsbuild', 'Thumbs.db', '.DS_Store']
source_suffix = '.txt'
htmlhelp_basename = 'equadraturesdoc'
latex_elements = {
    'papersize': 'a4paper',
    'pointsize': '12pt',
}

# Manual and text info outputs
#########################
man_pages = [
    (master_doc, 'effectivequadratures', u'equadratures documentation',
     [author], 1)
]
texinfo_documents = [
    (master_doc, 'equadratures', u' ',
     author, 'equadratures', 'equadratures is an open-source python code.',
     'Miscellaneous'),
]

# Templates, static files, and css (bug with css being copied to _static/css, this must be done manually for now)
#########################
templates_path = ['_templates']
html_static_path = ['_static']
html_css_files = [
        'css/styles.css',
        'css/footer.css',
        'css/fonts.css'
]

# Intersphinx mapping
intersphinx_mapping = {
        'https://docs.python.org/': None,
        'numpy': ('https://numpy.org/doc/stable/', None),
        'matplotlib': ('https://matplotlib.org/stable/', None),
        'scipy': ('https://docs.scipy.org/doc/scipy/reference/',None),
        'sklearn': ('http://scikit-learn.org/stable',
            (None, './_intersphinx/sklearn-objects.inv'))
        }


# Sphinx panels
#########################
panels_add_bootstrap_css = False # sphinx-panels shouldn't add bootstrap css since the pydata-sphinx-theme already loads it

# nbsphinx setting
#########################
nbsphinx_execute_arguments = [
    "--InlineBackend.figure_formats={'svg', 'pdf'}",
    "--InlineBackend.rc=figure.dpi=96",
]
nbsphinx_input_prompt = 'In [%s]:'
nbsphinx_output_prompt = 'Out[%s]:'
nbsphinx_thumbnails = {
    'theory/notebooks/1_Introduction': '_static/gallery/whale.png',
    'theory/notebooks/2_Parameter': '_static/gallery/parameter.png',
    'theory/notebooks/3_Basis': '_static/gallery/total.png',
    'theory/notebooks/4_Polynomial': '_static/gallery/polynomials.png',
    'tutorials/tutorials/3_Model_Fitting': '_static/gallery/tensor-grid.png',
    'tutorials/tutorials/Computing_Moments_for_CFD': '_static/gallery/vki_turbine.jpeg',
    'tutorials/tutorials/Dimension_Reduction_Turbomachinery': '_static/gallery/probe.jpeg',
    'tutorials/tutorials/Machine_Learning_Airfoil': '_static/gallery/airfoil_noise.png',
}


# sphinxcontrib.bibtex setting
##############################
bibtex_bibfiles = ['references.bib']

# pydata theme options
#########################
html_theme_options = {
    "show_prev_next": False,
    "search_bar_text": "Search documentation",
}
