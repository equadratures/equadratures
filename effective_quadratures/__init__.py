'Initialization file'
import effective_quadratures.parameter, effective_quadratures.polynomial, effective_quadratures.indexset, effective_quadratures.effectivequads, effective_quadratures.computestats, effective_quadratures.integrals, effective_quadratures.plotting, effective_quadratures.analyticaldistributions, effective_quadratures.utils, effective_quadratures.qr
#from effective_quadratures.base import *

# Set default logging handler to avoid "No handler found" warnings.
#import logging
#try:  # Python 2.7+
#    from logging import NullHandler
#except ImportError:
#    class NullHandler(logging.Handler):
#        def emit(self, record):
#            pass

# remove stderr logger
#logging.getLogger(__name__).addHandler(NullHandler())