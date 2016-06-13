from effective_quadratures.PolyParams import *
#from effective_quadrautres.PolyParentFile import *
#from effective_quadrautres.MatrixRoutines import *
# Set default logging handler to avoid "No handler found" warnings.
import logging
try:  # Python 2.7+
    from logging import NullHandler
except ImportError:
    class NullHandler(logging.Handler):
        def emit(self, record):
            pass

# remove stderr logger
logging.getLogger(__name__).addHandler(NullHandler())
