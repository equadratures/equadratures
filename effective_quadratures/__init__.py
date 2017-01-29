'Initialization file'
from base import *

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