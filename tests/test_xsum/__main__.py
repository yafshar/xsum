import unittest

from .. import prepare_load_tests
from . import __path__

load_tests = prepare_load_tests(__path__)

unittest.main()
