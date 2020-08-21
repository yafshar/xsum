import unittest


def prepare_load_tests(start_dir):
    """Prepare loading tests."""
    suite = unittest.TestLoader().discover(start_dir[0])

    def load_tests(_loader, _standard_tests, _pattern):
        """Load tests."""
        return suite

    return load_tests
