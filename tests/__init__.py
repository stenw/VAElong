"""
Test runner for all tests.
"""

import unittest
import sys

# Import test modules
from .test_model import TestLongitudinalVAE, TestVAELoss
from .test_data import TestLongitudinalDataset, TestSyntheticDataGeneration
from .test_trainer import TestVAETrainer


def suite():
    """Create test suite."""
    test_suite = unittest.TestSuite()
    
    # Add model tests
    test_suite.addTest(unittest.makeSuite(TestLongitudinalVAE))
    test_suite.addTest(unittest.makeSuite(TestVAELoss))
    
    # Add data tests
    test_suite.addTest(unittest.makeSuite(TestLongitudinalDataset))
    test_suite.addTest(unittest.makeSuite(TestSyntheticDataGeneration))
    
    # Add trainer tests
    test_suite.addTest(unittest.makeSuite(TestVAETrainer))
    
    return test_suite


if __name__ == '__main__':
    runner = unittest.TextTestRunner(verbosity=2)
    result = runner.run(suite())
    
    # Exit with error code if tests failed
    sys.exit(not result.wasSuccessful())
