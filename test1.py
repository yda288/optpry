import unittest
from models import *
import pdb
from functools import wraps
import traceback
import sys


class BlackScholesPricerTest(unittest.TestCase):
    def setUp(self):
        # Option parameters
        self.myCall = EuropeanOption(opt_type=OptionType.call, 
                            asset_price=100., 
                            strike=100., 
                            rho=.01, 
                            q=0., 
                            maturity_years=0.5, 
                            vol=0.35, 
                            pricing_model=BlackScholes())
        self.myCall.price()

    def test_EuropeanCall(self):
        self.assertAlmostEqual(self.myCall.delta, 0.557228735645, delta=1e-12)

class MonteCarloPricerTest(unittest.TestCase):
    def setUp(self):
        self.myCall = EuropeanOption(opt_type=OptionType.call,
                                asset_price=100, 
                                strike=95, 
                                rho=0.1, 
                                q=0.,
                                maturity_years=0.25, 
                                vol=0.5,
                                pricing_model=MonteCarlo())
        self.myCall.price()

    def test_EuropeanCall(self):
        self.assertAlmostEquals(self.myCall.price()[0], 13.69, delta=1e-2)

class BoostMonteCarloPricer(unittest.TestCase):
    def setUp(self):
        self.myCall = EuropeanOption(opt_type=OptionType.call,
                                asset_price=100, 
                                strike=95, 
                                rho=0.1, 
                                q=0.,
                                maturity_years=0.25, 
                                vol=0.5,
                                pricing_model=MonteCarlo(boost=True))

        self.myPut = EuropeanOption(opt_type=OptionType.put,
                                asset_price=100, 
                                strike=95, 
                                rho=0.1, 
                                q=0.,
                                maturity_years=0.25, 
                                vol=0.5,
                                pricing_model=MonteCarlo())
        self.myCall.price()
        self.myPut.price()

    def test_EuropeanCall_boost(self):
        self.assertAlmostEquals(self.myCall.price()[0], 13.69, delta=1e-2)

    def test_EuropeanPut_boost(self):
        self.assertAlmostEquals(self.myPut.price()[0], 6.35, delta=1e-2)



if __name__ == '__main__':
    unittest.main()
