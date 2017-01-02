from abc import ABCMeta, abstractmethod
from enum import Enum
import numpy as np
from scipy.stats import norm
import BoostTest

class OptionType(Enum):
    call = 1
    put = -1

class BaseOption(object, metaclass=ABCMeta):
    """Base Option interface"""
    def __init__(self, opt_type, asset_price, strike, rho, q, maturity_years, vol, pricing_model):
        self._type = opt_type                    # 1 for a Call, - 1 for a put
        self._S0 = asset_price                    # Underlying asset price
        self._K = strike                         # Option strike K
        self._r = rho                            # Continuous compounded risk fee rate
        self._div = q                              # Continuously compounded Dividend yield
        self._T = maturity_years         # Compute time to expiry (fraction of year)
        self._sigma = vol            # diffusion volatility factor
        self.model = pricing_model

    def price(self):
        if not self.model._value:
            self.model.price_option(self)
        return self.model._value

    @property
    def delta(self):
        return self.model._delta

    @property
    def gamma(self):
        return self.model._gamma

    @property
    def vega(self):
        return self.model._vega

    @property
    def rho(self):
        return self.model._rho

    @property
    def theta(self):
        return self.model._theta

    @property
    def type(self):
        return self._type


class EuropeanOption(BaseOption):
    """European Option implementation"""
    def __init__(self, opt_type, asset_price, strike, rho, q, maturity_years, vol, pricing_model):
        super().__init__(opt_type, asset_price, strike, rho, q, maturity_years, vol, pricing_model)


class PricingStrategy(object, metaclass=ABCMeta):
    """Base pricing strategy interface"""
    def __init__(self, boost=False):
        self._delta=None
        self._gamma=None
        self._rho=None
        self._vega=None
        self._theta=None
        self._value=None
        self.boost=boost

    @abstractmethod
    def price_option(self):
        raise NotImplementedError

    def calculate(self, option):
        self.delta(option)
        self.gamma(option)
        self.rho(option)
        self.vega(option)
        self.theta(option)
        self.value(option)


class BlackScholes(PricingStrategy):
    """BS closed form interface"""
    def __init__(self):
        super().__init__()

    def price_option(self, option: BaseOption):
        d1 = ((np.log(option._S0 / option._K) + 
              (option._r - option._div + 0.5 * (option._sigma ** 2)) * option._T) / 
              float( option._sigma * np.sqrt(option._T)))
        d2 = float(d1 - option._sigma * np.sqrt(option._T))
        print(d1, d2)
        self._Nd1 = norm.cdf(d1, 0, 1)
        self._Nnd1 = norm.cdf(-d1, 0, 1)
        self._Nd2 = norm.cdf(d2, 0, 1)
        self._Nnd2 = norm.cdf(-d2, 0, 1)
        self._pNd1 = norm.pdf(d1, 0, 1)
        self.calculate(option)

    def value(self, option):
        if  option._type == OptionType.call:
            self._value = (option._S0 * np.exp(-option._div * option._T) * self._Nd1 -
                     option._K * np.exp(-option._r * option._T) * self._Nd2)
        else:
           self._value = (option._K * np.exp(-option._r * option._T) * self._Nnd2 -
                     option._S0 * np.exp(-option.div * option._T) * self._Nnd1)

    def delta(self, option):
        self._delta = np.exp(-option._div * option._T) * (self._Nd1 
                                                         if option._type == OptionType.call 
                                                         else 
                                                         self.Nd1 - 1 )
    def gamma(self, option):
        self._gamma = (self._pNd1 * np.exp(-option._div * option._T) / 
                 float(option._S0 * option._sigma * np.sqrt(option._T)))

    def rho(self, option):
        self._rho = (option._K 
                     if option._type == OptionType.call
                    else 
                    -option._K) * option._T * np.exp(-option._r * option._T) * self._Nd2

    def vega(self, option):
         self._vega = option._S0 * self._pNd1 * np.exp(-option._div * option._T) * np.sqrt(option._T)
        
    def theta(self, option):
        self._theta = (
            (
            (-option._S0
             if option._type == OptionType.call
             else
             option._S0) * self._pNd1 * option._sigma * np.exp(- option._div * option._T)
             ) /
                     float(2. * np.sqrt(option._T)) + 
                     option._div * option._S0 * self._Nnd1 * np.exp(- option._div * option._T) - 
                     option._r * option._K * np.exp(- option._r * option._T) * self._Nnd2
                     )


class MonteCarlo(PricingStrategy):
     def __init__(self, boost=False):
         super().__init__(boost)
         
     def price_option(self, option: BaseOption):
         if self.boost:
            self.value_boost(option)
         else:
            self.value(option)
       
     def value(self, option):
         trials = [self.montecarlo(option) for _ in range(1000)]
         self._value = (np.mean(trials), np.var(trials))

     def value_boost(self, option):
         trials = BoostTest.montecarlo(option) ##returns np.ndarray
         self._value = (np.mean(trials), np.var(trials)) 
                 
     def montecarlo(self, option):
        rng = np.random.normal(size=1e5)
        discount = np.exp(-option._r * option._T)
        payoff = lambda sim: np.maximum(
            (sim - option._K) if option._type == OptionType.call 
            else (option._K - sim), 
            0)
        sampled = option._S0 * np.exp((option._r - option._div - 0.5 * option._sigma ** 2) *option._T +
                                          option._sigma * np.sqrt(option._T) * rng)
        vals = payoff(sampled)
        return discount * np.sum(vals) / len(vals)


class BinomialTree(PricingStrategy):
    def __init__(self):
        super().__init__()
