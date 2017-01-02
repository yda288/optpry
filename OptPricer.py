from models import *
from pprint import pprint

def main():
    import time
    start = time.time()  
    myCall = EuropeanOption(opt_type=OptionType.call, 
                            asset_price=100., 
                            strike=100., 
                            rho=.01, 
                            q=0., 
                            maturity_years=0.5, 
                            vol=0.35, 
                            pricing_model=MonteCarlo(True))
    px = myCall.price()
    end = time.time()
    elapsed = end - start
    pprint('Time elapsed {0}'.format(elapsed))

    pprint(myCall.model.__dict__)

if __name__ == "__main__":
    main()