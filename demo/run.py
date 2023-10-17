import boolmore.genetic_algorithm
from boolmore.conversions import prime2bnet

base, start, final = boolmore.genetic_algorithm.run_ga("CAD_config.json", "CAD_start.txt",
                                                       run_name="demo", stop_if_max=True)

for node in base.primes:
    print("base:" + prime2bnet(node, base.primes[node])) # type: ignore
    print("final:" + prime2bnet(node, final.primes[node])) # type: ignore