# run with the location of the script as the working directory


from boolmore.genetic_algorithm import run_ga
from boolmore.conversions import prime2bnet, prime2rr

# seed for the random number generator
SEED = 0

base, start, final, log = run_ga("CAD_config.json",
                                 "CAD_start.bnet",
                                 run_name="demo",
                                 stop_if_max=True,
                                 seed=SEED)

print("\n-----comparing with the baseline functions-----")
modified = 0
for node in base.primes:
    if prime2rr(base.primes[node])[1] != prime2rr(final.primes[node])[1]:
        modified += 1
        print("base:" + prime2bnet(node, base.primes[node]))
        print("final:" + prime2bnet(node, final.primes[node]))

print(f"\n{modified} out of {len(base.primes)} functions differ from the baseline")