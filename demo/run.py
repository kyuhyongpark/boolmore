# run with the location of the script as the working directory


from boolmore.algo.genetic_algorithm import run_ga
from boolmore.core.conversions import prime2bnet, prime2rr

# seed for the random number generator
SEED = 0

base, start, final, log = run_ga(
    run_type="NAV",
    json_file="CAD_config.json",
    start_model="CAD_start.bnet",
    run_name="demo",
    stop_if_max=True,
    seed=SEED)

print("\n-----comparing with the baseline functions-----")
modified = 0
for node in base.model.primes:
    if prime2rr(base.model.primes[node])[1] != prime2rr(final.model.primes[node])[1]:
        modified += 1
        print("base:" + prime2bnet(node, base.model.primes[node]))
        print("final:" + prime2bnet(node, final.model.primes[node]))

print(f"\n{modified} out of {len(base.model.primes)} functions differ from the baseline")