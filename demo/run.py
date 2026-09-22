# run with the location of the script as the working directory


from boolmore.genetic.algorithm import run_ga
from boolmore.boolean_functions import prime2bnet, prime2rr

# seed for the random number generator
SEED = 0
CORE = 2

base, start, states = run_ga(
    json_file="CAD_config.json",
    seed=SEED,
    run_name="demo",
    core=CORE
    )

final = states[-1].population[0]

print("\n-----comparing with the baseline functions-----")
modified = 0
for node in base.model.primes:
    if prime2rr(base.model.primes[node])[1] != prime2rr(final.model.primes[node])[1]:
        modified += 1
        print("base:" + prime2bnet(node, base.model.primes[node]))
        print("final:" + prime2bnet(node, final.model.primes[node]))

print(f"\n{modified} out of {len(base.model.primes)} functions differ from the baseline")