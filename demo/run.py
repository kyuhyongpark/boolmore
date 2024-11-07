from boolmore.genetic_algorithm import run_ga
from boolmore.conversions import prime2bnet, prime2rr

base, start, final = run_ga("CAD_config.json",
                            "CAD_start.bnet",
                            run_name="demo",
                            stop_if_max=True)

print("\n-----comparing with the baseline functions-----")
modified = 0
for node in base.primes:
    if prime2rr(base.primes[node])[1] != prime2rr(final.primes[node])[1]:
        modified += 1
        print("base:" + prime2bnet(node, base.primes[node])) # type: ignore
        print("final:" + prime2bnet(node, final.primes[node])) # type: ignore

print(f"\n{modified} out of {len(base.primes)} functions differ from the baseline")