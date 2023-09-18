import json
import random
import pystablemotifs as sm
import boolmore.config
from boolmore.model import Model, mix_models
from boolmore.experiment import import_exps
from boolmore.conversions import prime2bnet

SETTINGS = 'BoolMoRe/ABA_case_study/data/ABA_2017.json'
START_MODEL = None
NAME = "ref_20230917"

def run_ga(json_file:str, start_model:str|None=None, name:str|None=None):
    """
    Imports parameters, experiments, base model in the json file
    Optionally imports starting model and the output file name.
    Runs genetic algorithm and exports refined models.

    Parameters
    ----------

    json_file   - location of the json file containing parameters   :str

    name        - name of the generated models                      :str|None
                  models are exported as name_id_gen.txt
                  log is exported as name_log_txt
                  if None, takes the start_model as the name
    start_model - location of the txt file of the starting model    :str|None
                  if None, base_model is the starting model

    """
    f = open(json_file)
    json_dict = json.load(f)

    # take parameters from the json file
    STARTING_ID = json_dict['parameters']['starting_id']
    STARTING_GEN = json_dict['parameters']['starting_gen']
    TOTAL_ITERATIONS = json_dict['parameters']['total_iterations']
    PER_ITERATION = json_dict['parameters']['per_iteration']
    KEEP = json_dict['parameters']['keep']
    EXPORT_TOP = json_dict['parameters']['export_top']
    EXPORT_THRESHOLD = json_dict['parameters']['export_threshold']
    PROB = json_dict['parameters']['prob']
    EDGE_PROB = json_dict['parameters']['edge_prob']

    # take model specific data from the json file
    DEFAULT_SOURCES = json_dict['default_sources']
    CONSTRAINTS = json_dict['constraints']
    EDGE_POOL = json_dict['edge_pool']

    DATA = json_dict['data']
    BASE = json_dict['base']

    if start_model != None:
        START_MODEL = start_model
    # if starting model is not given, take the base as the start
    else:
        START_MODEL = BASE

    if name != None:
        NAME = name
    else:
        NAME = START_MODEL.split("/")[-1][:-4]
    FILE = NAME + '_log.txt'

    boolmore.config.id = STARTING_ID


    print("Loading experimental data . . .")
    exps, pert, max_score = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = sm.format.import_primes(BASE)
    base = Model.import_model(base_primes, constraints=CONSTRAINTS,
                              edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)
    print("Base model loaded.")
    base.max_score = max_score
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    print("Loading starting model . . .")
    primes = sm.format.import_primes(START_MODEL)
    n1 = Model.import_model(primes, boolmore.config.id, STARTING_GEN, base)
    print("Starting model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps)
    n1.info()
    print()


    fp = open(FILE, "w")

    fp.write(f'# {DATA=}\n')
    fp.write(f'# {DEFAULT_SOURCES=}\n')
    fp.write(f'# {CONSTRAINTS=}\n')
    fp.write(f'# {EDGE_POOL=}\n')
    fp.write(f'\n# {BASE=}\n')
    fp.write(f'# extra edges: {base.extra_edges}\n')
    fp.write(f'# score: {base.score}\n')
    with open(BASE, 'r') as base_text:
        for line in base_text:
            if not line.startswith('#') and not line.isspace():
                fp.write('# ' + line)
    fp.write(f'\n# {START_MODEL=}\n')
    if BASE != START_MODEL:
        fp.write(f'# score: {n1.score}\n')
        fp.write(f'# extra edges: {n1.extra_edges}\n')
        with open(START_MODEL, 'r') as model_text:
            for line in model_text:
                if not line.startswith('#') and not line.isspace():
                    fp.write('# ' + line)


    ### Genetic Algorithm
    print("- - - - - iteration ", 0, " - - - - -")
    iteration = []
    iteration.append(n1)
    for i in range(PER_ITERATION-1):
        new_model = n1.mutate(PROB, EDGE_PROB)
        new_model.get_predictions(pert)
        new_model.get_model_score(exps)
        iteration.append(new_model)
    
    iteration = sorted(iteration, key=lambda x: (len(x.extra_edges), x.complexity))
    iteration = sorted(iteration, key=lambda x: x.score, reverse=True)

    final = iteration[0]    
    for i in range(EXPORT_TOP):
        iteration[i].export(NAME, EXPORT_THRESHOLD)

    print("top score :", round(iteration[0].score,2))
    print("extra edges :", len(iteration[0].extra_edges))
    print("complexity of the top model :", iteration[0].complexity)
    if not iteration[0].check_constraint():
        print("ERROR: model does not follow constraints")

    fp.write('iteration\ttop score\textra edges\tcomplexity\n')
    fp.write('0\t'+ str(iteration[0].score) +'\t')
    fp.write(str(len(iteration[0].extra_edges)) +'\t')
    fp.write(str(iteration[0].complexity) +'\n')
    
    for i in range(TOTAL_ITERATIONS):
        print("- - - - - iteration ", i+1, " - - - - -")
        new_iteration = []
        # keep the best ones
        for j in range(KEEP):
            new_iteration.append(iteration[j])
    
        # mix the best ones
        to_be_mixed = sorted(new_iteration, key=lambda x: (len(x.extra_edges), x.complexity))
        to_be_mixed = sorted(to_be_mixed, key=lambda x: x.score, reverse=True)
        weights = list(range(1, KEEP+1))
        weights.reverse()
        for j in range(KEEP):
            mix = random.choices(to_be_mixed, weights=weights, k=2)
            mixed_model = mix_models(mix[0], mix[1])
            mixed_model.get_predictions(pert)
            mixed_model.get_model_score(exps)
            new_iteration.append(mixed_model)
    
        # mutate the best ones
        weights = list(range(1, 2*KEEP+1))
        weights.reverse()
        new_iteration = sorted(new_iteration, key=lambda x: (len(x.extra_edges), x.complexity))
        new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)
        targets = random.choices(new_iteration, weights=weights, k=PER_ITERATION-2*KEEP)
        for target in targets:
            new_model = target.mutate(PROB, EDGE_PROB)
            new_model.get_predictions(pert)
            new_model.get_model_score(exps)
            new_iteration.append(new_model)
        
        new_iteration = sorted(new_iteration, key=lambda x: (len(x.extra_edges), x.complexity))
        new_iteration = sorted(new_iteration, key=lambda x: x.score, reverse=True)
    
        print("top score : ", round(new_iteration[0].score,2))
        print("number of added edges in the top model :", len(new_iteration[0].extra_edges))
        print("complexity of the top model :", new_iteration[0].complexity)
        if not new_iteration[0].check_constraint():
            print("ERROR: model does not follow constraints")
        fp.write(str(i+1) +'\t'+ str(new_iteration[0].score) +'\t')
        fp.write(str(len(new_iteration[0].extra_edges)) +'\t')
        fp.write(str(new_iteration[0].complexity) +'\n')

        # Export models that exceed the threshold score
        for j in range(EXPORT_TOP):
            new_iteration[j].export(NAME, EXPORT_THRESHOLD)

        # Always export the final best model
        final = new_iteration[0]

        # Stop iteration if max score is reached
        if new_iteration[0].score == base.max_score:
            final = new_iteration[0]
            print("max score reached")
            break

        iteration = new_iteration

    mutated = set()
    for node in n1.primes:
        for value in [0, 1]:
            sorted_primes1 = sorted([sorted(d.items()) for d in n1.primes[node][value]])
            sorted_primes2 = sorted([sorted(d.items()) for d in final.primes[node][value]])
            if sorted_primes1 != sorted_primes2:
                mutated.add(node)
    mutated = sorted(list(mutated))

    print()
    print(f"""The algorithm ran for {i+1} iterations,
          generating {boolmore.config.id - STARTING_ID} models.\n
          Mutated {len(mutated)} functions, 
          and increased score from {round(n1.score,2)} to {round(final.score,2)}.""")
    print()

    final.export(NAME, 0)
    final.info()
    print()

    for node in mutated:
        print("start:" + prime2bnet(node, n1.primes[node])) # type: ignore
        print("final:" + prime2bnet(node, final.primes[node])) # type: ignore

    return base, n1, final

if __name__=='__main__':
    run_ga(SETTINGS, START_MODEL, NAME)