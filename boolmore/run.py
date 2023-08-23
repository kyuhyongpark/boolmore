import random
import pystablemotifs as sm
import config
from model import Model, mix_models
from experiment import import_exps


NAME = config.NAME
FILE = NAME + '_log.txt'

# take parameters from the config file
STARTING_ID = config.parameters['starting_id']
STARTING_GEN = config.parameters['starting_gen']
TOTAL_ITERATIONS = config.parameters['total_iterations']
PER_ITERATION = config.parameters['per_iteration']
KEEP = config.parameters['keep']
EXPORT_TOP = config.parameters['export_top']
EXPORT_THRESHOLD = config.parameters['export_threshold']
PROB = config.parameters['prob']
EDGE_PROB = config.parameters['edge_prob']

# if starting model is not given, take the base as the start
if config.MODEL != None:
    STARTING_MODEL = config.MODEL
else:
    STARTING_MODEL = config.data_bank[config.RUN_TYPE]['base']

DATA = config.data_bank[config.RUN_TYPE]['data']
BASE = config.data_bank[config.RUN_TYPE]['base']
DEFAULT_SOURCES = config.data_bank[config.RUN_TYPE]['default_sources']
CONSTRAINTS = config.data_bank[config.RUN_TYPE]['constraints']
EDGE_POOL = config.data_bank[config.RUN_TYPE]['edge_pool']


config.id = STARTING_ID

if __name__ == '__main__':
    print("Loading experimental data . . .")
    exps, pert = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = sm.format.import_primes(BASE)
    base = Model.import_model(base_primes, constraints=CONSTRAINTS, edge_pool=EDGE_POOL, default_sources=DEFAULT_SOURCES)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    print("Loading starting model . . .")
    primes = sm.format.import_primes(STARTING_MODEL)
    n1 = Model.import_model(primes, config.id, STARTING_GEN, base)
    print("Starting model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps)
    n1.info()
    print()

    # ### make one mutated model
    # new_model = n1.mutate(PROB, EDGE_PROB)
    # new_model.get_predictions(pert)
    # new_model.get_model_score(exps)
    # new_model.info()
    # new_model.export(NAME)

    # from pyboolnet.prime_implicants import primes_are_equal
    # print(primes_are_equal(n1.primes,new_model.primes))

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
    fp.write(f'\n# {STARTING_MODEL=}\n')
    if BASE != STARTING_MODEL:
        fp.write(f'# score: {n1.score}\n')
        fp.write(f'# extra edges: {n1.extra_edges}\n')
        with open(STARTING_MODEL, 'r') as model_text:
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
    
    for i in range(EXPORT_TOP):
        iteration[i].export(NAME, EXPORT_THRESHOLD)

    print("top score :", iteration[0].score)
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
    
        for j in range(EXPORT_TOP):
            new_iteration[j].export(NAME, EXPORT_THRESHOLD)

        # Always export the final best model
        if i+1 == TOTAL_ITERATIONS:
            new_iteration[0].export(NAME, 0)

        print("top score : ", new_iteration[0].score)
        print("extra edges :", len(new_iteration[0].extra_edges))
        print("complexity of the top model :", new_iteration[0].complexity)
        if not new_iteration[0].check_constraint():
            print("ERROR: model does not follow constraints")
        fp.write(str(i+1) +'\t'+ str(new_iteration[0].score) +'\t')
        fp.write(str(len(new_iteration[0].extra_edges)) +'\t')
        fp.write(str(new_iteration[0].complexity) +'\n')
    
        iteration = new_iteration