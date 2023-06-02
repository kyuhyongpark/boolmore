import pystablemotifs as sm
from Model import *
from experiment import *
from mutation import *
import config

RUN_GA = True
NAME = 'osc_ca_ext_more_edges_20230602'
run_type = 'osc_ca_ext'

# BASE = 'networkmutation/baseline/ABA_full_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_full_fix_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_A_20230501.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_B_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_A_20230511_ca_ext.txt'
BASE = 'networkmutation/baseline/ABA_GA_base_B_20230511_ca_ext.txt'

MODEL = BASE
### GA_try
# MODEL = 'networkmutation/models/try_20230522_7854_gen121.txt'
### GA0
# MODEL = 'networkmutation/models/no_edge_20230303_3807_gen54_mod.txt'
### GA1
# MODEL = 'networkmutation/models/20230512_7790_gen125_mod.txt'
### GA2
# MODEL = 'networkmutation/models/osc_20230514_8025_gen128.txt'

DEFAULT_SOURCES = {'ABA':0}
if run_type == 'normal':
    DATA = 'networkmutation/data_20230426.tsv'

    # ### constraints for the typical run
    # CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
    #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #                'possible_constant': {'GEF1_4_10',}}
    ### constraints for the 'more edges' version 20230524 - allow change of Ca2c, move to regulate
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}

    # ### edge pool for the typical run
    # EDGE_POOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
    #              ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
    #              ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'))
    # ### No edge version
    # EDGE_POOL = tuple()
    ### more edges version 20230529 - 13 edges
    EDGE_POOL = (('PA', 'ABI2', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'),
                 ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('GHR1', 'KOUT', '1'),
                 ('NO', 'KEV', '1'),
                 ('CIS', 'AnionEM', '1'),
                 ('GPA1', 'Ca2c', '1'), ('PA', 'Ca2c', '1'),
                 ('pHc', 'SPHK1_2', '1'),
                 ('InsP3', 'SPHK1_2', '1'))

elif run_type == 'osc':
    DATA = 'networkmutation/data_osc_20230426.tsv'

    # ### constraints for the typical run for ca_osc model
    # CONSTRAINTS = {'fixed': {'Ca2osc', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
    #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #                'possible_constant': {'GEF1_4_10',}}
    ### constraints for the 'more edges' version - allow modification of Ca2osc, move to regulate
    CONSTRAINTS = {'fixed': {'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2osc':('CaIM','CIS')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}

    # ### edge pool for the typical run for the ca_osc model - 16 edges
    # EDGE_POOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
    #              ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
    #              ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'))
    ### more edges version 20230529 - 13 edges
    EDGE_POOL = (('PA', 'ABI2', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'),
                 ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('GHR1', 'KOUT', '1'),
                 ('NO', 'KEV', '1'),
                 ('CIS', 'AnionEM', '1'),
                 ('GPA1', 'Ca2osc', '1'), ('PA', 'Ca2osc', '1'),
                 ('pHc', 'SPHK1_2', '1'),
                 ('InsP3', 'SPHK1_2', '1'))

elif run_type == 'ca_ext':
    DATA = 'networkmutation/data_ca_ext_20230511.tsv'

    # ### constraints for the typical run
    # CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
    #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #                'possible_constant': {'GEF1_4_10',}}
    ### constraints for the 'more edges' version 20230524 - allow change of Ca2c, move to regulate
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2c':('Ca_ext','CIS','Ca2_ATPase')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}

    # ### edge pool for the typical run
    # EDGE_POOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
    #              ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
    #              ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'))
    ### more edges version 20230529 - 13 edges
    EDGE_POOL = (('PA', 'ABI2', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'),
                 ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('GHR1', 'KOUT', '1'),
                 ('NO', 'KEV', '1'),
                 ('CIS', 'AnionEM', '1'),
                 ('GPA1', 'Ca2c', '1'), ('PA', 'Ca2c', '1'),
                 ('pHc', 'SPHK1_2', '1'),
                 ('InsP3', 'SPHK1_2', '1'))

elif run_type == 'osc_ca_ext':
    DATA = 'networkmutation/data_osc_ca_ext_20230511.tsv'

    # ### constraints for the typical run
    # CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2osc', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
    #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #                'possible_constant': {'GEF1_4_10',}}
    ### constraints for the 'more edges' version - allow modification of Ca2osc, move to regulate
    CONSTRAINTS = {'fixed': {'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2osc':('Ca_ext','CIS')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}

    # ### edge pool for the typical run
    # EDGE_POOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
    #              ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
    #              ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'))
    ### more edges version 20230529 - 13 edges
    EDGE_POOL = (('PA', 'ABI2', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'),
                 ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('GHR1', 'KOUT', '1'),
                 ('NO', 'KEV', '1'),
                 ('CIS', 'AnionEM', '1'),
                 ('GPA1', 'Ca2osc', '1'), ('PA', 'Ca2osc', '1'),
                 ('pHc', 'SPHK1_2', '1'),
                 ('InsP3', 'SPHK1_2', '1'))


FILE = NAME + '_log.txt'
STARTING_ID = 1
STARTING_GEN = 1
ITERATIONS = 100
PER_ITERATION = 100
KEEP = 20
EXPORT_TOP = 1
EXPORT_THRESHOLD = 430
PROB = 0.01
EDGE_PROB = 0.5

config.id = STARTING_ID

if __name__ == '__main__':
    print("Loading experimental data . . .")
    exps, pert = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = sm.format.import_primes(BASE)
    base = Model.Model.import_model(base_primes, CONSTRAINTS, EDGE_POOL, DEFAULT_SOURCES)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps)
    base.info()
    print()

    print("Loading starting model . . .")
    primes = sm.format.import_primes(MODEL)
    n1 = Model.Model.import_model(primes, CONSTRAINTS, EDGE_POOL, DEFAULT_SOURCES, id=config.id, generation=STARTING_GEN, base=base)
    print("Starting model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps)
    n1.info()
    print()

    # ### make one mutated model
    # new_model = n1.mutate(PROB, EDGE_PROB)
    # new_model.get_predictions(pert)
    # new_model.get_model_score(exps, PENALTY)
    # new_model.info()
    # new_model.export(NAME)

    # from pyboolnet.prime_implicants import primes_are_equal
    # print(primes_are_equal(n1.primes,new_model.primes))

    if RUN_GA == True:
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
        fp.write(f'\n# {MODEL=}\n')
        if BASE != MODEL:
            fp.write(f'# score: {n1.score}\n')
            fp.write(f'# extra edges: {n1.extra_edges}\n')
            with open(MODEL, 'r') as model_text:
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
        fp.write('iteration\ttop score\textra edges\tcomplexity\n')
        fp.write('0\t'+ str(iteration[0].score) +'\t')
        fp.write(str(len(iteration[0].extra_edges)) +'\t')
        fp.write(str(iteration[0].complexity) +'\n')
        
        for i in range(ITERATIONS):
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
            if i+1 == ITERATIONS:
                new_iteration[0].export(NAME, 0)

            print("top score : ", new_iteration[0].score)
            print("extra edges :", len(new_iteration[0].extra_edges))
            print("complexity of the top model :", new_iteration[0].complexity)
            fp.write(str(i+1) +'\t'+ str(new_iteration[0].score) +'\t')
            fp.write(str(len(new_iteration[0].extra_edges)) +'\t')
            fp.write(str(new_iteration[0].complexity) +'\n')
        
            iteration = new_iteration