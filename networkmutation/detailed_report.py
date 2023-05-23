import pystablemotifs as sm
from Model import *
from experiment import *
from mutation import *
import config
import pyboolnet.trap_spaces
import pyboolnet.prime_implicants

run_type = 'osc'

# BASE = 'networkmutation/baseline/ABA_full_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_full_fix_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_A_20230501.txt'
BASE = 'networkmutation/baseline/ABA_GA_base_B_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_A_20230511_ca_ext_cis.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_B_20230511_ca_ext_cis.txt'

MODEL = BASE
### GA_try
# MODEL = 'networkmutation/models/try_20230522_7854_gen121.txt'
### GA0
# MODEL = 'networkmutation/models/no_edge_20230303_3807_gen54_mod.txt'
### GA1
# MODEL = 'networkmutation/models/20230512_7790_gen125_mod.txt'
### GA2
# MODEL = 'networkmutation/models/osc_20230514_8025_gen128.txt'

if run_type == 'normal':
    DATA = 'networkmutation/data_20230426.tsv'
    DEFAULT_SOURCES = {'ABA':0}
    ### constraints for the typical run
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}
    ### constraints for the 'more edges' version
    ### allow change of Ca2c
    # CONSTRAINTS = {
    #     'fixed': {'Ca2_ATPase', 'Closure', 'DAG', 'InsP3', 'H2O_Efflux', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #     'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
    #     'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #     'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #     'possible_constant': {'GEF1_4_10',}
    #     }

    ### edge pool for the typical run
    EDGE_POOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
                 ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                 ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'))
    ### No edge version
    # EDGE_POOL = tuple()
    ### more edges version 20230315 - 17 edges
    # EDGE_POOL = (('PA', 'ABI2', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'),
    #              ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'),
    #              ('ABI1', 'GEF1_4_10', '1'),
    #              ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'),
    #              ('OST1', 'CaIM', '1'), 
    #              ('GHR1', 'KOUT', '1'),
    #              ('NO', 'KEV', '1'),
    #              ('pHc', 'NIA1_2', '1'),
    #              ('CIS', 'AnionEM', '1'),
    #              ('GPA1', 'Ca2c', '1'), ('PA', 'Ca2c', '1'))
elif run_type == 'osc':
    DATA = 'networkmutation/data_osc_20230426.tsv'
    DEFAULT_SOURCES = {'ABA':0}
    ### constraints for the typical run for ca_osc model
    CONSTRAINTS = {'fixed': {'Ca2osc', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}
    ### constraints for the 'more edges' version
    ### Allow modification of Ca2osc
    # CONSTRAINTS = {'fixed': {'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2osc':('CaIM','CIS')},
    #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #                'possible_constant': {'GEF1_4_10',}}

    ### edge pool for the typical run for the ca_osc model - 16 edges
    EDGE_POOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
                 ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                 ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'))
    ### more edges version 20230315 - 16 edges
    # EDGE_POOL = (('PA', 'ABI2', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'),
    #              ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'),
    #              ('ABI1', 'GEF1_4_10', '1'),
    #              ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'),
    #              ('OST1', 'CaIM', '1'), 
    #              ('GHR1', 'KOUT', '1'),
    #              ('NO', 'KEV', '1'),
    #              ('pHc', 'NIA1_2', '1'),
    #              ('CIS', 'AnionEM', '1'),
    #              ('GPA1', 'Ca2osc', '1'), ('PA', 'Ca2osc', '1'))
elif run_type == 'two':
    DATA = 'networkmutation/data_two_20230317.txt'
    DEFAULT_SOURCES = {'ABA':0}
    ### constraints for the typical run for ca_two model
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c_l', 'Ca2c_h', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}
    
    ### typical run for the ca_two model
    EDGE_POOL = (('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                 ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('Microtubule_Depolymerization', 'ROS', '1'), ('H2O_Efflux', 'H2O_Efflux', '1'))
elif run_type == 'ca_ext':
    DATA = 'networkmutation/data_ca_ext_20230511.tsv'
    DEFAULT_SOURCES = {'ABA':0}
    ### constraints for the typical run
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}

    ### edge pool for the typical run
    EDGE_POOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
                 ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                 ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'))
elif run_type == 'osc_ca_ext':
    DATA = 'networkmutation/data_osc_ca_ext_20230511.tsv'
    DEFAULT_SOURCES = {'ABA':0}
    ### constraints for the typical run
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2osc', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_constant': {'GEF1_4_10',}}

    ### edge pool for the typical run
    EDGE_POOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
                 ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                 ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'))



FILE_NAME = MODEL.split("/")[-1][:-4]+'_score.csv'

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
    n1 = Model.Model.import_model(primes, CONSTRAINTS, EDGE_POOL, DEFAULT_SOURCES, id=config.id, generation=0, base=base)
    print("Starting model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps, report=True, file=FILE_NAME)
    n1.info()
    print()

    ### percolate the constants and reduce the model
    pprimes1 = pyboolnet.prime_implicants.percolate(primes, remove_constants = True, copy=True)
    print('percolated rules: ')
    sm.format.pretty_print_prime_rules({k:pprimes1[k] for k in sorted(pprimes1)})
    print()

    ### find nodes that have different functions
    pprimes0 = pyboolnet.prime_implicants.percolate(base_primes, remove_constants = True, copy=True)
    different_nodes = []
    for node in pprimes0:
        if node not in pprimes1:
            different_nodes.append(node)
            continue
        if pprimes0[node] == pprimes1[node]:
            pass
        else:
            different_nodes.append(node)
    for node in pprimes1:
        if node not in pprimes0:
            different_nodes.append(node)
    print('Nodes with changed percolated functions: ', sorted(different_nodes))
    print()

    ### find the edges that are deleted
    deleted_regulators = {}
    deleted_regulators_perc = {}
    for node in base_primes:
        base_regulators = pyboolnet.prime_implicants.find_predecessors(base_primes, [node])
        regulators = pyboolnet.prime_implicants.find_predecessors(primes, [node])
        deleted_regulator_list = [x for x in base_regulators if x not in regulators]
        if len(deleted_regulator_list) != 0:
            deleted_regulators[node] = deleted_regulator_list
        ### find extra edges that are deleted in the percolated rules
        if node in pprimes0 and node in pprimes1:
            base_regulators_perc = pyboolnet.prime_implicants.find_predecessors(pprimes0, [node])
            regulators_perc = pyboolnet.prime_implicants.find_predecessors(pprimes1, [node])
            deleted_regulator_list_perc = [x for x in base_regulators_perc if x not in regulators_perc and x not in deleted_regulator_list]
            if len(deleted_regulator_list_perc) != 0:
                deleted_regulators_perc[node] = deleted_regulator_list_perc
    print('deleted regulators: ',deleted_regulators)
    print('extra deleted regulators in the percolated rules: ',deleted_regulators_perc)
    print()

    ### find stable motifs
    ### TODO: not include the source nodes.
    stable_motifs = pyboolnet.trap_spaces.compute_trap_spaces(pprimes1, "max")
    print("Stable Motifs:")
    for stable_motif in stable_motifs:
        print(stable_motif)
    print()

    ### fix extra nodes to find conditionally stable motifs
    ### WARNING: fixing constant nodes do not work, as they are already percolated
    print("Conditionally Stable Motifs:")
    fixes = [{'ABA': 0}, {'ABA': 1}]
    for fix in fixes:
        print('fix - ', fix)
        pprimes2 = pyboolnet.prime_implicants.percolate(pprimes1, add_constants = fix, remove_constants = True, copy=True)
        # sm.format.pretty_print_prime_rules({k:primes2[k] for k in sorted(primes2)})
        conditinally_stable_motifs = pyboolnet.trap_spaces.compute_trap_spaces(pprimes2, "max")
        for conditinally_stable_motif in conditinally_stable_motifs:
            if conditinally_stable_motif not in stable_motifs:
                print(conditinally_stable_motif)
    print()

    ### fix nodes to find minimal trapspaces and LDOIs
    ### fixing constant nodes are not allowed
    print("LDOI, DOI and minimal trapspaces:")
    fixes = [{'ABA':0},
             {'ABA':1},
             {'ROS':1},
             {'NO':1},
             {'CaIM':1},
             {'cADPR':1},
             {'PA':1},
             {'pHc':1},
             {'S1P_PhytoS1P':1},
             {'AtRAC1':0},
             {'InsP3':1}]
    for fix in fixes:
        print("- - - - - - - - - -")
        print("fix -", fix)
        LDOI, LDOI_contra = sm.drivers.logical_domain_of_influence(fix,pprimes1)
        MPBN_DOI, MPBN_DOI_contra, MPBN_unknown, MPBN_unknown_contra, MPBN_ar = sm.drivers.domain_of_influence(fix,pprimes1,MPBN_update=True)

        print('LDOI: ',{k:LDOI[k] for k in sorted(LDOI)})
        print('only in MPBN_DOI: ',{k:MPBN_DOI[k] for k in MPBN_DOI if k not in LDOI})
        print()

        if 'ABA' not in fix:
            fix.update({'ABA':0})
        
        pprimes2 = pyboolnet.prime_implicants.percolate(pprimes1, add_constants = fix, remove_constants = False, copy=True)
        trs = pyboolnet.trap_spaces.compute_trap_spaces(pprimes2, "min")
        for tr in trs:
            for node in pprimes1.keys():
                if node not in tr.keys():
                    tr[node] = 'X'
        
        if len(trs) == 1:
            print("There is 1 minimal trapspace for ", dict(sorted(fix.items())))
        else:
            print("There are " + str(len(trs)) + " minimal trapspaces for", dict(sorted(fix.items())))
        for tr in trs:
            print(dict(sorted(tr.items())))
        print()