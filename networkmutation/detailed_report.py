import pystablemotifs as sm
from Model import *
from experiment import *
from mutation import *
import config

run_type = ''

if run_type != 'osc' and run_type != 'two':
    DATA = 'networkmutation/data_1202.txt'
    ### constraints for the typical run
    # CONSTRAINTS = {
    #     'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #     'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'H2O_Efflux':('K_efflux',), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
    #     'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1'), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #     'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #     'possible_source': {'AquaporinPIP2_1','GEF1_4_10'}
    #     }
    ### constraints for the 'more edges' version
    ### allow change of Ca2c
    CONSTRAINTS = {
        'fixed': {'Ca2_ATPase', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
        'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'H2O_Efflux':('K_efflux',), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
        'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1'), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
        'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
        'possible_source': {'AquaporinPIP2_1','GEF1_4_10'}
        }
    ### typical run
    # EDGE_POOL = (('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
    #              ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'), ('Actin_Reorganization', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
    #              ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'),
    #              ('Microtubule_Depolymerization', 'ROS', '1'), ('H2O_Efflux', 'H2O_Efflux', '1'))
    ### No edge version
    # EDGE_POOL = (('H2O_Efflux', 'H2O_Efflux', '1'),)
    ### more edges version 20230315 - 17 edges
    EDGE_POOL = (('PA', 'ABI2', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'ROS', '1'), ('Actin_Reorganization', 'RBOH', '1'),
                 ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'),
                 ('ABI1', 'GEF1_4_10', '1'),
                 ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('H2O_Efflux', 'H2O_Efflux', '1'),
                 ('OST1', 'CaIM', '1'), 
                 ('GHR1', 'KOUT', '1'),
                 ('NO', 'KEV', '1'),
                 ('pHc', 'NIA1_2', '1'),
                 ('CIS', 'AnionEM', '1'),
                 ('GPA1', 'Ca2c', '1'), ('PA', 'Ca2c', '1'))
elif run_type == 'osc':
    DATA = 'networkmutation/data_osc_1202.txt'
    ### constraints for the typical run for ca_osc model
    # CONSTRAINTS = {
    #     'fixed': {'Ca2osc', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
    #     'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'H2O_Efflux':('K_efflux',), 'Depolarization':('AnionEM',)},
    #     'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1'), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
    #     'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
    #     'possible_source': {'AquaporinPIP2_1','GEF1_4_10'}
    #     }
    ### constraints for the 'more edges' version
    ### Allow modification of Ca2osc
    CONSTRAINTS = {'fixed': {'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'H2O_Efflux':('K_efflux',), 'Depolarization':('AnionEM',), 'Ca2osc':('CaIM','CIS')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1'), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_source': {'AquaporinPIP2_1','GEF1_4_10'}}
    ### typical run for the ca_osc model - 16 edges
    # EDGE_POOL = (('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
    #              ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
    #              ('AquaporinPIP2_1', 'ROS', '1'), ('Actin_Reorganization', 'ROS', '1'),
    #              ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
    #              ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
    #              ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
    #              ('PA', 'Microtubule_Depolymerization', '1'),
    #              ('Microtubule_Depolymerization', 'ROS', '1'))
    ### more edges version 20230315 - 16 edges
    EDGE_POOL = (('PA', 'ABI2', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'),
                 ('Actin_Reorganization', 'ROS', '1'), ('Actin_Reorganization', 'RBOH', '1'),
                 ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'),
                 ('ABI1', 'GEF1_4_10', '1'),
                 ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('OST1', 'CaIM', '1'), 
                 ('GHR1', 'KOUT', '1'),
                 ('NO', 'KEV', '1'),
                 ('pHc', 'NIA1_2', '1'),
                 ('CIS', 'AnionEM', '1'),
                 ('GPA1', 'Ca2osc', '1'), ('PA', 'Ca2osc', '1'))
elif run_type == 'two':
    DATA = 'networkmutation/data_two_20230317.txt'
    ### constraints for the typical run for ca_two model
    CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c_l', 'Ca2c_h', 'Closure', 'DAG', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                   'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'H2O_Efflux':('K_efflux',), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
                   'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'H2O_Efflux':('AnionEM','AquaporinPIP2_1'), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                   'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                   'possible_source': {'AquaporinPIP2_1','GEF1_4_10'}}
    ### typical run for the ca_two model
    EDGE_POOL = (('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                 ('AquaporinPIP2_1', 'ROS', '1'), ('Actin_Reorganization', 'ROS', '1'),
                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                 ('GPA1', 'OST1', '1'), ('GHR1', 'CPK3_21', '1'),
                 ('PA', 'Microtubule_Depolymerization', '1'),
                 ('Microtubule_Depolymerization', 'ROS', '1'), ('H2O_Efflux', 'H2O_Efflux', '1'))


# BASE = 'PyStableMotifs/models/ABA_full.txt'
BASE = 'PyStableMotifs/models/ABA_full_fix.txt'
# BASE = 'PyStableMotifs/models/ABA_calosc.txt'
# BASE = 'PyStableMotifs/models/ABA_calosc_cis.txt'
# BASE = 'PyStableMotifs/models/ABA_calosc_mod.txt'


# MODEL = 'PyStableMotifs/models/ABA_full.txt'
MODEL = 'networkmutation/models/20230215_3995_gen59_mod.txt'
# MODEL = 'PyStableMotifs/models/ABA_full_fix.txt'
# MODEL = 'PyStableMotifs/models/ABA_calosc.txt'
### GA1
# MODEL = 'networkmutation/models/20230215_3995_gen59.txt'
### GA2-1
# MODEL = 'networkmutation/models/osc_20221126_8100_gen79.txt'
### GA2-2
# MODEL = 'networkmutation/models/osc_cis_20221127_4424_gen61_mod.txt'
### GA2-3
# MODEL = 'networkmutation/models/osc_20230221_7884_gen75.txt'

FILE_NAME = MODEL.split("/")[-1][:-4]+'_score.csv'

if __name__ == '__main__':
    print("Loading experimental data . . .")
    exps, pert = import_exps(DATA)
    print("Experimental data loaded.\n")

    print("Loading base model . . .")
    base_primes = sm.format.import_primes(BASE)
    base = Model.Model.import_model(base_primes, CONSTRAINTS, EDGE_POOL)
    print("Base model loaded.")
    base.get_predictions(pert)
    base.get_model_score(exps, 0)
    base.info()
    print()

    print("Loading starting model . . .")
    primes = sm.format.import_primes(MODEL)
    n1 = Model.Model.import_model(primes, CONSTRAINTS, EDGE_POOL, id=config.id, generation=0, base=base)
    print("Starting model loaded.")
    n1.get_predictions(pert)
    n1.get_model_score(exps, 0, True, FILE_NAME)
    n1.info()
    print()

