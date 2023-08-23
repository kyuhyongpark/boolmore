# name of the generated models and the log
NAME = 'ref_20230822'

# location of the starting model
STARTING_MODEL = None

# name of the key to use in the data bank
RUN_TYPE = 'ABA_normal'     


parameters = {
    'starting_id': 1,           
    'starting_gen': 1,          # starting generation
    'total_iterations': 100,
    'per_iteration': 100,       # models per iteration
    'keep': 20,                 # best models to keep for the next iteration
    'export_top': 1,            # number of top models to export when exceeding threshold score
    'export_threshold': 430,    # minimum score for export (best one at the end of the run is always exported)
    'prob': 0.01,               # probability to mutate a binary in the rule representation
    'edge_prob': 0.5            # probability to add or delete an edge from the pool
}

data_bank = {
    'ABA_normal':{
        'data': 'networkmutation/data/data_20230426.tsv',
        'base': 'networkmutation/baseline/ABA_GA_base_A_20230501.txt',
        'default_sources': {'ABA':0},
        ### constraints for the typical run 20230815
        'constraints': {'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                        'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
                        'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                        'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                        'possible_constant': set()},
        ### edge pool for the typical run 20230801 - 13 edges
        'edge_pool': [('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
                      ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                      ('AquaporinPIP2_1', 'ROS', '1'),
                      ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                      ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                      ('GHR1', 'CPK3_21', '1'),
                      ('PA', 'Microtubule_Depolymerization', '1')]
        },

    'ABA_normal_more_edges':{
        'data': 'networkmutation/data/data_20230426.tsv',
        'base': 'networkmutation/baseline/ABA_GA_base_A_20230501.txt',
        'default_sources': {'ABA':0},
        ### constraints for the 'more edges' version 20230815 - allow change of Ca2c, move to regulate
        'constraints' : {'fixed': {'Ca2_ATPase', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                         'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
                         'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                         'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                         'possible_constant': set()},
        ### more edges version 20230704 - 14 edges
        'edge_pool': [('PA', 'ABI2', '0'),
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
                      ('InsP3', 'SPHK1_2', '1'),
                      ('ABA', 'GEF1_4_10', '0')]
        },

    'ABA_osc':{
        'data': 'networkmutation/data/data_osc_20230426.tsv',
        'base': 'networkmutation/baseline/ABA_GA_base_B_20230407.txt',
        'default_sources': {'ABA':0},
        ### constraints for the typical run of the ca_osc model 20230815
        'constraints': {'fixed': {'Ca2osc', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                        'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
                        'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                        'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                        'possible_constant': set()},
        ### edge pool for the typical run of the ca_osc model 20230801 - 13 edges
        'edge_pool': [('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
                      ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
                      ('AquaporinPIP2_1', 'ROS', '1'),
                      ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
                      ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
                      ('GHR1', 'CPK3_21', '1'),
                      ('PA', 'Microtubule_Depolymerization', '1')],
        },
    
    'ABA_osc_more_edges':{
        'data': 'networkmutation/data/data_osc_20230426.tsv',
        'base': 'networkmutation/baseline/ABA_GA_base_B_20230407.txt',
        'default_sources': {'ABA':0},
        ### constraints for the 'more edges' version of ca_osc model 20230815 - allow modification of Ca2osc, move to regulate
        'constraints': {'fixed': {'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
                        'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2osc':('CaIM','CIS')},
                        'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
                        'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
                        'possible_constant': set()},
        ### more edges version 20230704 - 14 edges
        'edge_pool': [('PA', 'ABI2', '0'),
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
                      ('InsP3', 'SPHK1_2', '1'),
                      ('ABA', 'GEF1_4_10', '0')],
        }
    }

# STARTING_ID = 1
# STARTING_GEN = 1
# ITERATIONS = 100
# PER_ITERATION = 100
# KEEP = 20
# EXPORT_TOP = 1
# EXPORT_THRESHOLD = 430
# PROB = 0.01
# EDGE_PROB = 0.5

# osc = False
# more_edges = False
# run_type = 'normal'

# DEFAULT_SOURCES = {'ABA':0}



# # BASE = 'networkmutation/baseline/ABA_full_20230407.txt'
# # BASE = 'networkmutation/baseline/ABA_full_fix_20230407.txt'
# BASE = 'networkmutation/baseline/ABA_GA_base_A_20230501.txt'
# # BASE = 'networkmutation/baseline/ABA_GA_base_B_20230407.txt'

# MODEL = BASE
# ### GA1
# # MODEL = 'networkmutation/models/20230512_7790_gen125_mod.txt'
# ### GA2
# # MODEL = 'networkmutation/models/osc_20230514_8025_gen128.txt'




# DATA = 'networkmutation/data/data_20230426.tsv'

# ### constraints for the typical run
# CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Ca2c', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
#                 'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
#                 'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
#                 'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
#                 'possible_constant': set()}
# # ### constraints for the 'more edges' version 20230524 - allow change of Ca2c, move to regulate
# # CONSTRAINTS = {'fixed': {'Ca2_ATPase', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
# #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2c':('CaIM','CIS','Ca2_ATPase')},
# #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
# #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
# #                'possible_constant': set()}

# ### edge pool for the typical run 20230801 - 13 edges
# EDGE_POOL = [('Ca2c', 'ABI2', '0'),('Ca2c', 'HAB1', '0'),('Ca2c', 'PP2CA', '0'),
#                 ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
#                 ('AquaporinPIP2_1', 'ROS', '1'),
#                 ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
#                 ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
#                 ('GHR1', 'CPK3_21', '1'),
#                 ('PA', 'Microtubule_Depolymerization', '1')]
# # ### more edges version 20230704 - 14 edges
# # EDGE_POOL = (('PA', 'ABI2', '0'),
# #              ('AquaporinPIP2_1', 'ROS', '1'),
# #              ('Actin_Reorganization', 'RBOH', '1'),
# #              ('ROS', 'Actin_Reorganization', '1'),
# #              ('pHc', 'Vacuolar_Acidification', '1'),
# #              ('PA', 'Microtubule_Depolymerization', '1'),
# #              ('GHR1', 'KOUT', '1'),
# #              ('NO', 'KEV', '1'),
# #              ('CIS', 'AnionEM', '1'),
# #              ('GPA1', 'Ca2c', '1'), ('PA', 'Ca2c', '1'),
# #              ('pHc', 'SPHK1_2', '1'),
# #              ('InsP3', 'SPHK1_2', '1'),
# #              ('ABA', 'GEF1_4_10', '0'))

# if run_type == 'osc':
#     DATA = 'networkmutation/data/data_osc_20230426.tsv'

#     ### constraints for the typical run for ca_osc model
#     CONSTRAINTS = {'fixed': {'Ca2osc', 'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
#                    'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',)},
#                    'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
#                    'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
#                    'possible_constant': set()}
#     # ### constraints for the 'more edges' version - allow modification of Ca2osc, move to regulate
#     # CONSTRAINTS = {'fixed': {'Closure', 'DAG', 'H2O_Efflux', 'InsP3', 'InsP6', 'NO', 'PtdIns3_5P2', 'PtdIns4_5P2', 'RCARs', 'cADPR', 'cGMP'},
#     #                'regulate': {'ABI1':('RCARs',), 'ABI2':('RCARs',), 'HAB1':('RCARs',), 'PP2CA':('RCARs',), 'K_efflux':('KEV','KOUT'), 'OST1':('ABI1','ABI2'), 'Depolarization':('AnionEM',), 'Ca2osc':('CaIM','CIS')},
#     #                'necessary' : {'8-nitro-cGMP':('cGMP',), 'KOUT':('Depolarization',), 'Malate':('PEPC', 'AnionEM'), 'ROS':('NADPH', 'RBOH')},
#     #                'group': {'PA':(('PC','PLDalpha'),('PC','PLDdelta'),('DAG','DAGK')), 'S1P_PhytoS1P':(('SPHK1_2','Sph'),)},
#     #                'possible_constant': set()}

#     ### edge pool for the typical run for the ca_osc model 20230801 - 13 edges
#     EDGE_POOL = [('Ca2osc', 'ABI2', '0'),('Ca2osc', 'HAB1', '0'),('Ca2osc', 'PP2CA', '0'),
#                  ('PA', 'ABI2', '0'),('PA', 'HAB1', '0'),('PA', 'PP2CA', '0'),
#                  ('AquaporinPIP2_1', 'ROS', '1'),
#                  ('Actin_Reorganization', 'RBOH', '1'), ('ROS', 'Actin_Reorganization', '1'),
#                  ('pHc', 'Vacuolar_Acidification', '1'), ('ABI1', 'GEF1_4_10', '1'),
#                  ('GHR1', 'CPK3_21', '1'),
#                  ('PA', 'Microtubule_Depolymerization', '1')]
#     # ### more edges version 20230704 - 14 edges
#     # EDGE_POOL = (('PA', 'ABI2', '0'),
#     #              ('AquaporinPIP2_1', 'ROS', '1'),
#     #              ('Actin_Reorganization', 'RBOH', '1'),
#     #              ('ROS', 'Actin_Reorganization', '1'),
#     #              ('pHc', 'Vacuolar_Acidification', '1'),
#     #              ('PA', 'Microtubule_Depolymerization', '1'),
#     #              ('GHR1', 'KOUT', '1'),
#     #              ('NO', 'KEV', '1'),
#     #              ('CIS', 'AnionEM', '1'),
#     #              ('GPA1', 'Ca2osc', '1'), ('PA', 'Ca2osc', '1'),
#     #              ('pHc', 'SPHK1_2', '1'),
#     #              ('InsP3', 'SPHK1_2', '1'),
#     #              ('ABA', 'GEF1_4_10', '0'))





id = 0
