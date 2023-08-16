from __future__ import annotations

import random
import pyboolnet.trap_spaces
import mutation as m
import conversions as conv
import constraint as cons
import score
import config

class Model():
    def __init__(self):
        self.id = None
        self.generation = None

        self.base = None
        self.constraints = {'fixed': set(), 'regulate': {}, 'necessary' : {}, 'group': {}, 'possible_constant': set()}
        self.edge_pool = []
        self.default_sources = {}
        
        self.primes = None
        self.regulators = None
        self.signs = None
        self.rrs = None
        self.extra_edges = None

        self.complexity = None
        self.predictions = None
        self.score = None

    @classmethod
    def import_model(cls, primes:PrimesType, id:int=0, generation:int=0,
                     base:Model|None = None, constraints:dict={}, edge_pool:list[tuple[str]]=[], default_sources:dict[str,int]={}) -> Model:
        x = cls()
        x.id = id
        x.generation = generation

        x.base = base
        x.constraints.update(constraints)
        x.edge_pool.extend(edge_pool)
        x.default_sources.update(default_sources)

        x.primes = primes
        x.regulators = {}
        x.signs = {}
        x.rrs = {}
        x.extra_edges = []

        x.complexity = 0

        for node in primes:
            # find current regulators and signs
            regulators, rr, signs = conv.prime2rr(primes[node], regulators = None, signs = None)

            # check the extra edges (signs are not checked yet)
            for edge in edge_pool:
                if edge[1] == node and edge[0] in regulators:
                    x.extra_edges.append(edge)

            # get complexity
            for prime_implicant in primes[node][1]:
                x.complexity += len(prime_implicant)

            if base == None:
                x.regulators[node] = regulators
                x.rrs[node] = rr
                x.signs[node] = signs

            else:
                regulators = list(base.regulators[node])
                signs = base.signs[node]

                for edge in x.extra_edges:
                    if edge[1] == node:
                        regulators.append(edge[0])
                        signs += edge[2]

                regulators = tuple(regulators)

                regulators, rr, signs = conv.prime2rr(primes[node], regulators=regulators, signs=signs)
                x.regulators[node] = regulators
                x.rrs[node] = rr
                x.signs[node] = signs

        if base == None:
            x.base = x

        return x

    def mutate(self, probability, edge_prob, bias = 0.5):
        config.id += 1
        mutated_model = Model()
        mutated_model.id = config.id
        mutated_model.generation = self.generation + 1

        mutated_model.base = self.base
        mutated_model.constraints = self.constraints
        mutated_model.edge_pool = self.edge_pool
        mutated_model.default_sources = self.default_sources

        mutated_model.primes = self.primes.copy()
        mutated_model.regulators = self.regulators.copy()
        mutated_model.signs = self.signs.copy()        
        mutated_model.rrs = self.rrs.copy()
        mutated_model.extra_edges = self.extra_edges.copy()

        mutated_model.complexity = 0

        rnd = random.random()
        if rnd < edge_prob:
            new_edge = random.choice(self.edge_pool)

            new_edge_node = new_edge[1]
            new_regulator = new_edge[0]
            new_sign = new_edge[2]

            regulators = mutated_model.regulators[new_edge_node]
            rr = mutated_model.rrs[new_edge_node]
            signs = mutated_model.signs[new_edge_node]

            if new_regulator not in regulators:
                mutated_model.extra_edges.append(new_edge)
                modified_regulators, modified_rr, modified_signs = m.add_regulator(regulators, rr, signs, new_regulator, new_sign)
            elif new_regulator in regulators:
                mutated_model.extra_edges.remove(new_edge)
                modified_regulators, modified_rr, modified_signs = m.delete_regulator(regulators, rr, signs, new_regulator)

            mutated_model.regulators[new_edge_node] = modified_regulators
            mutated_model.rrs[new_edge_node] = modified_rr
            mutated_model.signs[new_edge_node] = modified_signs
            prime1 = conv.rr2prime(modified_regulators, modified_rr, modified_signs, inverted = False)
            mutated_model.primes[new_edge_node] = prime1

        for node in mutated_model.rrs:
            # get mutated_rr from rr
            mutated_rr, modified = m.mutate_rr_constraint(mutated_model.regulators[node],
                                                          mutated_model.rrs[node],
                                                          mutated_model.base.rrs[node],
                                                          mutated_model.constraints,
                                                          node, probability, bias)
            mutated_model.rrs[node] = mutated_rr

            # get primes from the mutated_rr
            # if the representations are equivalent, take the old prime
            if modified:
                prime1 = conv.rr2prime(mutated_model.regulators[node], mutated_rr, mutated_model.signs[node], inverted = False)
                mutated_model.primes[node] = prime1
                # irr = get_max_irr(mutated_model.rrs[node])
                # prime2 = rr2prime(mutated_model.regulators[node], irr, mutated_model.signs[node], inverted = True)
                # assert prime1 == prime2, "rr and irr lead to different result!"
            
            # get complexity
            for prime_implicant in mutated_model.primes[node][1]:
                mutated_model.complexity += len(prime_implicant)        

        return mutated_model

    def check_constraint(self) -> bool:
        check = True
        for node in self.primes:
            check = cons.check_node(self.regulators[node], self.rrs[node], self.base.rrs[node], self.constraints, node) and check
        return check

    def get_predictions(self, interventions):
        '''
        Assigns self.predictions when given interventions

        Parameters
        ----------
        interventions - list of fixes : list[FixesType]
            fixes - ((nodeA, value1),(nodeB, value2), ...) : tuple(tuple(str, int))
                    
        Assigns
        -------
        self.predictions : dict
            keys : fixes
            values : dict
                average value of every node in the attractors
        '''
        predictions = {}
        for fixes in interventions:
            perturbation = {}
            for fix in fixes:
                perturbation[fix[0]] = fix[1]
            # print("- - - - - - - - - -")
            # print("fixed: ", perturbation)

            new_primes = self.primes.copy()
            for node in perturbation.keys():
                assert node in new_primes.keys(), f"{node} is not in the model"
                if int(perturbation[node]) == 0:
                    new_primes[node] = [[{}],[]]
                else:
                    new_primes[node] = [[],[{}]]

            tr = pyboolnet.trap_spaces.compute_trap_spaces(new_primes, "min")

            for i in tr:
                for node in self.primes.keys():
                    if node not in i.keys():
                        i[node] = '?'
                    else:
                        i[node] = str(i[node])

            result = {}
            for i in tr:
                for node in self.primes.keys():
                    if node not in result.keys():
                        result[node] = 0.0
                    if i[node] == '1':
                        result[node] += (1.0/len(tr))
                    elif i[node] == '?':
                        result[node] += (0.5/len(tr))

            predictions[fixes] = result

        self.predictions = predictions

    def get_model_score(self, exps, report = False, file = None):
        '''
        To be modified to meet the criteria
        '''
        # self.score = score.get_score(exps, self.predictions, self.extra_edges)
        agreements = score.get_agreement(exps, self.predictions)
        self.score = score.get_hierarchy_score(agreements, self.default_sources, report=report, file=file)

    def export(self, name, threshold = 0.0):
        # TODO: print out total score as well

        '''
        Exports the model rules with scores above a certain threshold.

        Parameters
        ----------
        threshold : float
            the threshold score

        Returns
        -------
        None

        Exports
        -------
        id :

        '''
        if threshold != 0.0 and self.score < threshold:
            return
        fp = open(name + "_" + str(self.id) + "_gen" + str(self.generation) + ".txt", "w")
        fp.write("# id: " + str(self.id) + '\n')
        fp.write('# generation: ' + str(self.generation) + '\n')
        fp.write('# extra edges: ' + str(self.extra_edges) + '\n')
        fp.write('# score: ' + str(self.score) + '\n')
        fp.write('# following constraints: ' + str(self.check_constraint()) + '\n')
        fp.write('# complexity: ' + str(self.complexity) + '\n')
        primes = {k:self.primes[k] for k in sorted(self.primes)}
        for k,v in primes.items():
            s = k + "* = "
            sl = []
            for c in v[1]:
                sll = []
                for kk,vv in c.items():
                    if vv: sli = kk
                    else: sli = '!'+kk
                    sll.append(sli)
                if len(sll) > 0:
                    sl.append(' & '.join(sll))
            if len(sl) > 0:
                s += ' | '.join(sl)
            if v[1]==[]:
                s = k + "* = 0"
            if v[1]==[{}]:
                s = k + "* = 1"
            fp.write(s + '\n')
        fp.close()

    def info(self):
        # TODO: print out total score as well
        print('id: ', self.id)
        print('generation: ', self.generation)
        print('extra edges: ', self.extra_edges)
        print('score: ', self.score)
        print('following constraints:', self.check_constraint())
        print('complexity:', self.complexity)


def mix_models(model1:Model, model2:Model) -> Model:
    config.id += 1
    mixed_model = Model()
    mixed_model.id = config.id
    mixed_model.generation = max(model1.generation,model2.generation) + 1

    mixed_model.base = model1.base
    mixed_model.constraints = model1.constraints
    mixed_model.edge_pool = model1.edge_pool
    mixed_model.default_sources = model1.default_sources

    mixed_model.primes = {}
    mixed_model.regulators = {}
    mixed_model.signs = {}    
    mixed_model.rrs = {}
    mixed_model.extra_edges = []

    mixed_model.complexity = 0

    for node in model1.rrs:
        # get mutated_rr from rr
        rnd = random.random()
        if rnd < 0.5:
            get = model1
        else:
            get = model2
        mixed_model.primes[node] = get.primes[node]
        mixed_model.regulators[node] = get.regulators[node]
        mixed_model.signs[node] = get.signs[node]
        mixed_model.rrs[node] = get.rrs[node]
        for edge in get.extra_edges:
            if edge[1] == node:
                mixed_model.extra_edges.append(edge)

        # get complexity
        for prime_implicant in mixed_model.primes[node][1]:
            mixed_model.complexity += len(prime_implicant)

    return mixed_model