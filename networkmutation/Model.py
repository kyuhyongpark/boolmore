import random
import pyboolnet.trap_spaces
import mutation as m
import constraint as cons
import score
import config

class Model():
    def __init__(self):
        self.id = None
        self.generation = None
        self.regulators = None
        self.signs = None
        self.constraints = {'fixed': set(), 'regulate': {}, 'necessary' : {},
        'group': {}, 'possible_source': set()}
        self.edge_pool = []
        self.primes = None
        self.rrs = None
        self.extra_edges = None
        self.predictions = None
        self.score = None

    @classmethod
    def import_model(cls, primes, constraints={}, edge_pool=[], id = 0, generation = 0, base = None):
        x = cls()
        x.id = id
        x.generation = generation
        x.regulators = {}
        x.signs = {}
        x.constraints.update(constraints)
        x.edge_pool.extend(edge_pool)
        x.primes = primes
        x.rrs = {}
        x.extra_edges = []

        for node in primes:
            # find current regulators and signs
            regulators, rr, signs = m.prime2rr(primes[node], regulators = None, signs = None)

            # check the extra edges (signs are not checked yet)
            for edge in edge_pool:
                if edge[1] == node and edge[0] in regulators:
                    x.extra_edges.append(edge)

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

                regulators, rr, signs = m.prime2rr(primes[node], regulators=regulators, signs=signs)
                x.regulators[node] = regulators
                x.rrs[node] = rr
                x.signs[node] = signs

        return x

    def mutate(self, probability, edge_prob, bias = 0.5):
        config.id += 1
        mutated_model = Model()
        mutated_model.id = config.id
        mutated_model.generation = self.generation + 1
        mutated_model.regulators = self.regulators.copy()
        mutated_model.signs = self.signs.copy()
        mutated_model.constraints = self.constraints
        mutated_model.edge_pool = self.edge_pool
        mutated_model.primes = self.primes.copy()
        mutated_model.rrs = self.rrs.copy()
        mutated_model.extra_edges = self.extra_edges.copy()

        modify_edge = False
        new_edge_node = None
        rnd = random.random()
        if rnd < edge_prob:
            new_edge = random.choice(self.edge_pool)
            new_edge_node = new_edge[1]
            regulators = mutated_model.regulators[new_edge_node]
            new_regulator = new_edge[0]
            modify_edge = True

        if modify_edge and new_regulator not in regulators:
            rr = mutated_model.rrs[new_edge_node]
            signs = mutated_model.signs[new_edge_node]
            new_sign = new_edge[2]

            mutated_model.extra_edges.append(new_edge)
            modified_regulators, modified_rr, modified_signs = m.add_regulator(regulators, rr, signs, new_regulator, new_sign)

        elif modify_edge and new_regulator in regulators:
            rr = mutated_model.rrs[new_edge_node]
            signs = mutated_model.signs[new_edge_node]

            mutated_model.extra_edges.remove(new_edge)
            modified_regulators, modified_rr, modified_signs = m.delete_regulator(regulators, rr, signs, new_regulator)

        if modify_edge:
            mutated_model.regulators[new_edge_node] = modified_regulators
            mutated_model.rrs[new_edge_node] = modified_rr
            mutated_model.signs[new_edge_node] = modified_signs
            prime1 = m.rr2prime(modified_regulators, modified_rr, modified_signs, inverted = False)
            mutated_model.primes[new_edge_node] = prime1

        for node in mutated_model.rrs:
            # get mutated_rr from rr
            regulators = mutated_model.regulators[node]
            rr = mutated_model.rrs[node]
            signs = mutated_model.signs[node]
            mutated_rr, modified = m.mutate_rr_constraint(regulators, rr, mutated_model.constraints, node, probability)
            mutated_model.rrs[node] = mutated_rr

            # get primes from the mutated_rr
            # if the representations are equivalent, take the old prime
            if modified or node == new_edge_node:
                prime1 = m.rr2prime(regulators, mutated_rr, signs, inverted = False)
                mutated_model.primes[node] = prime1
                # irr = get_max_irr(rr)
                # prime2 = rr2prime(regulators, irr, signs, inverted = True)
                # assert prime1 == prime2, "rr and irr lead to different result!"

        return mutated_model

    def check_constraint(self):
        for node in self.primes:
            check = cons.check_node(self.regulators[node], self.rrs[node], self.constraints, node)
            if check == False:
                return check
        return check

    def get_predictions(self, interventions):
        '''
        Returns predictions when given interventions

        Parameters
        ----------
        interventions : list of tuples
            interventions in the form [((nodeA, value1),(nodeB, value2), ...), ...]

        Returns
        -------
        self.predictions : dict
            keys : interventions
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

    def get_model_score(self, exps):
        '''
        To be modified to meet the criteria
        '''
        # self.score = score.get_score(exps, self.predictions, self.extra_edges)
        agreements = score.get_agreement(exps, self.predictions)
        self.score = score.get_hierarchy_score(exps, agreements, self.extra_edges)

    def export(self, name, threshold = 0.0):
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
        print('id: ', self.id)
        print('generation: ', self.generation)
        print('extra edges: ', self.extra_edges)
        print('score: ', self.score)
        print('following constraints:', self.check_constraint())
