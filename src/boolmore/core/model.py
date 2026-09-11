from __future__ import annotations
import random
import os
import pickle

import boolmore.algo.mutation as m
import boolmore.core.conversions as conv
import boolmore.eval.constraint as cons

PrimeType = list[list[dict[str, int]]]
FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict]


class Model():
    def __init__(self):
        """
        Attributes
        ----------
        id              - unique id for a model in a single run                     :int
        generation      - starting model considered as 0th gen                      :int
                          first mutated models are 1st gen
        name            - name of the model                                         :str
        
        base            - the base model (not the neccesarily the starting model)   :Model class
                          from which the regulators, fixed functions, constants,
                          extra edges, etc. are decided.
        constraints     - represents 5 types of constraints                         :dict[str, list or dict]
                          (fixed, regulate, necessary, group, possible_constant)
        edge_pool       - the pool of edges. 0 is negative, 1 is positive           :list[list[str]]
                          [[regulator, target, sign], ...]
        default_sources - Shows the default settings for the source nodes,          :dict[str, int]
                          which is considered the top of the hierarchy
                          These source nodes must have a defined value in
                          every experiments

        primes          - pyboolnet primes dictionary                               :length N dict[str, PrimeType]
                          {node: prime}                          
        regulators_dict - dictionary of the regulating nodes                        :length N dict[str, tuple[str]]
        signs_dict      - dictionary of the signs of regulators                     :length N dict[str, str]
        rr_dict         - dictionary of the binary rule representations             :length N dict[str, str]
        extra_edges     - edges from the pool that are present in the model         :list[list[str]]
                          [[regulator, target, sign], ...]

        n_edges         - number of edges in the model                              :int
        n_extra_edges   - number of extra edges in the model                        :int
        n_self_edges    - number of self-edges in the model                         :int
        n_prime_implicants - number of prime implicants in the model                 :int

        """
        self.id = 0
        self.generation = 0
        self.name = ""

        self.base = None
        self.constraints = {"fixed": [], "regulate": {}, "necessary" : {},
                            "group": {}, "possible_constant": []}
        self.edge_pool = []
        
        self.primes:dict[str, PrimeType] = {}
        self.regulators_dict = {}
        self.signs_dict = {}
        self.rr_dict = {}
        self.extra_edges = []
        self.n_edges = 0
        self.n_extra_edges = 0
        self.n_self_edges = 0
        self.n_prime_implicants = 0

    @classmethod
    def import_model(cls, primes:dict[str, PrimeType], id:int=-1, generation:int=0,
                     base:Model|None=None, constraints:dict={}, edge_pool:list[list[str]]=[],
        ) -> Model:
        """
        Import a model.
        If base=None, the output model is considered the base model,
        and the given primes is used to construct it.

        Parameters
        ----------

        primes          - pyboolnet primes dictionary                               :length N dict[str, PrimeType]
                          {node: prime}
        id              - unique id for a model in a single run                     :int
        generation      - starting model considered as 0th gen                      :int
                          first mutated models are 1st gen
                          
        base            - the base model (not the neccesarily the starting model)   :Model class
                          from which the regulators, fixed functions, constants,
                          extra edges, etc. are decided.
                          if None, the output model is considered the base

        # if base is given, below parameters take the value of the base
        constraints     - represents 5 types of constraints                         :dict[str, list or dict]
                          (fixed, regulate, necessary, group, possible_constant)
        edge_pool       - the pool of edges. 0 is negative, 1 is positive           :list[list[str]]
                          [[regulator, target, sign], ...]

        Returns
        -------
        model - the imported model :Model class

        """
        x = cls()

        x.id = id
        x.generation = generation
        x.primes = primes

        # get constraints, edge pool
        if base == None:
            x.constraints.update(constraints)
            x.edge_pool.extend(edge_pool)

        else:
            x.base = base
            x.constraints.update(base.constraints)
            x.edge_pool.extend(base.edge_pool)
            x.name = base.name
        
        for node in x.primes:
            # find current regulators and signs
            regulators, rr, signs = conv.prime2rr(x.primes[node])

            # check the extra edges (TODO: check signs)
            for edge in x.edge_pool:
                if edge[1] == node and edge[0] in regulators: # type: ignore
                    x.extra_edges.append(edge)

            if base == None:
                x.regulators_dict[node] = regulators
                x.rr_dict[node] = rr
                x.signs_dict[node] = signs

            else:
                regulators = list(base.regulators_dict[node])
                signs = base.signs_dict[node]

                for edge in x.extra_edges:
                    if edge[1] == node:
                        regulators.append(edge[0])
                        signs += edge[2]

                regulators = tuple(regulators)

                rr = conv.prime2rr(primes[node], regulators=regulators, signs=signs)[1] # type: ignore
                x.regulators_dict[node] = regulators
                x.rr_dict[node] = rr
                x.signs_dict[node] = signs

        if base == None:
            x.base = x

        x.get_complexity()

        x.check_constraint()

        return x

    def check_constraint(self) -> bool:
        """
        Checks if the model follows the constraints.
        It does not check group constraints yet.
        TODO: implement group constraint check

        Returns
        -------
        check - True if the model follows constraints       :bool

        """
        check = True
        for node in self.primes:
            check = cons.check_node(self.regulators_dict[node], self.rr_dict[node], self.base.rr_dict[node], # type: ignore
                                    self.constraints, node) and check
        return check

    def get_complexity(self):
        self.n_edges = 0
        self.n_self_edges = 0
        self.n_prime_implicants = 0
        
        for node in self.primes:
            regulators_set = set()
            for prime_implicant in self.primes[node][1]:
                self.n_prime_implicants += len(prime_implicant)

                for reg in prime_implicant:
                    regulators_set.add(reg)
            self.n_edges += len(regulators_set)
            if node in regulators_set:
                self.n_self_edges += 1

    def mutate(self, model_id:int, probability:float, edge_prob:float, bias:float=0.5, seed:int|None=None) -> Model:
        """
        Returns a mutated model.

        Parameters
        ----------
        probability - probability that each binary is mutated               :float between 0 and 1
        edge_prob   - probability to add or delete an edge from the pool    :float between 0 and 1
        bias        - probability to invert rr when mutating                :float between 0 and 1
        seed        - random seed                                           :int|None

        Returns
        -------
        mutated_model - a new model with mutated functions  :Model class

        """
        mutated_model = Model()
        mutated_model.id = model_id
        mutated_model.generation = self.generation + 1

        mutated_model.base = self.base
        mutated_model.constraints = self.constraints
        mutated_model.edge_pool = self.edge_pool
        mutated_model.name = self.name

        mutated_model.primes = self.primes.copy()
        mutated_model.regulators_dict = self.regulators_dict.copy()
        mutated_model.signs_dict = self.signs_dict.copy()        
        mutated_model.rr_dict = self.rr_dict.copy()
        mutated_model.extra_edges = self.extra_edges.copy()

        if seed != None:
            random.seed(seed)

        rnd = random.random()
        if rnd < edge_prob and len(mutated_model.edge_pool)>0:
            new_edge = random.choice(self.edge_pool)

            new_edge_node = new_edge[1]
            new_regulator = new_edge[0]
            new_sign = new_edge[2]

            regulators = mutated_model.regulators_dict[new_edge_node]
            rr = mutated_model.rr_dict[new_edge_node]
            signs = mutated_model.signs_dict[new_edge_node]

            if new_regulator not in regulators:
                mutated_model.extra_edges.append(new_edge)
                modified_regulators, modified_rr, modified_signs = m.add_regulator(regulators, rr, signs, new_regulator, new_sign)
            else:
                mutated_model.extra_edges.remove(new_edge)
                modified_regulators, modified_rr, modified_signs = m.delete_regulator(regulators, rr, signs, new_regulator)

            mutated_model.regulators_dict[new_edge_node] = modified_regulators
            mutated_model.rr_dict[new_edge_node] = modified_rr
            mutated_model.signs_dict[new_edge_node] = modified_signs
            prime1 = conv.rr2prime(modified_regulators, modified_rr, modified_signs, inverted = False)
            mutated_model.primes[new_edge_node] = prime1

        for node in mutated_model.rr_dict:
            # get mutated_rr from rr
            mutated_rr, modified = m.mutate_rr_constraint(mutated_model.regulators_dict[node],
                                                          mutated_model.rr_dict[node],
                                                          mutated_model.base.rr_dict[node], # type: ignore
                                                          mutated_model.constraints,
                                                          node, probability, bias)
            mutated_model.rr_dict[node] = mutated_rr

            # get primes from the mutated_rr
            # if the representations are equivalent, take the old prime
            if modified:
                prime1 = conv.rr2prime(mutated_model.regulators_dict[node], mutated_rr, mutated_model.signs_dict[node], inverted = False)
                mutated_model.primes[node] = prime1
                # irr = get_max_irr(mutated_model.rr_dict[node])
                # prime2 = rr2prime(mutated_model.regulators_dict[node], irr, mutated_model.signs_dict[node], inverted = True)
                # assert prime1 == prime2, "rr and irr lead to different result!"
            
        mutated_model.get_complexity()       

        return mutated_model

    def info(self):
        """
        prints out a brief summary of the model info
        """
        print("id: ", self.id)
        print("generation: ", self.generation)
        print("extra edges: ", self.extra_edges)
        print("following constraints:", self.check_constraint())
        print("number of edges: ", self.n_edges)
        print("number of self edges: ", self.n_self_edges)
        print("number of prime implicants: ", self.n_prime_implicants)

    def export(self, file_name:str|None=None, details:bool=True):
        """
        Exports the model rules.

        Parameters
        ----------
        file_name : str
            location of the output file
            if None, output file is in the form "(model's name)_id_gen.txt"
        details : bool
            whether to print out the details of the model as comments
            
        """
        if file_name == None:
            file_name = self.name + "_" + str(self.id) + "_gen" + str(self.generation)
        bnet_file_name = file_name + ".bnet"
        pkl_file_name = file_name + ".pkl"


        # write bnet file
        fp = open(bnet_file_name, "w")

        if details:
            fp.write("# id: " + str(self.id) + "\n")
            fp.write("# generation: " + str(self.generation) + "\n")
            fp.write("# extra edges: " + str(self.extra_edges) + "\n")
            # fp.write("# score: " + str(self.score) + " / " + str(self.max_score) + "\n")
            fp.write("# following constraints: " + str(self.check_constraint()) + "\n")
            fp.write("# number of edges: " + str(self.n_edges) + "\n")
            fp.write("# number of self edges: " + str(self.n_self_edges) + "\n")
            fp.write("# number of prime implicants: " + str(self.n_prime_implicants) + "\n\n")

        fp.write("targets,\tfactors\n")
        primes = {k:self.primes[k] for k in sorted(self.primes)}
        for k in primes:
            s = conv.prime2bnet(k, primes[k])
            fp.write(s + "\n")
        fp.close()
        
        # also pickle primes
        with open(pkl_file_name, "wb") as f:
            pickle.dump(self.primes, f)

        print("Exported generated model to", os.path.abspath(bnet_file_name))
        print("Pickled primes to", os.path.abspath(pkl_file_name))


def mix_models(model_id:int, model1:Model, model2:Model) -> Model:
    """
    For each node, take the rule from one of the parent model randomly.

    Parameters
    ----------
    model1, model2 - parent models                              :Model class
    
    Returns
    -------
    mixed_model - model that takes functions from the parents   :Model class
    """
    mixed_model = Model()
    mixed_model.id = model_id
    mixed_model.generation = max(model1.generation,model2.generation) + 1

    mixed_model.base = model1.base
    mixed_model.constraints = model1.constraints
    mixed_model.edge_pool = model1.edge_pool
    mixed_model.name = model1.name

    for node in model1.rr_dict:
        # get mutated_rr from rr
        rnd = random.random()
        if rnd < 0.5:
            get = model1
        else:
            get = model2
        mixed_model.primes[node] = get.primes[node]
        mixed_model.regulators_dict[node] = get.regulators_dict[node]
        mixed_model.signs_dict[node] = get.signs_dict[node]
        mixed_model.rr_dict[node] = get.rr_dict[node]
        for edge in get.extra_edges:
            if edge[1] == node:
                mixed_model.extra_edges.append(edge)

    mixed_model.get_complexity()

    return mixed_model