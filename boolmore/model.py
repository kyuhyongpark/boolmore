from __future__ import annotations
import random
import os

import pyboolnet.trap_spaces

import boolmore.mutation as m
import boolmore.conversions as conv
import boolmore.constraint as cons
import boolmore.score as score
import boolmore.config as config


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
        complexity      - shows the complexity of the model functions,              :int
                          by summing the number of prime implicants

        predictions : PredictType
            average attractor values for all fixes
            keys : FixesType
            values : dict[str, float]
                average values of nodes - {observed_node: predict_value}

        score : float
            how well the model agrees with experimental results
            one point in score means agreement to one perturbation

        non_hierarchy_score : float
            how well the model agrees with experimental results, ignoring hierarchy
 
        max_score : float
            possible maximum score w/o hierarchy

        """
        self.id = 0
        self.generation = 0
        self.name = ""

        self.base = None
        self.constraints = {"fixed": [], "regulate": {}, "necessary" : {},
                            "group": {}, "possible_constant": []}
        self.edge_pool = []
        self.default_sources = {}
        
        self.primes:dict[str, PrimeType] = {}
        self.regulators_dict = {}
        self.signs_dict = {}
        self.rr_dict = {}
        self.extra_edges = []
        self.complexity = 0

        self.predictions:PredictType = {}
        self.score = 0.0
        self.non_hierarchy_score = 0.0
        self.max_score = 0.0

    @classmethod
    def import_model(cls, primes:dict[str, PrimeType], id:int=0, generation:int=0,
                     base:Model|None=None, constraints:dict={}, edge_pool:list[list[str]]=[],
                     default_sources:dict[str,int]={}) -> Model:
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
        default_sources - Shows the default settings for the source nodes,          :dict[str, int]
                          which is considered the top of the hierarchy
                          if given an empty dict, all sources being 0 is taken
                          as the default value
        Returns
        -------
        model           - model with all attributes except predictions and score    :Model class

        """
        x = cls()

        x.id = id
        x.generation = generation
        x.primes = primes

        # get constraints, edge pool, and default sources
        if base == None:
            x.constraints.update(constraints)
            x.edge_pool.extend(edge_pool)
            x.default_sources.update(default_sources)
            if len(x.default_sources) == 0:
                generate_default_sources = True
            else:
                generate_default_sources = False
        else:
            x.base = base
            x.constraints.update(base.constraints)
            x.edge_pool.extend(base.edge_pool)
            x.default_sources.update(base.default_sources)
            generate_default_sources = False
            x.max_score = base.max_score
            x.name = base.name
        
        for node in x.primes:
            # generate default_sources if necessary
            if generate_default_sources:
                if x.primes[node] == [[{node:0}],[{node:1}]]:
                    x.default_sources[node] = 0

            # get complexity
            for prime_implicant in x.primes[node][1]:
                x.complexity += len(prime_implicant)

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

    def get_predictions(self, interventions:list[FixesType]):
        """
        Assigns self.predictions when given interventions

        Parameters
        ----------
        interventions - summarized list of fixes for convenience    :list[FixesType]
            fixes     - ((nodeA, value1),(nodeB, value2), ...)      :FixesType = tuple[tuple[str, int]]
                    
        Assigns
        -------
        self.predictions - average attractor values for all fixes   :PredictType = dict[FixesType, dict]
            key   - fixes                                           :FixesType = tuple[tuple[str, int]]
                    ((node A, value1), (node B, value2), ...)
            value - average value of a node in the attractors       :dict[str, float]
                    {observed_node: predict_value}

        """
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
                    if node not in i.keys(): # type: ignore
                        i[node] = "?" # type: ignore
                    else:
                        i[node] = str(i[node]) # type: ignore

            result = {}
            for i in tr:
                for node in self.primes.keys():
                    if node not in result.keys():
                        result[node] = 0.0
                    if i[node] == "1": # type: ignore
                        result[node] += (1.0/len(tr))
                    elif i[node] == "?": # type: ignore
                        result[node] += (0.5/len(tr))

            predictions[fixes] = result

        self.predictions = predictions

    def get_model_score(self, exps:list[ExpType], report:bool=False, file:str="score_report.tsv"):
        """
        Assigns self.score when given experiments.
        Requires self.predictions to be calculated beforehand.

        Can be modified to meet the desired criteria.
        
        Assigns
        -------
        self.max_score : float
            max possible score of the model

        self.non_hierarchy_score : float
            how well the model agrees with experimental results, ignoring hierarchy
        
        self.score : float
            how well the model agrees with experimental results
            one point in score means agreement to one perturbation
        
        """
        self.max_score = 0.0
        for exp in exps:
            self.max_score += exp[1]
        agreements, self.non_hierarchy_score = score.get_agreement(exps, self.predictions)
        self.score = score.get_hierarchy_score(agreements, self.default_sources, report=report, file=file)

    def mutate(self, probability:float, edge_prob:float, bias:float=0.5, seed:int|None=None) -> Model:
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
        config.id += 1
        mutated_model = Model()
        mutated_model.id = config.id
        mutated_model.generation = self.generation + 1

        mutated_model.base = self.base
        mutated_model.constraints = self.constraints
        mutated_model.edge_pool = self.edge_pool
        mutated_model.default_sources = self.default_sources
        mutated_model.max_score = self.max_score
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
            
            # get complexity
            for prime_implicant in mutated_model.primes[node][1]:
                mutated_model.complexity += len(prime_implicant)        

        return mutated_model

    def info(self):
        """
        prints out a brief summary of the model info
        """
        # TODO: print out total score as well
        print("id: ", self.id)
        print("generation: ", self.generation)
        print("extra edges: ", self.extra_edges)
        print(f"score: {round(self.score,2)} / {self.max_score} ({round(self.score/self.max_score*100,1)}%)")
        print("following constraints:", self.check_constraint())
        print("complexity:", self.complexity)

    def export(self, file_name:str|None=None, threshold:float=0.0):
        """
        Exports the model rules with scores above a certain threshold.

        Parameters
        ----------
        file_name : str
            location of the output file
            if None, output file is in the form "(model's name)_id_gen.txt"
        threshold : float
            only models with score higher than the threshold get exported

        """
        if threshold != 0.0 and self.score < threshold:
            return
        if file_name == None:
            file_name = self.name + "_" + str(self.id) + "_gen" + str(self.generation) + ".bnet"

        fp = open(file_name, "w")
        fp.write("# id: " + str(self.id) + "\n")
        fp.write("# generation: " + str(self.generation) + "\n")
        fp.write("# extra edges: " + str(self.extra_edges) + "\n")
        fp.write("# score: " + str(self.score) + " / " + str(self.max_score) + "\n")
        fp.write("# following constraints: " + str(self.check_constraint()) + "\n")
        fp.write("# complexity: " + str(self.complexity) + "\n\n")
        fp.write("targets,\tfactors\n")
        primes = {k:self.primes[k] for k in sorted(self.primes)}
        for k in primes:
            s = conv.prime2bnet(k, primes[k])
            fp.write(s + "\n")
        fp.close()
        
        print("Exported generated model to", os.path.abspath(file_name))


def mix_models(model1:Model, model2:Model) -> Model:
    """
    For each node, take the rule from one of the parent model randomly.

    Parameters
    ----------
    model1, model2 - parent models                              :Model class
    
    Returns
    -------
    mixed_model - model that takes functions from the parents   :Model class
    """
    config.id += 1
    mixed_model = Model()
    mixed_model.id = config.id
    mixed_model.generation = max(model1.generation,model2.generation) + 1

    mixed_model.base = model1.base
    mixed_model.constraints = model1.constraints
    mixed_model.edge_pool = model1.edge_pool
    mixed_model.default_sources = model1.default_sources
    mixed_model.max_score = model1.max_score
    mixed_model.name = model1.name

    mixed_model.primes = {}
    mixed_model.regulators_dict = {}
    mixed_model.signs_dict = {}    
    mixed_model.rr_dict = {}
    mixed_model.extra_edges = []

    mixed_model.complexity = 0

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

        # get complexity
        for prime_implicant in mixed_model.primes[node][1]:
            mixed_model.complexity += len(prime_implicant)

    return mixed_model