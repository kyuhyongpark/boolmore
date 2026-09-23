import os
import pickle

import boolmore.boolean_functions as bf

PrimeType = list[list[dict[str, int]]]
FixesType = tuple[tuple[str, int]]
ExpType = tuple[int, float, FixesType, str, str]
PredictType = dict[FixesType, dict]


class Model():
    def __init__(self):
        """
        Attributes
        ----------        
        base            - the base model (not the neccesarily the starting model)   :Model class
                          from which the regulators, fixed functions, constants,
                          extra edges, etc. are decided.
        edge_pool       - the pool of edges. 0 is negative, 1 is positive           :list[list[str]]
                          [[regulator, target, sign], ...]

        primes          - pyboolnet primes dictionary                               :length N dict[str, PrimeType]
                          {node: prime}                          

        regulators_dict - dictionary of the regulating nodes                        :length N dict[str, tuple[str]]
        signs_dict      - dictionary of the signs of regulators                     :length N dict[str, str]
        rr_dict         - dictionary of the binary rule representations             :length N dict[str, str]
        
        extra_edges     - edges from the pool that are present in the model         :list[list[str]]
                          [[regulator, target, sign], ...]
        n_extra_edges   - number of extra edges in the model                        :int

        """

        self.base = None
        self.edge_pool = []
        
        self.primes:dict[str, PrimeType] = {}

        self.regulators_dict = {}
        self.signs_dict = {}
        self.rr_dict = {}

        self.extra_edges = []
        self.n_extra_edges = 0

    @classmethod
    def import_model(cls, primes:dict[str, PrimeType],
                     base:Model|None=None, edge_pool:list[list[str]]=[],
        ) -> Model:
        """
        Import a model.
        If base=None, the output model is considered the base model,
        and the given primes is used to construct it.

        Parameters
        ----------

        primes          - pyboolnet primes dictionary                               :length N dict[str, PrimeType]
                          {node: prime}
                          
        base            - the base model (not the neccesarily the starting model)   :Model class
                          from which the regulators, fixed functions, constants,
                          extra edges, etc. are decided.
                          if None, the output model is considered the base

        # if base is given, below parameters take the value of the base
        edge_pool       - the pool of edges. 0 is negative, 1 is positive           :list[list[str]]
                          [[regulator, target, sign], ...]

        Returns
        -------
        model - the imported model :Model class

        """
        x = cls()

        x.primes = primes

        # get edge pool
        if base == None:
            x.edge_pool.extend(edge_pool)

        else:
            x.base = base
            x.edge_pool.extend(base.edge_pool)
        
        for node in x.primes:
            # find current regulators and signs
            regulators, rr, signs = bf.prime2rr(x.primes[node])

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

                rr = bf.prime2rr(primes[node], regulators=regulators, signs=signs)[1] # type: ignore
                x.regulators_dict[node] = regulators
                x.rr_dict[node] = rr
                x.signs_dict[node] = signs

        x.n_extra_edges = len(x.extra_edges)

        if base == None:
            x.base = x

        return x


    def info(self):
        """
        prints out a brief summary of the model info
        """
        print("extra edges: ", self.extra_edges)
        print("number of extra edges: ", self.n_extra_edges)

    def export(self, file_name:str, details:bool=True):
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
        bnet_file_name = file_name + ".bnet"
        pkl_file_name = file_name + ".pkl"


        # write bnet file
        fp = open(bnet_file_name, "w")

        if details:
            fp.write("# extra edges: " + str(self.extra_edges) + "\n")
            fp.write("# number of extra edges: " + str(self.n_extra_edges) + "\n")
        fp.write("targets,\tfactors\n")
        primes = {k:self.primes[k] for k in sorted(self.primes)}
        for k in primes:
            s = bf.prime2bnet(k, primes[k])
            fp.write(s + "\n")
        fp.close()
        
        # also pickle primes
        with open(pkl_file_name, "wb") as f:
            pickle.dump(self.primes, f)

        print("Exported generated model to", os.path.abspath(bnet_file_name))
        print("Pickled primes to", os.path.abspath(pkl_file_name))