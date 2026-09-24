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
        base : Model or None
            The base model (not necessarily the starting model) from which the
            regulators, fixed functions, constants, and other model properties
            are determined.

        edge_pool : list[tuple[str, str, str]]
            Pool of candidate edges. Each edge is represented as
            ``(regulator, target, sign)``, where sign is ``"0"`` for negative
            and ``"1"`` for positive.

        primes : dict[str, PrimeType]
            PyBoolNet primes dictionary mapping node names to prime implicants.

        regulators_dict : dict[str, tuple[str]]
            Dictionary mapping each node to its regulating nodes.

        signs_dict : dict[str, str]
            Dictionary mapping each node to the signs of its regulators.

        rr_dict : dict[str, str]
            Dictionary mapping each node to its binary rule representation.

        edges : list[dict[str, str | bool]]
            List of edges currently present in the model. Each edge contains
            ``regulator``, ``target``, ``sign``, ``source``, and ``effective``.
        """

        self.base = None
        self.edge_pool = []
        
        self.primes:dict[str, PrimeType] = {}

        self.regulators_dict = {}
        self.signs_dict = {}
        self.rr_dict = {}

        self.edges = []

    @classmethod
    def from_primes(
        cls,
        primes: dict[str, PrimeType],
        base: Model | None = None,
        edge_pool: list[tuple[str, str, str]] | None = None,
    ) -> Model:
        """
        Import a model.
        If base=None, the output model is considered the base model,
        and the given primes is used to construct it.

        Parameters
        ----------
        primes : dict[str, PrimeType]
            PyBoolNet primes dictionary mapping node names to prime implicants.

        base : Model or None, default=None
            Base model from which regulators, fixed functions, constants, extra
            edges, etc. are determined. If None, the output model is considered
            the base model.

        edge_pool : list[tuple[str, str, str]] or None, default=None
            Pool of candidate edges. Each edge is represented as
            ``[regulator, target, sign]``, where sign is "0" for negative and
            "1" for positive. If `base` is given, the edge pool is taken from
            `base` instead.

        Returns
        -------
        Model
            The imported model.
        """
        x = cls()

        x.primes = primes

        if edge_pool is None:
            edge_pool = []

        # primes is the base model
        if base is None:
            x.edge_pool = list(edge_pool)
            for node in x.primes:
                regulators, rr, signs = bf.prime2rr(x.primes[node])

                x.regulators_dict[node] = regulators
                x.rr_dict[node] = rr
                x.signs_dict[node] = signs

            x.base = x
            x.validate_edge_pool()
            x.edges = x._construct_edges()

            return x

        # primes is not the base model
        if edge_pool != []:
            raise ValueError("edge_pool must be empty if base is given.")
        x.base = base
        x.edge_pool = base.edge_pool
        x.validate_edge_pool()
        x.validate_primes()
        x.edges = x._construct_edges()

        for node in x.primes:
            extra_edges = x.get_edges(target=node, source="edge_pool")

            base_regulators = list(base.regulators_dict[node])
            base_signs = base.signs_dict[node]

            regulators = base_regulators.copy()
            signs = base_signs
            for edge in extra_edges:
                regulators.append(edge["regulator"])
                signs += edge["sign"]

            regulators = tuple(regulators)

            rr = bf.prime2rr(primes[node], regulators=regulators, signs=signs)[1]
            x.regulators_dict[node] = regulators
            x.rr_dict[node] = rr
            x.signs_dict[node] = signs

        return x

    def validate_edge_pool(self) -> None:
        """Validate the edge pool."""
        pairs = set()

        for edge in self.edge_pool:
            if (
                not isinstance(edge, tuple)
                or len(edge) != 3
                or not all(isinstance(value, str) for value in edge)
                or edge[2] not in {"0", "1"}
            ):
                raise ValueError(
                    f"Invalid edge: {edge}. "
                    "Each edge must be a tuple of three strings, "
                    "with the sign equal to '0' or '1'."
                )

            pair = edge[:2]

            if pair in pairs:
                raise ValueError(
                    f"Duplicate regulator-target pair: {pair}"
                )

            pairs.add(pair)

            if edge[0] in self.base.regulators_dict[edge[1]]:
                raise ValueError(
                    f"Edge {edge} already exists in the base model."
                )

    def validate_primes(self) -> None:
        """Validate regulators and signs in the model's primes."""
        for node, primes in self.primes.items():
            regulators, _, signs = bf.prime2rr(primes)

            base_regulators = self.base.regulators_dict[node]
            base_signs = self.base.signs_dict[node]

            allowed_regulators = set(base_regulators)
            allowed_regulators.update(
                edge[0]
                for edge in self.edge_pool
                if edge[1] == node
            )

            unexpected_regulators = set(regulators) - allowed_regulators

            if unexpected_regulators:
                raise ValueError(
                    f"Unexpected regulators for node {node}: "
                    f"{unexpected_regulators}"
                )

            for position, regulator in enumerate(regulators):
                if regulator in base_regulators:
                    base_position = base_regulators.index(regulator)

                    if signs[position] != base_signs[base_position]:
                        raise ValueError(
                            f"Sign mismatch for {regulator} -> {node}"
                        )
                else:
                    for edge in self.edge_pool:
                        if edge[0] == regulator and edge[1] == node:
                            if signs[position] != edge[2]:
                                raise ValueError(
                                    f"Sign mismatch for {regulator} -> {node}"
                                )

    def _construct_edges(self):
        edges = []

        for target in self.primes:

            _regulators, _, _signs = bf.prime2rr(self.primes[target])
            base_regulators, _, base_signs = bf.prime2rr(self.base.primes[target])
            
            for _regulator, _sign in zip(_regulators, _signs):
                # Is this edge one of the candidate edge-pool edges?
                if (_regulator, target, _sign) in self.edge_pool:
                    source = "edge_pool"
                else:
                    source = "base"

                # Determine whether this edge is effective
                effective = False
                for implicant in self.primes[target][1]:
                    if _regulator in implicant:
                        effective = True
                        break

                edges.append({
                    "regulator": _regulator,
                    "target": target,
                    "sign": _sign,
                    "source": source,
                    "effective": effective
                })

            for base_regulator in base_regulators:
                if base_regulator not in _regulators:
                    edges.append({
                        "regulator": base_regulator,
                        "target": target,
                        "sign": base_signs[base_regulators.index(base_regulator)],
                        "source": "base",
                        "effective": False
                    })

        edges.sort(
            key=lambda edge: (
                edge["regulator"],
                edge["target"],
                edge["sign"],
            )
        )

        return edges

    def get_edges(
        self,
        regulator: str | None = None,
        target: str | None = None,
        sign: str | None = None,
        source: str | None = None,
        effective: bool | None = None,
    ):
        edges = self.edges

        if regulator is not None:
            edges = [edge for edge in edges if edge["regulator"] == regulator]

        if target is not None:
            edges = [edge for edge in edges if edge["target"] == target]

        if sign is not None:
            edges = [edge for edge in edges if edge["sign"] == sign]

        if source is not None:
            edges = [edge for edge in edges if edge["source"] == source]

        if effective is not None:
            edges = [edge for edge in edges if edge["effective"] == effective]

        return edges

    @property
    def unadded_edges(self):
        added_edges = {
            (edge["regulator"], edge["target"], edge["sign"])
            for edge in self.get_edges(source="edge_pool")
        }

        return [
            edge for edge in self.edge_pool
            if tuple(edge) not in added_edges
        ]

    @property
    def nodes(self):
        return list(self.primes.keys())

    def info(self):
        """
        Return a brief summary of the model information.
        """
        lines = [
            f"# number of nodes: {len(self.primes)}",
            f"# number of edges: {len(self.edges)}",
            f"# number of effective base edges: "
            f"{len(self.get_edges(source='base', effective=True))}",
            f"# number of ineffective base edges: "
            f"{len(self.get_edges(source='base', effective=False))}",
            f"# extra edges: {self.get_edges(source='edge_pool')}",
            f"# number of extra edges: "
            f"{len(self.get_edges(source='edge_pool'))}",
            f"# number of effective extra edges: "
            f"{len(self.get_edges(source='edge_pool', effective=True))}",
            f"# number of ineffective extra edges: "
            f"{len(self.get_edges(source='edge_pool', effective=False))}",
            f"# number of unadded edges: "
            f"{len(self.unadded_edges)}",
        ]

        return "\n".join(lines)