import random
import math
import itertools as it
import csv
from pyboolnet.external.bnet2primes import bnet_text2primes
import pyboolnet.trap_spaces
from experiment import *
from score import *
from constraint import *
import config


def import_exps(location):
    """
    Returns exps

    Parameters
    ----------
    location : data location

    Returns
    -------
    exps : dictionary of list of tuples
        key represents the perturbation
        value is a list where outcomes are stored
        each outcome is a tuple (measured_node, 1, 0, 0)
        1,0,0 denotes constitutive activation
    """

    file = open(location, "r")
    data = csv.reader(file, delimiter='\t')

    lst = []
    for row in data:
        exp = []
        # get the fix
        fix = []
        row_iter = iter(row)
        for sth in row_iter:
            fix1 = tuple([sth, next(row_iter)])
            fix.append(fix1)
        fix = tuple(sorted(fix, key= lambda x:x[0]))
        # get the result
        result = tuple(next(data))

        exp.append(fix)
        exp.append(result)
        lst.append(exp)

    lst = sorted(lst, key = lambda x: x[0])

    exps = {}
    for exp in lst:
        if exp[0] not in exps:
            exps[exp[0]] = []
        exps[exp[0]].append(exp[1])

    return exps

def rr2bnet(regulators, rr, signs):
    """
    Returns rules when given the regulating nodes, the rule representation
    and the node signs.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        rule representation
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node

    Returns
    -------
    bnet : str
        bnet formatted rules

    """
    nodes = list(regulators)
    N = len(nodes)

    assert 2**N == len(rr), "The length of rr should be equal to 2^(number of nodes)"
    assert N == len(signs), "signs should be a list with the number of nodes as length"

    for i, num in enumerate(signs):
        if num == '0':
            nodes[i] = '!' + nodes[i]

    # start constructing bnet = (0 and A and B and C) or (1 and A and B) or ...
    bnet = '('
    for i, num in enumerate(rr):
        bnet += str(num)
        # bin = 111, 110, 101, ...
        bi = format(2**N-i-1, '0' + str(N) + 'b')
        for j, node in enumerate(nodes):
            if bi[j] == '1':
                bnet += ' & ' + node
        bnet += ') | ('
    bnet = bnet.removesuffix(' | (')

    return bnet

def prime2rr(prime, regulators = None, signs = None):
    """
    Returns the representation of the rule of a node
    when given the prime implicants of that rule.

    Parameters
    ----------
    prime : list of list of dictionaries
        prime implicants of the rule
    regulators : tuple of strings
        the regulating nodes
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node

    Returns
    -------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        rule representation
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    """
    # Get the tuple of the regulator nodes
    if regulators == None:
        nodes = set()
        for dct in prime[1]:
            nodes = nodes.union(set(dct.keys()))
        nodes = list(nodes)
        nodes.sort()
        regulators = tuple(nodes)

    # check if the node is a fixed node
    if len(regulators) == 0:
        signs = ''
        if bool(prime[1]): # 1 implicant is not empty
            rr = '1'
        else:
            rr = '0'
        return regulators, rr, signs

    # Get the rr and signs of the regulators
    rr = ['0'] * (2**len(regulators))
    if signs == None:
        signs = ['1'] * len(regulators)
        for implicant in prime[1]:
            # check whether certain node is inside
            for i, node in enumerate(regulators):
                if node in implicant:
                    if implicant[node] == 0:
                        signs[i] = '0'
        signs = ''.join(signs)
    # Get a binary number that gives us the position of the implicant on the rr
    # for every piece of rule that gives 1 to the regulated node,
    for implicant in prime[1]:
        bi = ''
        # check whether certain node is inside
        for i, node in enumerate(regulators):
            if node in implicant:
                # if it exists, put 1
                bi += '1'
            else:
                # if it does not exist, put 0
                bi += '0'
        # put 1 on that position of the rr
        rr[-int(bi, 2)-1] = '1'

    rr = ''.join(rr)

    return regulators, rr, signs

def get_uni_rr(rr, max = True):
    """
    Returns the unique representation of the rule of a node(max or min)
    when given any representation of that rule.

    Parameters
    ----------
    rr : length 2^N binary str
        represents the rules
    max : Boole
        if True, get maximal representation
        if False, get minimal representation

    Returns
    -------
    uni_rr : length 2^N binary str
        unique representation of the rules
        if maximal, it is equivalent to the truth table
        if minimal, it is in a Blake canonical form

    """
    # N = number of regulators
    n = len(rr)
    N = int(math.log2(n))

    # already minimal if the node is a fixed node
    if N == 0:
        return rr

    uni_rr = list(rr)

    modified = []

    # iterate through the string in reverse
    for i in range(n):
        # get the regulator values if there is '1'
        if uni_rr[-i-1] == '1':
            # bi represents the regulator values in binary string
            bi = format(i, '0' + str(N) + 'b')
        else:
            continue
        # find all the positions that need to be changed to 0
        if bi in modified:
            continue

        lst = []
        for num in bi:
            if num == '0':
                lst.append(['0', '1'])
            else:
                lst.append(['1'])
        positions = [''.join(p) for p in it.product(*lst)]
        # do not change the original position
        positions.remove(bi)
        modified.extend(positions)
        # change to '0' if min or '1' if max in the gained positions
        for position in positions:
            j = n - int(position,2) - 1
            uni_rr[j] = str(int(max))

    uni_rr = ''.join(uni_rr)

    return uni_rr

def get_max_irr(rr):
    """
    Returns the inveted maximal representation of the rule of a node
    when given any representation of that rule.

    Parameters
    ----------
    rr : length 2^N binary str
        represents the rule

    Returns
    -------
    max_irr : length 2^N binary str
        inverted maximal representation of the rule

    """
    max_rr = get_uni_rr(rr, max = True)
    max_rr = max_rr.replace('0', 'X')
    max_rr = max_rr.replace('1', '0')
    max_rr = max_rr.replace('X', '1')
    max_irr = max_rr[::-1]

    return(max_irr)

def rr2prime(regulators, rr, signs, inverted = False):
    """
    Returns the prime implicants of the rule of a node
    when given any representation of that rule.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    inverted : Boolean
        If True, get the prime assuming the input rr is inverted.
        Default is False.

    Returns
    -------
    prime : list of list of dictionaries
        prime implicants of the rule

    """
    N = len(regulators)
    assert 2**N == len(rr), "The length of rr should be equal to 2^(number of nodes)"
    assert N == len(signs), "signs should be a list with the numbe of nodes as length"

    inverted = int(inverted)
    min_rr = get_uni_rr(rr, max = False)
    max_irr = get_max_irr(rr)
    min_irr = get_uni_rr(max_irr, max = False)

    prime = [[],[]]
    for i in range(len(rr)):
        # get the regulator values if there is '1'
        if min_rr[i] == '1':
            # bi represents the regulator values in binary string
            bi = format(2**N - i - 1, '0' + str(N) + 'b')
        else:
            continue
        implicant = {}
        for i, num in enumerate(bi):
            if num == '1':
                implicant[regulators[i]] = (int(signs[i]) - inverted)%2
        prime[1-inverted].append(implicant)

    for i in range(len(rr)):
        # get the regulator values if there is '1'
        if min_irr[i] == '1':
            # bi represents the regulator values in binary string
            bi = format(2**N - i - 1, '0' + str(N) + 'b')
        else:
            continue
        implicant = {}
        for i, num in enumerate(bi):
            if num == '1':
                implicant[regulators[i]] = (1-int(signs[i]) - inverted)%2
        prime[0-inverted].append(implicant)

    return prime


def mutate_rr(rr, probability):
    '''
    Returns the mutated representation of that rule when given any representation of a rule.
    This function is biased to activate all the implicants.
    Use mutate_rr_bias for non-biased or specific biased mutations.

    Parameters
    ----------
    rr : length 2^N binary str
        representation of the rules
    probability : float between 0 and 1
        probability that each of the binary number in the representation is mutated.

    Returns
    -------
    mutated_rr : length 2^N binary str
        representation of the rules
    '''
    lst = []
    for num in rr:
        lst.append(int(num))
    for i in range(len(lst)):
        rnd = random.random()
        if rnd < probability:
            lst[i] += 1
    mutated_rr = ''
    for num in lst:
        mutated_rr += str(num % 2)
    return mutated_rr

def mutate_rr_bias(rr, probability, bias = 0.5):
    """
    Returns the mutated representation of that rule with certain acitivity bias
    when given any representation of a rule.

    Parameters
    ----------
    rr : length 2^N binary str
        representation of the rules
    probability : float between 0 and 1
        probability that each of the binary number in the representation is mutated.
    bias : float between 0 and 1
        bias to deactivate implicants

    Returns
    -------
    mutated_rr : length 2^N binary str
        representation of the rules
    """
    rnd = random.random()
    if rnd < bias:
        irr = get_max_irr(rr)
        irr = mutate_rr(irr, probability)
        mutated_rr = get_max_irr(irr)
    else:
        mutated_rr = mutate_rr(rr, probability)

    return mutated_rr

def mutate_rr_constraint(regulators, rr, constraints, node, probability, bias = 0.5):
    """
    Returns the mutated representation of that rule with certain acitivity bias
    that follows all the constraints when given any representation of a rule.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    constraints : dictionary of dictionary of tuples
        represents 4 types of constraints
        fixed
        regulate
        necessary
        group
    node : string
        the target node
    probability : float between 0 and 1
        probability that each of the binary number in the representation is mutated.
    bias : float between 0 and 1
        bias to deactivate implicants

    Returns
    -------
    mutated_rr : length 2^N binary str
        representation of the rules
    modified : Boole
        True if modified
    """
    # no need to mutate fixed nodes
    if len(regulators) == 0:
        return rr, False
    # no need to mutate source nodes
    if len(regulators) == 1 and regulators[0] == node:
        return rr, False
    # impose constraints
    # no need to mutate nodes whose rules are fixed
    if node in constraints['fixed']:
        return rr, False
    # no need to mutate nodes with one regulator and cannot be a source node
    if len(regulators) == 1 and node not in constraints['possible_source']:
        return rr, False

    # mutation
    # impose group nodes
    if node in constraints['group']:
        groups = constraints['group'][node]
        group_rr, group_regulators = rr2group_rr(regulators, rr, groups)
        mutated_group_rr = mutate_rr_bias(group_rr, probability, bias = 0.5)
        mutated_rr = group_rr2rr(regulators, mutated_group_rr, groups, group_regulators)
    else: # no group constraints
        mutated_rr = mutate_rr_bias(rr, probability, bias = 0.5)

    # impose constraints
    # impose necessary nodes
    if node in constraints['necessary']:
        for necc in constraints['necessary'][node]:
            mutated_rr = impose_necessary(regulators, mutated_rr, necc)

    # impose regulating nodes
    if node in constraints['regulate']:
        # check if the constrained nodes are regulating the target
        check = True
        for reg in constraints['regulate'][node]:
            check = check and check_regulation(regulators, mutated_rr, reg)
            if check == False:
                break
        while check == False:
            if node in constraints['group']:
                groups = constraints['group'][node]
                group_rr, group_regulators = rr2group_rr(regulators, rr, groups)
                mutated_group_rr = mutate_rr_bias(group_rr, probability, bias = 0.5)
                mutated_rr = group_rr2rr(regulators, mutated_group_rr, groups, group_regulators)
            else: # no group constraints
                mutated_rr = mutate_rr_bias(rr, probability, bias = 0.5)
            check = True
            for reg in constraints['regulate'][node]:
                check = check and check_regulation(regulators, mutated_rr, reg)
                if check == False:
                    break

    # need to check if a node became a source node and mutate more if it did
    # nodes that have a regulating node or a necessary node cannot be a source
    if node not in constraints['regulate']:
        # check if the constrained nodes are regulating the target
        check = check_source(mutated_rr)
        while check == True:
            if node in constraints['group']:
                groups = constraints['group'][node]
                group_rr, group_regulators = rr2group_rr(regulators, rr, groups)
                mutated_group_rr = mutate_rr_bias(group_rr, probability, bias = 0.5)
                mutated_rr = group_rr2rr(regulators, mutated_group_rr, groups, group_regulators)
            else: # no group constraints
                mutated_rr = mutate_rr_bias(rr, probability, bias = 0.5)

            # impose necessary nodes again
            if node in constraints['necessary']:
                for necc in constraints['necessary'][node]:
                    mutated_rr = impose_necessary(regulators, mutated_rr, necc)

            check = check_source(mutated_rr)

    max_original = get_uni_rr(rr)
    max_mutated = get_uni_rr(mutated_rr)
    if max_original == max_mutated:
        modified = False
    else:
        modified = True

    return mutated_rr, modified

def add_regulator(regulators, rr, signs, new_regulator, new_sign, bias=0.5):
    """
    Returns a new representation of a rule with an extra regulator.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    new_regulator : str
        the added regulating node
    new_sign : length 1 binary str
        represents sign of the new regulator. 0 means negation of the node

    Returns
    -------
    added_regulators : tuple of strings
        the regulating nodes
    added_rr : length 2^(N+1) binary str
        representation of the rules
    added_signs : length N+1 binary str
        represents signs of regulators. 0 means negation of the node

    """
    nodes = list(regulators)
    nodes.append(new_regulator)
    added_regulators = tuple(nodes)

    added_signs = signs + new_sign

    N = len(regulators)

    rnd = random.random()
    if rnd < bias:
        rr = get_max_irr(rr)

    added_rr = ''
    for bi in rr:
        added_rr += str(random.randint(0,1))
        added_rr += bi

    if rnd < bias:
        added_rr = get_max_irr(added_rr)

    return added_regulators, added_rr, added_signs

def delete_regulator(regulators, rr, signs, target_regulator, bias=0.5):
    """
    Returns a new representation of a rule with the target regulator deleted.

    Parameters
    ----------
    regulators : tuple of strings
        the regulating nodes
    rr : length 2^N binary str
        representation of the rules
    signs : length N binary str
        represents signs of regulators. 0 means negation of the node
    target_regulator : str
        the regulatort to be deleted

    Returns
    -------
    deleted_regulators : tuple of strings
        the regulating nodes
    deleted_rr : length 2^(N-1) binary str
        representation of the rules
    deleted_signs : length N-1 binary str
        represents signs of regulators. 0 means negation of the node

    """
    N = len(regulators)
    n = regulators.index(target_regulator)

    nodes = list(regulators)
    del nodes[n]
    deleted_regulators = tuple(nodes)

    deleted_signs = signs[:n] + signs[n+1:]

    deleted_rr = ''

    rnd = random.random()
    if rnd < bias:
        rr = get_max_irr(rr)

    bi = list(rr)
    for i in range(2**N):
        if i % 2**(N-n) > 2**(N-1-n)-1:
            deleted_rr += bi[i]

    if rnd < bias:
        deleted_rr = get_max_irr(deleted_rr)

    return deleted_regulators, deleted_rr, deleted_signs

class Network:
    def __init__(self):
        self.id = None
        self.generation = None
        self.rules = None
        self.primes = None
        self.regulators = None
        self.rrs = None
        self.signs = None
        self.extra_edges = None
        self.predictions = None
        self.score = None

    @classmethod
    def import_network(cls, primes, id = 0, generation = 0, extra_edges = 0, given_regulators = None, given_signs = None):
        x = cls()
        x.id = id
        x.generation = generation
        x.primes = primes
        x.regulators = {}
        x.rrs = {}
        x.signs = {}
        x.extra_edges = extra_edges

        for node in primes:
            if given_regulators == None:
                regulators, rr, signs = prime2rr(primes[node], regulators = None, signs = None)
            else:
                regulators, rr, signs = prime2rr(primes[node], regulators = given_regulators[node], signs = given_signs[node])
            x.regulators[node] = regulators
            x.rrs[node] = rr
            x.signs[node] = signs

        return x

    def mutate(self, constraints, edge_pool, probability, edge_prob, bias = 0.5):
        config.id += 1
        mutated_network = Network()
        mutated_network.id = config.id
        mutated_network.generation = self.generation + 1
        mutated_network.regulators = self.regulators.copy()
        mutated_network.signs = self.signs.copy()
        mutated_network.rrs = {}
        mutated_network.extra_edges = self.extra_edges
        mutated_network.primes = {}

        for node in self.rrs:
            # get mutated_rr from rr
            regulators = self.regulators[node]
            rr = self.rrs[node]
            signs = self.signs[node]
            mutated_rr, modified = mutate_rr_constraint(regulators, rr, constraints, node, probability)
            mutated_network.rrs[node] = mutated_rr

            # get primes from the mutated_rr
            # if the representations are equivalent, take the old prime
            if not modified:
                # print(node, 'is not modified')
                mutated_network.primes[node] = self.primes[node]
            else:
                prime1 = rr2prime(regulators, mutated_rr, signs, inverted = False)
                mutated_network.primes[node] = prime1
                # irr = get_max_irr(rr)
                # prime2 = rr2prime(regulators, irr, signs, inverted = True)
                # assert prime1 == prime2, "rr and irr lead to different result!"

            # # old version using rr2bnet to get primes from the mutated_rr
            # else:
            #     rule = mutated_network.rules[node]
            #     prime = list(bnet_text2primes(rule).values())[0]
            #     mutated_network.primes[node] = prime

        add_edge = False
        rnd = random.random()
        if rnd < edge_prob:
            new_edge = random.choice(edge_pool)
            node = new_edge[1]
            regulators = mutated_network.regulators[node]
            new_regulator = new_edge[0]
            add_edge = True

        if add_edge and new_regulator in regulators:
            add_edge = False
            rr = mutated_network.rrs[node]
            signs = mutated_network.signs[node]

            mutated_network.extra_edges -= 1
            deleted_regulators, deleted_rr, deleted_signs = delete_regulator(regulators, rr, signs, new_regulator)

            mutated_network.regulators[node] = deleted_regulators
            mutated_network.rrs[node] = deleted_rr
            mutated_network.signs[node] = deleted_signs
            prime1 = rr2prime(deleted_regulators, deleted_rr, deleted_signs, inverted = False)
            mutated_network.primes[node] = prime1

        if add_edge and new_regulator not in regulators:
            rr = mutated_network.rrs[node]
            signs = mutated_network.signs[node]
            new_sign = new_edge[2]

            mutated_network.extra_edges += 1
            added_regulators, added_rr, added_signs = add_regulator(regulators, rr, signs, new_regulator, new_sign)

            mutated_network.regulators[node] = added_regulators
            mutated_network.rrs[node] = added_rr
            mutated_network.signs[node] = added_signs
            prime1 = rr2prime(added_regulators, added_rr, added_signs, inverted = False)
            mutated_network.primes[node] = prime1

        return mutated_network

    def get_predictions(self, exps):
        '''
        Returns predictions when given interventions

        Parameters
        ----------
        exps : dict or list
            interventions in the form [((nodeA, value1),(nodeB, value2), ...), ...]

        Returns
        -------
        self.predictions : dict
            keys : interventions
            values : dict
                average value of every node in the attractors
        '''
        predictions = {}
        for exp in exps:
            perturbation = {}
            for fix in exp:
                perturbation[fix[0]] = fix[1]
            # print("- - - - - - - - - -")
            # print("fixed: ", perturbation)

            new_primes = self.primes.copy()
            for node in perturbation.keys():
                assert node in new_primes.keys(), "perturbed node should be in the network"
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

            predictions[exp] = result

        self.predictions = predictions

    def get_network_score(self, exps):
        '''
        To be modified to meet the criteria
        '''
        self.score = get_score(exps, self.predictions, self.extra_edges)

    def export(self, name, threshold = 0.0):
        '''
        Exports the network rules with scores above a certain threshold.

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
        # print(self.rules)
        # print(self.primes)
        # print(self.regulators)
        # print(self.rrs)
        # print(self.signs)
        print('extra edges: ', self.extra_edges)
        print('score: ', self.score)
