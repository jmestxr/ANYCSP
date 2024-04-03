from src.csp.csp_data import CSP_Data
from src.data.dataset import nx_to_col, nx_to_maxcut
from src.utils.rb_utils import get_random_RB

import numpy as np
import networkx as nx
import torch
import cnfgen

import random

class ResAlloc_Generator:
    def __init__(self, n1, n2, n3, n4, num_phys):
        self.n1 = n1 # number of days
        self.n2 = n2 # number of periods
        self.n3 = n3 # number of health units
        self.n4 = n4 # number of vacancies (physicians needed in one period)
        self.num_phys = num_phys # number of physicians available for allocation

        self.num_var = self.n1 * self.n2 * self.n3 * self.n4 # number of units Mi


    def generate_unary_constraints(self, csp_data):
        # csp_data.add_uniform_constraint_data(True, torch.tensor([[0], [0]]), torch.tensor([[[5]], [[10]]]), torch.tensor([1, 2]))\
        # [[[5],[10]]]

        min_weight = 1
        max_weight = 3

        # Parameters:
        threshold_pct = 0.2 # Set minimum % of variables having unary constraints to 20%
        num_cst = int(0.1 * self.num_phys) # number of unary constraints per variable
        num_vars_with_unary_constraints = np.random.randint(int(threshold_pct * self.num_var), self.num_var + 1)
        # num_vars_with_unary_constraints = int(threshold_pct * self.num_var)

        random_vars_with_unary_constraints = random.sample(range(self.num_var), num_vars_with_unary_constraints)
        var_idx = torch.cat([torch.full((num_cst,), i) for i in random_vars_with_unary_constraints])

        val_idx = []
        for _ in range(num_vars_with_unary_constraints): 
            val_idx.append(random.sample(range(self.num_phys), num_cst))

        val_idx = torch.tensor(val_idx).reshape(num_vars_with_unary_constraints, num_cst, 1)
        # val_idx = torch.arange(self.num_phys).tile(num_cst).reshape(num_cst, self.num_phys, 1)

        cst_weights = torch.randint(low=min_weight, high=max_weight+1, size=(num_vars_with_unary_constraints,))

        csp_data.add_uniform_constraint_data(True, var_idx, val_idx, cst_weights)


        # for var_idx in var_idxs:
        #     var_idx = torch.full((num_cst, 1), var_idx)

        #     val_idx = list(range(self.num_phys))
        #     random.shuffle(val_idx)
        #     val_idx = torch.tensor(val_idx[:num_cst]).reshape((num_cst, 1, 1))
        #     # val_idx = torch.randint(low=0, high=self.num_phys, size=(num_cst,)).reshape((num_cst, 1, 1))
        #     cst_weights = torch.randint(low=min_weight, high=max_weight+1, size=(num_cst,))

        #     csp_data.add_uniform_constraint_data(True, var_idx, val_idx, cst_weights)


    def generate_multiple_constraints(self, csp_data):
        # Add 100 AllDiff binary constraints on random units
        num_cst = 100
        var_idx = []

        while len(var_idx) < num_cst:
            new_pair = torch.randint(0, self.num_var, size=(2,))
            if all((new_pair != row).any() for row in var_idx):
                var_idx.append(new_pair)

        var_idx = torch.stack(var_idx)

        # generate random constraint weights
        # multiplier = 100 # need to tune this multiplier...
        # cst_weights = torch.rand(num_cst) * multiplier
        min_weight = 1
        max_weight = 3
        cst_weights = torch.randint(low=min_weight, high=max_weight+1, size=(num_cst,))
        # cst_weights = torch.full((num_cst,), 1)

        csp_data.add_all_different_constraint_data(var_idx, cst_weights)
            

    def create_random_instance(self):

        # To model unary constraints, generate random domain for each unit (set of physicians that can be allocated to each unit Mi)
        # domains = [torch.sort(torch.unique(torch.randint(0, self.num_phys, size=(torch.randint(1, self.num_phys + 1, size=(1,)).item(),))))[0] for _ in range(num_var)]
        # domain_sizes = torch.tensor([len(dom) for dom in domains])
        # domains = torch.cat(domains)

        domains = torch.tile(torch.arange(self.num_phys), (self.num_var,))
        domain_sizes = torch.full((self.num_var,), self.num_phys)

        data = CSP_Data(num_var=self.num_var, domain_size=domain_sizes, domain=domains)

        # self.generate_unary_constraints(data)

        self.generate_multiple_constraints(data)

        return data



class KSAT_Generator:

    def __init__(self, min_n=100, max_n=100, min_k=3, max_k=3, min_alpha=4.0, max_alpha=5.0):
        self.min_n = min_n
        self.max_n = max_n
        self.min_k = min_k
        self.max_k = max_k
        self.min_alpha = min_alpha
        self.max_alpha = max_alpha

    def create_random_instance(self):
        k = np.random.randint(self.min_k, self.max_k + 1)
        n = np.random.randint(self.min_n, self.max_n + 1)
        alpha = np.random.uniform(self.min_alpha, self.max_alpha)
        m = max(int(np.ceil(n * alpha)), 1)
        cnf = cnfgen.RandomKCNF(k, n, m)
        
        cnf = [cls for cls in cnf.clauses()]
        cnf = [np.int64(c) for c in cnf]

        num_var = np.max([np.abs(c).max() for c in cnf]) # number of variables
        num_const = len(cnf) # number of constraints

        arity = np.int64([c.size for c in cnf]) # for each contraint, how many variables participate in it. E.g. if there are 3 constraints, with C1 and C2 having 2 variables and C3 having 5 variables, then arity = [2,2,5]
        const_idx = np.arange(0, num_const, dtype=np.int64) # Constraint indices. If there are 3 constraints, then const_idx = [0,1,2]
        tuple_idx = np.repeat(const_idx, arity) # Given above arity and const_idx examples, we get tuple_idx = [0, 0, 1, 1, 2, 2, 2, 2]

        cat = np.concatenate(cnf, axis=0)
        var_idx = np.abs(cat) - 1 # If C1 and C2 each has variables X0 and X1 participating, and C2 has variables X5...X9, then var_idx=[[0, 1], [0, 1], [5, 6, 7, 8, 9]]
        val_idx = np.int64(cat > 0).reshape(-1)

        data = CSP_Data(num_var=num_var, domain_size=2) # number of Mi, number of physicians
        data.add_constraint_data(
            True,
            torch.tensor(const_idx),
            torch.tensor(tuple_idx),
            torch.tensor(var_idx),
            torch.tensor(val_idx)
        )
        return data


class COL_Generator_Base:

    def __init__(self, min_col, max_col):
        self.min_col = min_col
        self.max_col = max_col

    def sample_nx_graph_(self):
        raise NotImplementedError

    def create_random_instance(self):
        G = self.sample_nx_graph_()
        coloring = nx.greedy_color(G)
        greedy_num_col = max([c for v, c in coloring.items()]) + 1
        min_col = max(self.min_col, min(self.max_col, greedy_num_col - 1))
        max_col = max(self.min_col, min(self.max_col, greedy_num_col - 1))
        k = np.random.randint(min_col, max_col + 1)
        data = nx_to_col(G, k)
        return data


class COL_GNM_Generator(COL_Generator_Base):

    def __init__(self, min_n, max_n, min_deg, max_deg, min_col, max_col):
        self.min_n = min_n
        self.max_n = max_n
        self.min_deg = min_deg
        self.max_deg = max_deg
        super(COL_GNM_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        deg = np.random.uniform(self.min_deg, self.max_deg)
        num_edge = int(np.ceil(n * deg / 2.0))
        G = nx.gnm_random_graph(n, num_edge)
        return G


class COL_ER_Generator(COL_Generator_Base):

    def __init__(self, min_n=20, max_n=70, min_p=0.2, max_p=0.3, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_p = min_p
        self.max_p = max_p
        super(COL_ER_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        num_vert = np.random.randint(self.min_n, self.max_n + 1)
        p = np.random.uniform(self.min_p, self.max_p)
        G = nx.erdos_renyi_graph(num_vert, p)
        while G.number_of_edges() <= 0:
            G = nx.erdos_renyi_graph(num_vert, p)
        return G


class COL_BA_Generator(COL_Generator_Base):

    def __init__(self, min_n=20, max_n=50, min_m=2, max_m=10, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_m = min_m
        self.max_m = max_m
        super(COL_BA_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        m = np.random.randint(self.min_m, self.max_m + 1)
        G = nx.barabasi_albert_graph(n, m)
        return G


class COL_REG_Generator(COL_Generator_Base):

    def __init__(self, min_n=50, max_n=50, min_d=3, max_d=20, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_d = min_d
        self.max_d = max_d
        super(COL_REG_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        d = np.random.randint(self.min_d, self.max_d + 1)
        G = nx.random_regular_graph(d, n)
        return G


class COL_GEO_Generator(COL_Generator_Base):

    def __init__(self, min_n=100, max_n=100, min_r=0.1, max_r=0.2, min_col=3, max_col=10):
        self.min_n = min_n
        self.max_n = max_n
        self.min_r = min_r
        self.max_r = max_r
        super(COL_GEO_Generator, self).__init__(min_col, max_col)

    def sample_nx_graph_(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        r = np.random.uniform(self.min_r, self.max_r)
        G = nx.random_geometric_graph(n, r)
        while G.number_of_edges() <= 0:
            G = nx.random_geometric_graph(n, r)
        return G


class RB_Generator:

    def __init__(self, min_k=2, max_k=4, min_n=5, max_n=40):
        self.min_k = min_k
        self.max_k = max_k
        self.min_n = min_n
        self.max_n = max_n

    def create_random_instance(self):
        k = np.random.randint(self.min_k, self.max_k + 1)
        n = np.random.randint(self.min_n, self.max_n + 1)
        data = get_random_RB(k, n)
        return data


class MC_ER_Generator:

    def __init__(self, min_n, max_n, min_p, max_p, weighted_prob=0.5):
        self.min_n = min_n
        self.max_n = max_n
        self.min_p = min_p
        self.max_p = max_p
        self.weighted_prob = weighted_prob

    def create_random_instance(self):
        n = np.random.randint(self.min_n, self.max_n + 1)
        p = np.random.uniform(self.min_p, self.max_p)
        G = nx.erdos_renyi_graph(n, p)
        num_edge = G.number_of_edges()

        if np.random.random() < self.weighted_prob:
            edge_weights = np.random.choice([-1, 1], (num_edge,))
        else:
            edge_weights = np.ones((num_edge,), dtype=np.int64)

        data = nx_to_maxcut(G, edge_weights)
        return data


generator_dict = {
    'KSAT': KSAT_Generator,
    'COL': COL_GNM_Generator,
    'COL_GNM': COL_GNM_Generator,
    'COL_BA': COL_BA_Generator,
    'COL_ER': COL_ER_Generator,
    'COL_GEO': COL_GEO_Generator,
    'COL_REG': COL_REG_Generator,
    'RB': RB_Generator,
    'MC_ER': MC_ER_Generator,
    "RESALLOC": ResAlloc_Generator
}
