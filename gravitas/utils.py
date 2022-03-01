import os
import warnings

import numpy as np
import pandas as pd
import torch
from networkx import Graph, minimum_spanning_edges


def freeze(listoflayers, unfreeze=True):
    """freeze the parameters of the list of layers """
    for l in listoflayers:
        for p in l.parameters():
            p.requires_grad = unfreeze


def calc_min_eucl_spanning_tree(d_test: torch.tensor):
    dist_mat = torch.cdist(d_test, d_test)
    dist_mat = dist_mat.cpu().detach().numpy()

    nodes = list(range(len(dist_mat)))
    d = [(src, dst, dist_mat[src, dst]) for src in nodes for dst in nodes if src != dst]

    df = pd.DataFrame(data=d, columns=['src', 'dst', 'eucl'])

    g = Graph()
    for index, row in df.iterrows():
        g.add_edge(row['src'], row['dst'], weight=row['eucl'])

    return list(minimum_spanning_edges(g))  # fixme: VAE produces NAN sometimes


def check_diversity(representation, title, epsilon=0.01):
    """

    :param representation: ndarray.
    :param title: name of the matrix
    :param epsilon: float: the value needed to exceed (should be close to zero)
    :raises: Warning if representation is not diverse
    """
    # Check for (naive) representation collapse by checking sparsity after
    # translation by 90% quantile
    translated = representation - np.quantile(representation, 0.9, axis=0)
    sparsity = (translated < epsilon).sum() / np.product(representation.shape)
    if sparsity >= 0.95:
        warnings.warn(f'The {title} representation is not diverse.')

        # Warning(f'The {title} representation is not diverse.')
        print(representation)

def check_or_create_dir(dir):

    # If folder doesn't exist, then create it.
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print("created folder : ", dir)

    else:
        print(dir, "folder already exists.")
