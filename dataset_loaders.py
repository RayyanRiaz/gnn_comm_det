import os.path as osp
from enum import Enum

import numpy as np
import torch
import torch_geometric.transforms as T
from torch_geometric.data import Data
from torch_geometric.datasets import Planetoid, SNAPDataset, CitationFull
from torch_sparse import coalesce

from helpers import matrix_to_cnl_format

DATASETS_DIR = osp.join(osp.dirname(osp.realpath(__file__)), 'data')


class PlanetoidDataset(Enum):
    Cora = "Cora"
    CiteSeer = "CiteSeer"


class CitationFullDataset(Enum):
    Cora = "Cora"
    CoraML = "Cora_ML"
    CiteSeer = "CiteSeer"


class SnapDataset(Enum):
    EgoFacebook = "ego-facebook"


class LargeDataset(Enum):
    Amazon = "amazon"
    Youtube = "youtube"
    DBLP = "dblp"


def load_non_overlapping_dataset(dataset_name: PlanetoidDataset or CitationFullDataset, transform=T.NormalizeFeatures()) -> Data:
    path = osp.join(DATASETS_DIR, dataset_name.value)

    if type(dataset_name) == PlanetoidDataset:
        data = Planetoid(path, dataset_name.value, transform=transform)[0]
    elif type(dataset_name) == CitationFullDataset:
        data = CitationFull(path, dataset_name.value, transform=transform)[0]
    else:
        raise Exception("Unknown dataset name")
    return data


def load_overlapping_dataset(dataset_name: SnapDataset, allow_features=True, data_idx=None) -> Data:
    path = osp.join(DATASETS_DIR, dataset_name.value)

    if type(dataset_name) == SnapDataset:
        data = SNAPDataset(path, dataset_name.value, T.NormalizeFeatures())
    else:
        raise Exception("Unknown dataset name")

    if data_idx is not None:
        data = data[data_idx]

    if not allow_features:
        data.x = torch.eye(data.x.size(0))

    if type(dataset_name) == SnapDataset:
        data.num_communities = data.circle_batch.max() + 1
        communities = np.zeros((data.num_communities, data.x.size(0)))
        communities[data.circle_batch, data.circle] = 1
        data.communities = communities
        data.communities_cnl_format = matrix_to_cnl_format(communities, data.num_communities)

    return data


def load_large_dataset(dataset_name, NUM_COMMUNITIES=5):
    # TODO: Add more datasets if required, from https://snap.stanford.edu/data/#communities
    #  This will be simple as you already have the code and only need to put files in the
    #  folders as done in amazon/youtube/dlbp case
    path = osp.join(DATASETS_DIR, dataset_name.value)
    edges_file = 'com-{}.ungraph.txt'.format(dataset_name.value, dataset_name)
    communities_file = 'com-{}.top5000.cmty.txt'.format(dataset_name.value, dataset_name)

    with open(osp.join(path, communities_file), 'r') as f:
        communities = [line.strip().split('\t') for line in f.readlines()]
    communities.sort(key=lambda x: len(x), reverse=True)
    communities = communities[:NUM_COMMUNITIES]
    nodes_in_communities_set = set(x for r in communities for x in r)
    nodes_in_communities_sorted = sorted(list(int(x) for x in nodes_in_communities_set))
    nodes_idx_map = {v: k for k, v in enumerate(list(nodes_in_communities_sorted))}

    for i in range(len(communities)):
        communities[i] = [nodes_idx_map[int(x)] for x in communities[i]]
    with open(osp.join(path, edges_file), 'r') as f:
        edges = [line.strip().split('\t') for line in f.readlines()]

    edges = [edge for edge in edges if edge[0] in nodes_in_communities_set and edge[1] in nodes_in_communities_set]

    edges = [[nodes_idx_map[int(edge[0])], nodes_idx_map[int(edge[1])]] for edge in edges]
    edges = coalesce(
        torch.from_numpy(np.array(edges)).T, None,
        len(nodes_in_communities_sorted), len(nodes_in_communities_sorted))[0]
    data = Data(x=torch.eye(len(nodes_in_communities_sorted)), edge_index=edges, communities=communities)
    data.num_communities = len(data.communities)
    data.communities_cnl_format = data.communities

    return data
