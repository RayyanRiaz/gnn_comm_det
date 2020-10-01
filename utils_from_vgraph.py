import os
from subprocess import check_output

import numpy as np

# This file mostly copied from vgraph repo: https://github.com/fanyun-sun/vGraph

class NF1(object):
    def __init__(self, communities, ground_truth):
        self.matched_gt = {}
        self.gt_count = 0
        self.id_count = 0
        self.gt_nodes = {}
        self.id_nodes = {}
        self.communities = communities
        self.ground_truth = ground_truth
        self.prl = []
        # self.__compute_precision_recall()

    def get_f1(self):
        """

        :param prl: list of tuples (precision, recall)
        :return: a tuple composed by (average_f1, std_f1)
        """

        gt_coms = {cid: nodes for cid, nodes in enumerate(self.ground_truth)}
        ext_coms = {cid: nodes for cid, nodes in enumerate(self.communities)}

        f1_list = []
        for cid, nodes in gt_coms.items():
            tmp = [self.__compute_f1(nodes2, nodes) for _, nodes2 in ext_coms.items()]
            f1_list.append(np.max(tmp))

        f2_list = []
        for cid, nodes in ext_coms.items():
            tmp = [self.__compute_f1(nodes, nodes2) for _, nodes2 in gt_coms.items()]
            f2_list.append(np.max(tmp))

        # print(f1_list, f2_list)
        return (np.mean(f1_list) + np.mean(f2_list)) / 2

    def __compute_f1(self, c, gt):
        c = set(c)
        gt = set(gt)

        try:
            precision = len([x for x in c if x in gt]) / len(c)
            recall = len([x for x in gt if x in c]) / len(gt)
            x, y = precision, recall
            z = 2 * (x * y) / (x + y)
            z = float("%.2f" % z)
            return z
        except ZeroDivisionError:
            return 0.


def calc_jaccard(result_comm_list, ground_truth_comm_list):
    def func(s1, s2):
        s1, s2 = set(s1), set(s2)
        return len(s1.intersection(s2)) / len(s1.union(s2))

    gt_coms = {cid: nodes for cid, nodes in enumerate(ground_truth_comm_list)}
    ext_coms = {cid: nodes for cid, nodes in enumerate(result_comm_list)}

    f1_list = []
    for _, nodes in gt_coms.items():
        tmp = [func(nodes2, nodes) for _, nodes2 in ext_coms.items()]
        f1_list.append(np.max(tmp))

    f2_list = []
    for _, nodes in ext_coms.items():
        tmp = [func(nodes, nodes2) for _, nodes2 in gt_coms.items()]
        f2_list.append(np.max(tmp))

    # print(f1_list, f2_list)
    return (np.mean(f1_list) + np.mean(f2_list)) / 2


def calc_f1(result_comm_list, ground_truth_comm_list):
    # print(len(result_comm_list), len(ground_truth_comm_list))
    assert len(result_comm_list) == len(ground_truth_comm_list)
    nf = NF1(result_comm_list, ground_truth_comm_list)
    return nf.get_f1()


def write_to_file(fpath, clist):
    with open(fpath, 'w') as f:
        for c in clist:
            f.write(' '.join(map(str, c)) + '\n')


def calc_overlapping_nmi(communities_pred_cnl_format, communities_cnl_format):
    write_to_file('./gt', communities_cnl_format)
    write_to_file('./pred', communities_pred_cnl_format)
    ret = check_output(["./onmi", "pred", "gt"]).decode('utf-8')
    os.remove('./gt')
    os.remove('./pred')
    return float(ret.split('\n')[2])
