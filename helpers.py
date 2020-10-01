from enum import Enum, auto

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, normalized_mutual_info_score, f1_score

from utils_from_vgraph import calc_f1, calc_jaccard


class AutoName(Enum):
    def _generate_next_value_(name, start, count, last_values):
        return name


class Scores(AutoName):
    COMMUNITY_OVERLAPPING_F1 = auto()
    COMMUNITY_OVERLAPPING_JACCARD = auto()
    COMMUNITY_NMI = auto()
    COMMUNITY_ARI = auto()
    NODE_CLASSIFICATION_ACCURACY = auto()
    NODE_CLASSIFICATION_F1_MICRO = auto()
    NODE_CLASSIFICATION_F1_MACRO = auto()


def map_labels(Y_pred, Y):
    assert Y_pred.size == Y.size
    D = max(Y_pred.max(), Y.max()) + 1
    w = np.zeros((D, D), dtype=np.int64)
    for i in range(Y_pred.size):
        w[Y_pred[i], Y[i]] += 1
    ind = linear_sum_assignment(w.max() - w)
    return ind


def matrix_to_cnl_format(communities_assignment_matrix, num_communities):
    return [np.where(communities_assignment_matrix[i])[0].tolist() for i in range(num_communities)]


def predict_node_classification(train_z, train_y, test_z,
                                # solver='lbfgs',
                                solver='liblinear', multi_class='auto', *args, **kwargs):
    clf = LogisticRegression(solver=solver, multi_class=multi_class, *args, **kwargs) \
        .fit(train_z.detach().cpu().numpy(), train_y.detach().cpu().numpy())

    return clf.predict(test_z.detach().cpu().numpy()).astype(int)


def kv_to_print_str(kvs):
    return "".join(["{}: {:.4f}||\t".format(str(k).replace("Scores.", ""), v) for k, v in kvs.items()])


def scores(keys, match_labels=True, print_down=False, **kwargs):
    if match_labels:
        if 'communities' in kwargs and 'communities_pred' in kwargs:
            kwargs['communities_pred'] = map_labels(kwargs['communities_pred'], kwargs['communities'])[1][kwargs['communities_pred']]
        if 'node_classes' in kwargs and 'node_classes' in kwargs:
            kwargs['node_classes_pred'] = map_labels(kwargs['node_classes_pred'], kwargs['node_classes'])[1][kwargs['node_classes_pred']]

    ret = {}
    for key in keys:
        if key == Scores.COMMUNITY_ARI:
            v = adjusted_rand_score(kwargs['communities'], kwargs['communities_pred'])
        elif key == Scores.COMMUNITY_NMI:
            v = normalized_mutual_info_score(kwargs['communities'], kwargs['communities_pred'], average_method='arithmetic')
        elif key == Scores.COMMUNITY_OVERLAPPING_F1:
            v = calc_f1(kwargs['communities_cnl_pred'], kwargs['communities_cnl'])
        elif key == Scores.COMMUNITY_OVERLAPPING_JACCARD:
            v = calc_jaccard(kwargs['communities_cnl_pred'], kwargs['communities_cnl'])
        elif key == Scores.NODE_CLASSIFICATION_ACCURACY:
            v = accuracy_score(kwargs['node_classes'], kwargs['node_classes_pred'])
        elif key == Scores.NODE_CLASSIFICATION_F1_MICRO:
            v = f1_score(kwargs['node_classes'], kwargs['node_classes_pred'], average='micro')
        elif key == Scores.NODE_CLASSIFICATION_F1_MACRO:
            v = f1_score(kwargs['node_classes'], kwargs['node_classes_pred'], average='macro')
        else:
            raise Exception("Unknown key")
        ret[key] = v
    if print_down:
        print(kv_to_print_str(ret))
    return ret
