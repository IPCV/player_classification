import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import accuracy_score
from sklearn.metrics import confusion_matrix


def build_cost_matrix(c1, c2, uc1=None, uc2=None):
    if uc1 is None or uc2 is None:
        uc1, uc2 = np.unique(c1), np.unique(c2)
    l1, l2 = uc1.size, uc2.size

    m = np.ones([l1, l2])
    for i in range(l1):
        it_i = np.nonzero(c1 == uc1[i])[0]
        for j in range(l2):
            it_j = np.nonzero(c2 == uc2[j])[0]
            m_ij = np.intersect1d(it_j, it_i)
            m[i, j] = -m_ij.size
    return m


def hungarian_mapping(y_true, y_pred, cost_matrix=None):
    if cost_matrix is None:
        cost_matrix = build_cost_matrix(y_pred, y_true)
    row_indices, col_indices = linear_sum_assignment(cost_matrix)

    uc = np.unique(y_true)
    assignments_map = {uc[k]: uc[v] for k, v in zip(row_indices, col_indices)}

    not_assigned_value = np.min(np.unique(y_pred)) - 1
    assign = np.vectorize(lambda x: assignments_map.get(x, not_assigned_value))
    return assign(y_pred)


def hungarian_mapping_accuracy_score(y_true, y_pred):
    return accuracy_score(y_true, hungarian_mapping(y_true, y_pred))


def build_players_cost_matrix(c1, c2):
    REFEREE = np.asarray([0])
    PLAYERS = np.asarray([1, 2])
    GOALKEEPERS = np.asarray([3, 4])

    m = np.zeros((5,5))
    for categories in [REFEREE, PLAYERS, GOALKEEPERS]:
        for i in categories:
            it_i = np.nonzero(c1 == i)[0]
            for j in categories:
                it_j = np.nonzero(c2 == j)[0]
                m_ij = np.intersect1d(it_j, it_i)
                m[i, j] = -m_ij.size
    return m


def hungarian_players_mapping(y_true, y_pred):
    return hungarian_mapping(y_true, y_pred, build_players_cost_matrix(y_pred, y_true))


def acc_per_class(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    return cm.diagonal() / cm.sum(axis=1)


def macro_acc(y_true, y_pred):
    return np.average(acc_per_role(y_true, y_pred))


def acc_per_role(y_true, y_pred):
    cm = confusion_matrix(y_true, y_pred)
    total_per_class = cm.sum(axis=1)
    true_positives = cm.diagonal()
    referee = true_positives[0] / total_per_class[0]
    players = true_positives[1:3].sum() / total_per_class[1:3].sum()
    goalkeepers = true_positives[3:5].sum() / total_per_class[3:5].sum()
    return referee, players, goalkeepers
