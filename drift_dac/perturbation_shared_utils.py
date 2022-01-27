import copy
import numpy as np


def sample_random_indices(total_size, fraction, replace=False):
    num_rows_to_pick = int(np.ceil(fraction * total_size))
    affected_indexes = sorted(list(np.random.choice(total_size, size=num_rows_to_pick, replace=replace)))
    return affected_indexes


class Shift(object):
    def __init__(self):
        self.name = None
        self.shifted_indices = None
        self.shifted_features = None
        self.feature_type = None

    def transform(self, X, y):
        return X, y

    def fit(self):
        pass


class NoShift(Shift):
    def __init__(self):
        super(NoShift, self).__init__()
        self.name = 'no_shift'
        self.feature_type = PerturbationConstants.ANY


class Mixture(Shift):
    def __init__(self, perturbations):
        super(Mixture, self).__init__()
        self.perturbations = perturbations
        self.name = '_'.join([perturbation.name for perturbation in self.perturbations])

    def transform(self, X, y):
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        for perturbation in self.perturbations:
            Xt, yt, shifted_indices, shifted_features = perturbation.transform(Xt, yt)
            if not (shifted_indices is None):
                self.shifted_indices = sorted(list(set(self.shifted_indices + shifted_indices)))
            if not (shifted_features is None):
                self.shifted_features = sorted(list(set(self.shifted_features + shifted_features)))
        return Xt, yt


def any_other_label(label_to_change, labels):
    if len(labels) == 1:
        # no other possible value to select. Returning the one single allowed value.
        return label_to_change
    new_label = np.random.choice(labels[labels != label_to_change])
    return new_label


class PerturbationConstants(object):
    NUMERIC = 0
    CATEGORICAL = 1
    ANY = 2
    TEXT = 3

