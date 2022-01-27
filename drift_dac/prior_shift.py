import numpy as np
from math import ceil
import copy
from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants

__all__ = ['OnlyOne', 'KnockOut', 'Rebalance']


class OnlyOne(Shift):
    """ Sample data to keep only one class.
    Args:
        keep_cl (int or str): class to keep
    Attributes:
        keep_cl (int or str): class to keep
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, keep_cl=0):
        super(OnlyOne, self).__init__()
        self.keep_cl = keep_cl
        self.name = 'oo_shift_%s' % keep_cl
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt = only_one_shift(Xt, yt, self.keep_cl)
        return Xt, yt


class KnockOut(Shift):
    """ Sample data to remove a portion of a given class.
    Args:
        cl (int or str): class to subsample
    Attributes:
        cl (int or str): class to subsample
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, cl=0, samples_fraction=1.0):
        super(KnockOut, self).__init__()
        self.cl = cl
        self.samples_fraction = samples_fraction
        self.name = 'ko_shift_%s_%.2f' % (cl, samples_fraction)
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt = knockout_shift(Xt, yt, self.cl, self.samples_fraction)
        return Xt, yt


class Rebalance(Shift):
    """ Sample data to match a given distribution of classes.
    Args:
        priors (array-like): desired classes frequencies
    Attributes:
        priors (array-like): desired classes frequencies
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, priors):
        super(Rebalance, self).__init__()
        self.priors = priors
        self.name = 'rebalance_shift'
        for p in priors:
            self.name += '_%.2f' % p
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt = rebalance_shift(Xt, yt, self.priors)
        return Xt, yt


# Resample instances of all classes by given priors.
def rebalance_shift(x, y, priors):
    labels, counts = np.unique(y, return_counts=True)
    n_labels = len(labels)
    n_priors = len(priors)
    assert (n_labels == n_priors)
    assert (np.sum(priors) == 1.)
    n_to_sample = y.shape[0]
    for label_idx, prior in enumerate(priors):
        if prior > 0:
            n_samples_label = counts[label_idx]
            max_n_to_sample = np.round(n_samples_label / prior)
            if n_to_sample > max_n_to_sample:
                n_to_sample = max_n_to_sample

    resampled_counts = [int(np.round(prior * n_to_sample)) for prior in priors]
    resampled_indices = []
    for cl, res_count in zip(labels, resampled_counts):
        if res_count:
            cl_indices = np.where(y == cl)[0]
            cl_res_indices = np.random.choice(cl_indices, res_count, replace=False)
            resampled_indices.extend(cl_res_indices)

    resampled_indices = np.random.choice(resampled_indices, x.shape[0], replace=True)

    x = x[resampled_indices, :]
    y = y[resampled_indices]
    return x, y


# Remove instances of a single class.
def knockout_shift(x, y, cl, delta):
    n_rows = x.shape[0]
    del_indices = np.where(y == cl)[0]
    until_index = ceil(delta * len(del_indices))
    if until_index % 2 != 0:
        until_index = until_index + 1
    del_indices = del_indices[:until_index]
    x = np.delete(x, del_indices, axis=0)
    y = np.delete(y, del_indices, axis=0)

    indices_cl = np.where(y == cl)[0]
    indices_not_cl = np.where(y != cl)[0]
    repeat_indices = np.random.choice(indices_not_cl, n_rows-len(indices_cl), replace=True)
    x_not_cl = x[repeat_indices, :]
    y_not_cl = y[repeat_indices]

    x = np.concatenate((x_not_cl, x[indices_cl, :]))
    y = np.concatenate((y_not_cl, y[indices_cl]))

    permuted_indices = np.random.permutation(n_rows)

    x = x[permuted_indices, :]
    y = y[permuted_indices]

    return x, y


# Remove all classes except for one via multiple knock-out.
def only_one_shift(x, y, keep_cl):
    labels = np.unique(y)
    for cl in labels:
        if cl != keep_cl:
            x, y = knockout_shift(x, y, cl, 1.0)

    return x, y
