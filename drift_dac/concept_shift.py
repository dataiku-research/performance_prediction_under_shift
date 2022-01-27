import numpy as np
from math import floor
import copy

from drift_dac.perturbation_shared_utils import Shift, PerturbationConstants, any_other_label

__all__ = ['LabelSwitch']


class LabelSwitch(Shift):
    """ Assign a random class to a portion of data.
    Args:
        perc_max_changes (float): proportion of samples to perturb.
    Attributes:
        perc_max_changes (float): proportion of samples to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """
    def __init__(self, perc_max_changes=0.3):
        super(LabelSwitch, self).__init__()
        self.perc_max_changes = perc_max_changes
        self.name = 'label_switch_shift_%.2f' % perc_max_changes
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X=None, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        if y is None:
            raise ValueError("Input y must be provided.")
        Xt = X
        yt = copy.deepcopy(y)
        yt, self.shifted_indices = random_class_subset_shift(yt, self.perc_max_changes)
        return Xt, yt


# Random change class of a subset of data.
def random_class_subset_shift(y, perc_max_changes=0.3):
    labels, counts = np.unique(y, return_counts=True)

    subset_indices = np.random.choice(y.shape[0], floor(y.shape[0] * perc_max_changes), replace=False)

    rand_labeler = lambda y: any_other_label(y, labels)
    vec_rand_labeler = np.vectorize(rand_labeler)
    y[subset_indices] = vec_rand_labeler(y[subset_indices])

    return y, subset_indices