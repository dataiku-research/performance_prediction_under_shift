import numpy as np
from math import ceil
import copy
import random

from imblearn.under_sampling import NearMiss
from imblearn.over_sampling import SMOTE, ADASYN

from sklearn.ensemble import RandomForestClassifier

try:  # < version 1.1.0
    from art.attacks import BoundaryAttack, ZooAttack, HopSkipJump
except ImportError:  # >= version 1.1.0
    from art.attacks.evasion import BoundaryAttack, ZooAttack, HopSkipJump

from art.classifiers import SklearnClassifier
from sklearn import preprocessing
from sklearn.model_selection import train_test_split
from sklearn.impute import SimpleImputer
from sklearn.metrics.pairwise import euclidean_distances

from drift_dac.features_utils import is_integer_feature
from drift_dac.perturbation_shared_utils import Shift, sample_random_indices, PerturbationConstants, any_other_label

__all__ = ['GaussianNoise', 'SwitchCategorical', 'SubsampleJoint', 'SubsampleNumeric', 'SubsampleCategorical',
           'UnderSample', 'OverSample', 'Adversarial', 'ConstantNumeric', 'ConstantCategorical', 'MissingValues',
           'PlusMinusSomePercent', 'FlipSign', 'Outliers', 'Scaling', 'SwappedValues', 'ErrorBasedSampling',
           'NearestNeighbors']


class GaussianNoise(Shift):
    """ Apply Gaussian noise to a portion of data.
    Args:
        noise_key (str): value in ['small', 'medium', 'large] indicating the level of noise.
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        clip (bool): flag to clip the value of a perturbed feature to the maximum value in the feature range.
        ceil_int (bool): flag to ceil the value of a perturbed feature for integer features.
        noise (float): level of noise
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
        clip (bool): flag to clip the value of a perturbed feature to the maximum value in the feature range.
        ceil_int (bool): flag to ceil the value of a perturbed feature for integer features.
        noise (float): amount of noise
    """

    def __init__(self, noise_key='small', samples_fraction=1.0, features_fraction=1.0, clip=True, ceil_int=True,
                 noise=None):
        super(GaussianNoise, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.clip = clip
        self.ceil_int = ceil_int
        if noise is None:
            noise_levels = {
                'large': 100.0,
                'medium': 10.0,
                'small': 1.0
            }
            self.noise = noise_levels[noise_key]
        else:
            self.noise = noise

        self.name = noise_key + '_gn_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt, self.shifted_indices, self.shifted_features = gaussian_noise_shift(Xt, yt,
                                                                                   self.noise, self.samples_fraction,
                                                                                   self.features_fraction, self.clip,
                                                                                   self.ceil_int)
        return Xt, yt


class SwitchCategorical(Shift):
    """ Assign a random category of categorical variables to a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=0.5, features_fraction=1.0):
        super(SwitchCategorical, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.name = 'switch_categorical_features_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.CATEGORICAL

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt, self.shifted_indices, self.shifted_features = switch_categorical_features_shift(Xt, yt,
                                                                                                self.samples_fraction,
                                                                                                self.features_fraction)
        return Xt, yt


class SubsampleJoint(Shift):
    """ Keep an observation with a probability which decreases as points are further away from the samples mean.
    Args:
        gamma (float): fraction of samples close to the mean to remove.
    Attributes:
        gamma (float): fraction of samples close to the mean to remove.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, gamma=0.2):
        super(SubsampleJoint, self).__init__()
        self.gamma = 1 - gamma  # fraction of samples to keep
        self.name = 'subsample_joint_shift_%.2f' % gamma
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt, self.shifted_features = subsample_joint_shift(Xt, yt, gamma=self.gamma)
        return Xt, yt


class SubsampleNumeric(Shift):
    """ Subsample with probability p when feature value less than the feature median value. If one_side=False, this also
    subsamples with probability 1-p when the feature value is larger than the feature median value.
    Args:
        features_fraction (float): proportion of samples to perturb.
        p (float): probability of sampling small values of features
        one_side (bool): flag to subsample only samples with small feature values (True, default) or also large values.
    Attributes:
        features_fraction (float): proportion of samples to perturb.
        p (float): probability of sampling small values of features
        one_side (bool): flag to subsample only samples with small feature values (True, default) or also large values.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, p=0.5, features_fraction=0.5, one_side=True):
        super(SubsampleNumeric, self).__init__()
        self.features_fraction = features_fraction
        self.p = p
        self.one_side = one_side
        self.name = 'subsample_feature_shift_%.2f' % features_fraction
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt, self.shifted_features = subsample_feature_shift(Xt, yt, feat_delta=self.features_fraction, p=self.p,
                                                                one_side=self.one_side)
        return Xt, yt


class SubsampleCategorical(Shift):
    """ Subsample with probability p when feature value falls in a sub list of categories (randomly defined).
    If one_side=False, this also subsamples with probability 1-p when the feature value in the remaining categories.
    This perturbation is applied to all categorical features.
    Args:
        p (float): probability of sampling subset of values of features
        features_fraction (float): proportion of samples to perturb.
        one_side (bool): flag to subsample only samples with small feature values (True, default) or also large values.
    Attributes:
        p (float): probability of sampling subset of values of features
        features_fraction (float): proportion of samples to perturb.
        one_side (bool): flag to subsample only samples with small feature values (True, default) or also large values.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, p=0.5, features_fraction=1.0, one_side=True):
        super(SubsampleCategorical, self).__init__()
        self.p = p
        self.features_fraction = features_fraction
        self.one_side = one_side
        self.name = 'subsample_categorical_feature_shift_%.2f' % p
        self.feature_type = PerturbationConstants.CATEGORICAL

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)

        n_categorical = Xt.shape[1]
        feat_indices = np.random.choice(n_categorical, ceil(n_categorical * self.features_fraction), replace=False)
        for f in feat_indices:
            Xt, yt = subsample_categorical_feature_shift(Xt, yt, f, p=self.p, one_side=self.one_side)

        return Xt, yt


class UnderSample(Shift):
    """ Subsample selecting samples close to the minority class (NearMiss3 heuristics).
    Args:
        samples_fraction (float): proportion of samples to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=0.5):
        super(UnderSample, self).__init__()
        self.samples_fraction = samples_fraction
        self.name = 'under_sample_shift_%.2f' % samples_fraction
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt = SimpleImputer(strategy='median').fit_transform(Xt)
        Xt, yt = under_sampling_shift(Xt, yt, self.samples_fraction)
        return Xt, yt


class OverSample(Shift):
    """ First subsample selecting samples close to the minority class (NearMiss3 heuristics), then add interpolated
    samples via SMOTE technique.
    Args:
        samples_fraction (float): proportion of samples to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=0.5):
        super(OverSample, self).__init__()
        self.samples_fraction = samples_fraction
        self.name = 'over_sample_shift_%.2f' % samples_fraction
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt = SimpleImputer(strategy='median').fit_transform(Xt)
        Xt, yt = over_sampling_shift(Xt, yt, self.samples_fraction)
        return Xt, yt


class Adversarial(Shift):
    """ Perturb a portion of samples and features via black-box adversarial perturbations attempting to make a
    classifier mis-predict the samples class.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        attack_type (str): name of the desired adversarial attack in ['zoo', 'boundary', 'hop-skip-jump'].
        model (sklearn.BaseEstimator): unfitted sklearn classifier against which to perform the adversarial attack.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        attack_type (str): name of the desired adversarial attack in ['zoo', 'boundary', 'hop-skip-jump'].
        model (sklearn.BaseEstimator): unfitted sklearn classifier against which to perform the adversarial attack.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0, attack_type='boundary',
                 model=RandomForestClassifier()):
        super(Adversarial, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.attack_type = attack_type
        self.model = model
        self.name = 'adversarial_attack_shift_%s_%.2f_%.2f' % (attack_type, samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)

        Xt, yt, self.shifted_indices, self.shifted_features = adversarial_attack_shift(Xt, yt, self.samples_fraction,
                                                                                       self.model,
                                                                                       self.attack_type,
                                                                                       self.features_fraction)
        return Xt, yt


class ConstantNumeric(Shift):
    """ Assign a constant value (median) to a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0):
        super(ConstantNumeric, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.name = 'constant_feature_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt, self.shifted_indices, self.shifted_features = constant_value_shift(Xt, yt, self.samples_fraction,
                                                                                   self.features_fraction)
        return Xt, yt


class ConstantCategorical(Shift):
    """ Assign a random constant category to a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0):
        super(ConstantCategorical, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.name = 'constant_feature_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.CATEGORICAL

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = copy.deepcopy(y)
        Xt, yt, self.shifted_indices, self.shifted_features = constant_categorical_shift(Xt, yt, self.samples_fraction,
                                                                                         self.features_fraction)
        return Xt, yt


class MissingValues(Shift):
    """ Insert missing values into a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        value_to_put_in (float): desired representation of the missing value
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        value_to_put_in (float): desired representation of the missing value
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0, value_to_put_in=None):
        super(MissingValues, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.value_to_put_in = value_to_put_in

        self.name = 'missing_value_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.ANY

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        self.shifted_features = sample_random_indices(Xt.shape[1], self.features_fraction)

        if self.value_to_put_in is None and np.issubdtype(Xt[:, self.shifted_features].dtype, np.integer):
            self.value_to_put_in = -9999

        Xt[np.transpose(np.array(self.shifted_indices)[np.newaxis]), np.array(self.shifted_features)[
            np.newaxis]] = self.value_to_put_in

        return Xt, yt


class FlipSign(Shift):
    """ Flip the sign of a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0):
        super(FlipSign, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction

        self.name = 'flip_sign_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        numerical_features = np.array(range(Xt.shape[1]))
        n_feats = len(numerical_features)
        self.shifted_features = list(
            np.array(numerical_features)[sample_random_indices(n_feats, self.features_fraction)])

        Xt[np.transpose(np.array(self.shifted_indices)[np.newaxis]), np.array(self.shifted_features)[np.newaxis]] *= -1

        return Xt, yt


class PlusMinusSomePercent(Shift):
    """ Increment the feature values by a percentage for a portion of data.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        percentage (float): percentage of the value to add/subtract.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        percentage (float): percentage of the value to add/subtract.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0, percentage=0.1):
        super(PlusMinusSomePercent, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.percentage = percentage

        self.name = 'plus_minus_perc_shift_%.2f_%.2f_%.2f' % (samples_fraction, features_fraction, percentage)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        numerical_features = np.array(range(Xt.shape[1]))
        n_feats = len(numerical_features)
        self.shifted_features = list(
            np.array(numerical_features)[sample_random_indices(n_feats, self.features_fraction)])

        row_indices, col_indices = np.transpose(np.array(self.shifted_indices)[np.newaxis]), \
                                   np.array(self.shifted_features)[np.newaxis]
        Xt[row_indices, col_indices] += \
            Xt[row_indices, col_indices] * np.random.uniform(-self.percentage, self.percentage,
                                                             size=(len(self.shifted_indices),
                                                                   len(self.shifted_features)))

        return Xt, yt


class Outliers(Shift):
    """ Replace a portion of data with outliers, obtained by adding random Gaussian noise.
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0):
        super(Outliers, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.name = 'outlier_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        numerical_features = np.array(range(Xt.shape[1]))
        n_feats = len(numerical_features)
        self.shifted_features = list(
            np.array(numerical_features)[sample_random_indices(n_feats, self.features_fraction)])

        stddevs = {column: np.std(Xt[:, column]) for column in self.shifted_features}
        scales = {column: random.uniform(1, 5) for column in self.shifted_features}

        for column in self.shifted_features:
            noise = np.random.normal(0, scales[column] * stddevs[column], size=Xt[self.shifted_indices, column].shape)
            Xt[self.shifted_indices, column] += noise

        return Xt, yt


class Scaling(Shift):
    """ Scale a portion of samples and features by a random value in [10, 100, 1000].
    Args:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        features_fraction (float): proportion of features to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0, features_fraction=1.0):
        super(Scaling, self).__init__()
        self.samples_fraction = samples_fraction
        self.features_fraction = features_fraction
        self.name = 'scaling_shift_%.2f_%.2f' % (samples_fraction, features_fraction)
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        numerical_features = np.array(range(Xt.shape[1]))
        n_feats = len(numerical_features)
        self.shifted_features = list(
            np.array(numerical_features)[sample_random_indices(n_feats, self.features_fraction)])
        row_indices, col_indices = np.transpose(np.array(self.shifted_indices)[np.newaxis]), \
                                   np.array(self.shifted_features)[np.newaxis]

        scale_factor = np.random.choice([10, 100, 1000])

        Xt[row_indices, col_indices] *= scale_factor

        return Xt, yt


class SwappedValues(Shift):
    """ Swap the values of two random features for a desired portion of samples.
    Args:
        samples_fraction (float): proportion of samples to perturb.
    Attributes:
        samples_fraction (float): proportion of samples to perturb.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, samples_fraction=1.0):
        super(SwappedValues, self).__init__()
        self.samples_fraction = samples_fraction
        self.name = 'swapped_values_shift_%.2f' % samples_fraction
        self.feature_type = PerturbationConstants.NUMERIC

    def transform(self, X, y=None):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """
        Xt = copy.deepcopy(X)
        yt = y

        self.shifted_indices = sample_random_indices(Xt.shape[0], self.samples_fraction)
        self.shifted_features = sorted(list(np.random.choice(Xt.shape[1], size=2, replace=False)))
        (column_a, column_b) = self.shifted_features[0], self.shifted_features[1]

        values_of_column_a = copy.deepcopy(Xt[self.shifted_indices, column_a])
        values_of_column_b = Xt[self.shifted_indices, column_b]

        Xt[self.shifted_indices, column_a] = values_of_column_b
        Xt[self.shifted_indices, column_b] = values_of_column_a

        return Xt, yt


class ErrorBasedSampling(Shift):
    """ Sample the observations to have a desired amount of wrongly predicted samples from a reference model.
    Args:
        error_fraction (float): proportion of wrongly predicted samples.
        model (sklearn.BaseEstimator): classifier with respect to which errors are computed.
        labelenc (sklearn.BaseEncoder): label encoder for the classifier.
    Attributes:
        error_fraction (float): proportion of wrongly predicted samples.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
        model (sklearn.BaseEstimator): classifier with respect to which errors are computed.
        labelenc (sklearn.BaseEncoder): label encoder for the classifier.
    """

    def __init__(self, error_fraction=1.0, model=None, labelenc=None):
        super(ErrorBasedSampling, self).__init__()
        self.error_fraction = error_fraction
        self.name = 'error_sampling_shift_%.2f' % error_fraction
        self.feature_type = PerturbationConstants.ANY
        self.model = model
        self.labelenc = labelenc

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """

        if self.model is None:
            raise NotImplementedError('You need to input a model. Reference model not implemented yet.')

        y_pred = self.model.predict(X)

        if self.labelenc is None:
            y_enc = y
        else:
            y_enc = self.labelenc.transform(y)

        error_indices = np.where(y_pred != y_enc)[0]
        correct_indices = np.where(y_pred == y_enc)[0]

        if error_indices.sum() == 0:
            raise ValueError('The model has 0 error on the dataset. ErrorBasedSampling cannot be built.')

        n_samples = X.shape[0]
        n_errors = int(np.ceil(self.error_fraction * n_samples))
        n_correct = n_samples - n_errors

        error_row_indices = np.random.choice(error_indices, size=n_errors, replace=True)
        correct_row_indices = np.random.choice(correct_indices, size=n_correct, replace=True)
        row_indices = np.r_[error_row_indices, correct_row_indices]
        np.random.shuffle(row_indices)

        Xt = X[row_indices, :]
        yt = y[row_indices]

        return Xt, yt


class NearestNeighbors(Shift):
    """ Simulate a particular demographic either appearing, or disappearing from production traffic.

    It does so by sampling a data point and then uses nearest neighbors to identify other data points that are similar
    (nearest neighbors) or dissimilar (furthest neighbors) and remove them from the dataset.
    Ref: https://arxiv.org/abs/2012.08625

    Args:
        fraction_to_remove (float): proportion of samples to remove from the nearest/furthest set.
        near_far_probability (float): probability of nearest or furthest bias.
    Attributes:
        fraction_to_remove (float): proportion of samples to remove from the nearest/furthest set.
        near_far_probability (float): probability of nearest or furthest bias.
        name (str): name of the perturbation
        feature_type (int): identifier of the type of feature for which this perturbation is valid
            (see PerturbationConstants).
    """

    def __init__(self, fraction_to_remove=0.5, near_far_probability=0.5):
        super(NearestNeighbors, self).__init__()
        self.fraction_to_remove = fraction_to_remove
        self.name = 'nearest_neighbors_shift_%.2f_%.2f' % (fraction_to_remove, near_far_probability)
        self.feature_type = PerturbationConstants.NUMERIC
        self.near_far_probability = near_far_probability

    def transform(self, X, y):
        """ Apply the perturbation to a dataset.
        Args:
            X (numpy.ndarray): feature data.
            y (numpy.ndarray): target data.
        """

        initial_size = X.shape[0]

        # choose random point p
        p_idx = np.random.choice(X.shape[0], size=1)

        # sort samples by distance to p
        dist_to_p = euclidean_distances(X, X[p_idx])[:, 0]
        idx_by_distance = np.argsort(dist_to_p)

        remove_size = int(np.ceil(self.fraction_to_remove * initial_size))
        keep_size = X.shape[0] - remove_size

        if random.random() < self.near_far_probability:
            # remove nearest
            keep_idx = idx_by_distance[-keep_size:]
        else:
            # remove farthest
            keep_idx = idx_by_distance[:keep_size]

        # resample to restore initial size (not in the original algorithm, introduce duplicates)
        keep_idx = np.append(keep_idx, np.random.choice(keep_idx, size=remove_size))

        Xt = X[keep_idx, :]
        yt = y[keep_idx]

        return Xt, yt


# Keeps an observation with a probability which decreases as points are further away from the samples mean
# gamma is the fraction of samples close to the mean we want to keep
def subsample_joint_shift(x, y, gamma=0.8):
    shift_features = list(range(x.shape[1]))

    n_rows = x.shape[0]

    x_mean = np.nanmean(x, axis=0)
    distance = np.sqrt(np.sum((x - x_mean) ** 2, axis=1))
    gamma_quantile = np.quantile(distance[~np.isnan(distance)], gamma)
    ln_prob_keep_far = np.log(0.5)  # sample with probability 50% samples with distance after gamma quantile
    probabilities = np.exp(ln_prob_keep_far / gamma_quantile * distance)
    probabilities[np.isnan(probabilities)] = np.nanmean(probabilities)
    keep_decisions = np.array([np.random.choice(['keep', 'remove'], size=1, p=[p, 1 - p])[0] for p in probabilities])
    keep_indices = np.where(keep_decisions == 'keep')[0]

    repeat_indices = np.random.choice(keep_indices, n_rows, replace=True)
    x = x[repeat_indices, :]
    y = y[repeat_indices]
    return x, y, shift_features


# Subsample feature f with probability p when f<=f_split and 1-p when f>f_split.
def subsample_one_feature_shift(x, y, f, f_split, p=0.5, one_side=True):
    n_rows = x.shape[0]

    smaller_than_split_indices = np.where(x[:, f] <= f_split)[0]
    n_smaller = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(x[:, f] > f_split)[0]
    n_larger = len(larger_than_split_indices)

    keep_smaller_decisions = np.random.choice(['keep', 'remove'], size=n_smaller, p=[p, 1 - p])
    keep_smaller_indices = smaller_than_split_indices[np.where(keep_smaller_decisions == 'keep')[0]]

    if one_side:
        keep_larger_indices = larger_than_split_indices
    else:
        keep_larger_decisions = np.random.choice(['keep', 'remove'], size=n_larger, p=[1 - p, p])
        keep_larger_indices = larger_than_split_indices[np.where(keep_larger_decisions == 'keep')[0]]

    keep_indices = np.hstack([keep_smaller_indices, keep_larger_indices])

    if keep_indices.size == 0:
        return x, y

    repeat_indices = np.random.choice(keep_indices, n_rows, replace=True)
    x = x[repeat_indices, :]
    y = y[repeat_indices]
    return x, y


# Subsample all features with probability p when f<=f_split and 1-p when f>f_split
def subsample_feature_shift(x, y, feat_delta=0.5, p=0.5, one_side=True):
    n_feat_subsample = max(1, ceil(x.shape[1] * feat_delta))
    feat_indices = np.random.choice(x.shape[1], n_feat_subsample, replace=False)

    for f in feat_indices:
        f_split = np.nanmedian(x[:, f])
        if np.isnan(f_split):
            continue
        x, y = subsample_one_feature_shift(x, y, f, f_split, p=p, one_side=one_side)
    return x, y, feat_indices


def split_categorical_feature(x):
    categories = np.unique(x.astype('<U32'))
    categories = np.delete(categories, np.where(categories == 'nan')).astype(object)
    n_half = ceil(0.5 * len(categories))
    choice = np.random.choice(categories.shape[0], size=(n_half,), replace=False)
    half_1 = np.zeros_like(categories, dtype=bool)
    half_1[choice] = True
    half_2 = ~half_1
    return categories[half_1], categories[half_2]


# Subsample categorical feature with probability p when f is in a random selection of categories,
# 1-p when f is in the remaining categories.
def subsample_categorical_feature_shift(x, y, f, p=0.5, return_groups=False, one_side=True):
    n_rows = x.shape[0]
    group_1, group_2 = split_categorical_feature(x[:, f])
    # if column contains only nan
    if group_1.size == 0 & group_2.size == 0:
        if not return_groups:
            return x, y
        else:
            return x, y, group_1, group_2
    is_in_group_1 = lambda val: (val in group_1)
    is_in_group_1_vec = np.vectorize(is_in_group_1)
    is_in_group_2 = lambda val: (val in group_2)
    is_in_group_2_vec = np.vectorize(is_in_group_2)
    smaller_than_split_indices = np.where(is_in_group_1_vec(x[:, f]))[0]
    n_smaller = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(is_in_group_2_vec(x[:, f]))[0]
    n_larger = len(larger_than_split_indices)

    keep_smaller_decisions = np.random.choice(['keep', 'remove'], size=n_smaller, p=[p, 1 - p])
    keep_smaller_indices = smaller_than_split_indices[np.where(keep_smaller_decisions == 'keep')[0]]

    if one_side:
        keep_larger_indices = larger_than_split_indices
    else:
        keep_larger_decisions = np.random.choice(['keep', 'remove'], size=n_larger, p=[1 - p, p])
        keep_larger_indices = larger_than_split_indices[np.where(keep_larger_decisions == 'keep')[0]]

    keep_indices = np.hstack([keep_smaller_indices, keep_larger_indices])
    if keep_indices.size == 0:
        if not return_groups:
            return x, y
        else:
            return x, y, group_1, group_2

    repeat_indices = np.random.choice(keep_indices, n_rows, replace=True)
    x = x[repeat_indices]
    y = y[repeat_indices]

    if not return_groups:
        return x, y
    else:
        return x, y, group_1, group_2


def subsample_all_categorical_feature_shift(x, y, p=0.5, one_side=True):
    for f in range(x.shape[1]):
        x, y = subsample_categorical_feature_shift(x, y, f, p=p, one_side=one_side)
    return x, y


# Gaussian Noise applied on delta portion of samples and feat_delta portion of features
def gaussian_noise_shift(x, y, noise_amt=10., delta=1.0, feat_delta=1.0,
                         clip=True, ceil_int=True):
    x, indices, feat_indices = gaussian_noise_subset(
        x, noise_amt, normalization=1.0, delta=delta, feat_delta=feat_delta, clip=clip, ceil_int=ceil_int)
    return x, y, indices, feat_indices


def gaussian_noise_subset(x, noise_amt, normalization=1.0, delta=1.0, feat_delta=1.0,
                          clip=True, ceil_int=True):
    # precompute for clip and ceil
    int_features = [f for f in range(x.shape[1]) if is_integer_feature(x, f)]
    x_mins = np.min(x, axis=0)
    x_maxs = np.max(x, axis=0)

    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    indices = np.transpose(indices[np.newaxis])

    feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)
    feat_indices = feat_indices[np.newaxis]

    x_mod = x[indices, feat_indices]
    noise = np.random.normal(0, noise_amt / normalization, (x_mod.shape[0], x_mod.shape[1]))
    x_mod = x_mod + noise

    if bool(int_features) & ceil_int:
        int_features = list(set(int_features) & set(feat_indices.flatten()))
        int_features_mapped = [list(feat_indices.flatten()).index(f_idx) for f_idx in int_features]
        x_mod[:, int_features_mapped] = np.ceil(x_mod[:, int_features_mapped])

    if clip:
        x_mod = np.clip(x_mod, x_mins[np.squeeze(feat_indices)], x_maxs[np.squeeze(feat_indices)])

    x[indices, feat_indices] = x_mod
    return x, indices, feat_indices.flatten()


# Set a fraction of samples and features to median constant values (numerical)
def constant_value_shift(x, y, delta=1.0, feat_delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)
    indices = np.transpose(indices[np.newaxis])

    feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)
    feat_indices = feat_indices[np.newaxis]

    x_mod = x[indices, feat_indices]
    med = np.median(x[:, feat_indices], axis=0)
    constant_val = np.repeat(med, x_mod.shape[0], axis=0)

    x[indices, feat_indices] = constant_val

    return x, y, indices.flatten(), feat_indices.flatten()


def switch_categorical_features_shift(x, y, delta=0.5, feat_delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)

    # compute the categories on the whole sample, otherwise might miss some values
    # consider to put as input
    n_categorical = x.shape[1]
    feat_indices = np.random.choice(n_categorical, ceil(n_categorical * feat_delta), replace=False)

    x_mod = x[indices, :]

    for f in feat_indices:
        categories = np.unique(x[:, f].astype('<U32'))
        categories = np.delete(categories, np.where(categories == 'nan')).astype(object)
        switch_val = lambda val: any_other_label(val, categories)
        switch_val_vec = np.vectorize(switch_val)
        x_mod[:, f] = switch_val_vec(x_mod[:, f])

    x[indices, :] = x_mod
    return x, y, indices, feat_indices


# Set a fraction of samples and features to random constant category (categorical)
def constant_categorical_shift(x, y, delta=0.5, feat_delta=1.0):
    indices = np.random.choice(x.shape[0], ceil(x.shape[0] * delta), replace=False)

    # compute the categories on the whole sample, otherwise might miss some values
    # consider to put as input
    n_categorical = x.shape[1]
    feat_indices = np.random.choice(n_categorical, ceil(n_categorical * feat_delta), replace=False)

    for f in feat_indices:
        categories = np.unique(x[:, f].astype('<U32'))
        categories = np.delete(categories, np.where(categories == 'nan')).astype(object)
        x[indices, f] = np.random.choice(categories, size=1)

    return x, y, indices, feat_indices


# Undersample by fraction selecting samples close to the minority class (NearMiss3 heuristics)
def under_sampling_shift_reduced(x, y, delta=0.5):
    labels, y_counts = np.unique(y, return_counts=True)
    y_counts_dict = {label: label_counts for label, label_counts in zip(labels, y_counts)}
    sampling_strategy = dict()
    for label in labels:
        # at least 2 to be able to oversampling via neighbours interpolation later
        sampling_strategy[label] = max(2, int(delta * y_counts_dict[label]))

    # version 3 subsamples respecting more the initial data structure,
    # but it gives less control on the n of final samples (initial resampling phase)
    # thus let use version 2 as the default
    nm1 = NearMiss(version=2, sampling_strategy=sampling_strategy)
    x_resampled, y_resampled = nm1.fit_resample(x, y)

    return x_resampled, y_resampled


# Undersample by fraction selecting samples close to the minority class (NearMiss3 heuristics)
def under_sampling_shift(x, y, delta=0.5):
    n_rows = x.shape[0]

    x_resampled, y_resampled = under_sampling_shift_reduced(x, y, delta)

    repeat_indices = np.random.choice(x_resampled.shape[0], n_rows, replace=True)
    x_resampled = x_resampled[repeat_indices, :]
    y_resampled = y_resampled[repeat_indices]

    y_resampled = y_resampled.reshape((y_resampled.shape[0], 1))

    return x_resampled, y_resampled


# Replace a fraction of samples with samples interpolated from the remaining part
def over_sampling_shift(x, y, delta=0.5, mode='smote', n_neighbors=5):
    assert (mode in ['smote', 'adasyn'])

    labels, y_counts = np.unique(y, return_counts=True)
    y_counts_dict = {label: label_counts for label, label_counts in zip(labels, y_counts)}

    x_resampled, y_resampled = under_sampling_shift_reduced(x, y, delta=delta)

    _, y_resampled_counts = np.unique(y_resampled, return_counts=True)
    n_min_samples = np.min(y_resampled_counts)
    n_neighbors = min(n_neighbors, n_min_samples - 1)

    if mode == 'smote':
        x_resampled, y_resampled = SMOTE(
            sampling_strategy=y_counts_dict, k_neighbors=n_neighbors).fit_resample(x_resampled, y_resampled)
    elif mode == 'adasyn':
        x_resampled, y_resampled = ADASYN(
            sampling_strategy=y_counts_dict, n_neighbors=n_neighbors).fit_resample(x_resampled, y_resampled)

    y_resampled = y_resampled.reshape((y_resampled.shape[0], 1))

    return x_resampled, y_resampled


# non targeted black box adversarial attacks
def adversarial_attack_shift(x, y, delta=1.0, model=RandomForestClassifier(), attack_type='boundary', feat_delta=1.0):
    # in this case delta is the portion of half the data on which to generate attacks
    # because the first half as a minimum has to be used to train a model against which generate the attacks
    assert (attack_type in ['zoo', 'boundary', 'hop-skip-jump'])

    y_final = copy.deepcopy(y)

    le = preprocessing.LabelEncoder()
    le.fit(np.squeeze(y))
    y = le.transform(y)

    indices = list(range(x.shape[0]))
    x_train, x_test, y_train, y_test, indices_train, indices_test = train_test_split(x, y, indices, test_size=0.5)

    n_x_test = len(indices_test)
    indices_test_to_attack = np.random.choice(n_x_test, size=int(np.floor(delta * n_x_test)))

    feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)

    other_features = list(set(range(x.shape[1])) - set(feat_indices))

    x_train_other = x_train[:, other_features]
    x_train_numerical = x_train[:, feat_indices]
    x_test_other = x_test[:, other_features]
    x_test_numerical = x_test[:, feat_indices]

    classifier = SklearnClassifier(model=model, clip_values=(0, np.max(x_train_numerical)))

    # Train the ART classifier

    classifier.fit(x_train_numerical, y_train)

    # Evaluate the ART classifier on benign test examples

    predictions = classifier.predict(x_test_numerical)
    accuracy = np.sum(np.argmax(predictions, axis=1) == y_test) / len(y_test)
    print("Accuracy on benign test examples: {}%".format(accuracy * 100))

    # Generate adversarial test examples
    if attack_type == 'zoo':
        attack = ZooAttack(
            classifier=classifier,
            confidence=0.0,
            targeted=False,
            learning_rate=1e-1,
            max_iter=10,
            binary_search_steps=10,
            initial_const=1e-3,
            abort_early=True,
            use_resize=False,
            use_importance=False,
            nb_parallel=x_test_numerical.shape[1],
            batch_size=1,
            variable_h=0.01,
        )
    elif attack_type == 'boundary':
        attack = BoundaryAttack(classifier, targeted=False, epsilon=0.02, max_iter=20, num_trial=10)
    elif attack_type == 'hop-skip-jump':
        attack = HopSkipJump(classifier,
                             targeted=False,
                             norm=2,
                             max_iter=20,
                             max_eval=10,
                             init_eval=9,
                             init_size=10)

    x_adv = attack.generate(x=x_test_numerical[indices_test_to_attack, :], y=y_test)

    # Evaluate the ART classifier on adversarial test examples
    # x_test_numerical_adversarial = copy.deepcopy(x_test_numerical)
    # x_test_numerical_adversarial[indices_test_to_attack, :] = x_adv

    # predictions_adv = classifier.predict(x_test_numerical_adversarial)
    predictions_adv = classifier.predict(x_adv)
    accuracy = np.sum(np.argmax(predictions_adv, axis=1) == y_test) / len(y_test)
    print("Accuracy on adversarial test examples: {}%".format(accuracy * 100))
    # print("Max difference: {}".format(np.max(
    #    np.abs(x_test_numerical[indices_test_to_attack, :] - x_adv) / x_test_numerical[indices_test_to_attack, :])))

    adv_indices = [indices_test[i] for i in indices_test_to_attack]
    x_final = copy.deepcopy(x)

    adv_row_indices, adv_col_indices = np.transpose(np.array(adv_indices)[np.newaxis]), np.array(feat_indices)[
        np.newaxis]
    x_final[adv_row_indices, adv_col_indices] = x_adv

    return x_final, y_final, adv_indices, feat_indices


# swap two random feature columns for a portion of samples
def swap_random_features(x, corruption_probability=0.2):
    p = [corruption_probability, 1 - corruption_probability]
    (n_samples, n_features) = x.shape
    corrupt_decisions = np.random.choice(['to_corrupt', 'ok'], size=n_samples, p=p)
    corrupt_indices = np.where(corrupt_decisions == 'to_corrupt')[0]
    features_to_switch = np.random.choice(range(n_features), size=2, replace=False)
    tmp = x[corrupt_indices, features_to_switch[0]]
    x[corrupt_indices, features_to_switch[0]] = x[corrupt_indices, features_to_switch[1]]
    x[corrupt_indices, features_to_switch[1]] = tmp

    return x


# consider features as single one-hot encoded variable and activate one feature per sample (set to 1)
# with uniform probability)
def random_set_features_to_uniform(x, corruption_probability=0.2):
    p = [corruption_probability, 1 - corruption_probability]
    (n_samples, n_features) = x.shape
    corrupt_decisions = np.random.choice(['to_corrupt', 'ok'], size=n_samples, p=p)
    corrupt_indices = np.where(corrupt_decisions == 'to_corrupt')[0]
    n_samples_corrupt = corrupt_indices.shape[0]
    p_uniform = 1. / n_features * np.ones((n_features,))
    feature_decisions = np.random.choice(list(range(n_features)), size=n_samples_corrupt, p=p_uniform)
    x[corrupt_indices, :] = np.zeros((n_samples_corrupt, n_features))
    x[tuple([corrupt_indices, feature_decisions])] = 1.
    return x
