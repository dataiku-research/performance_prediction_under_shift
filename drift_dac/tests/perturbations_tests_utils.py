from drift_dac.concept_shift import *
from drift_dac.prior_shift import *
from drift_dac.covariate_shift import *
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
import numpy as np
from math import ceil

from drift_dac.covariate_shift import subsample_one_feature_shift
from drift_dac.covariate_shift import subsample_categorical_feature_shift

TEST_TOLERANCE = 1e-1


def generate_synthetic_data(n_samples=1000,
                            n_num_features=5,
                            n_cat_features=3,
                            categories=['A', 'B', 'C'],
                            n_classes=2):
    rng = np.random.RandomState(0)
    X_num, y = make_classification(n_samples=n_samples, n_features=n_num_features, n_classes=n_classes,
                                   random_state=rng)

    X_cat = np.random.choice(categories, size=(n_samples, n_cat_features))

    X = np.hstack([X_num.astype(object), X_cat.astype(object)])

    is_categorical = np.array(n_num_features*[False] + n_cat_features*[True])

    return X, y, is_categorical


def generate_synthetic_text_data():

    x = np.array([['What I cannot create, I do not understand.', 'What I cannot create.'],
                  ['Oneword', 'Two words']])
    x = np.tile(x, (10, 1))
    y = np.random.choice([0, 1], size=x.shape[0])
    is_text = np.array([True, True])
    return x, y, is_text


def check_replace_word_shift(x, y, delta=0.5, pct_words_to_swap=0.1):
    x_prev = x.copy()

    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')

    x_new, y_new = ReplaceWord(delta, pct_words_to_swap).transform(x, y)
    print(x_new)

    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)

    return


def check_typos_shift(x, y, delta=0.5, pct_words_to_swap=0.1):
    x_prev = x.copy()

    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')

    x_new, y_new = Typos(delta, pct_words_to_swap).transform(x, y)
    print(x_new)

    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)

    return


def check_delete_word_shift(x, y, delta=0.5):
    x_prev = x.copy()

    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')

    x_new, y_new = WordDeletion(delta).transform(x, y)
    print(x_new)

    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)

    return
  

def check_random_class_subset_shift(x, y, perc_max_changes=0.3):
    y_prev = y.copy()
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')

    x_new, y_new = LabelSwitch(perc_max_changes).transform(x, y)

    n_changed = np.count_nonzero(y_new != y_prev)
    perc_changed = float(n_changed) / y.shape[0]
    perc_non_changed = 1 - perc_changed

    print(f'perc_max_changes: {perc_max_changes}')
    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')
    print(f'perc_non_changed: {perc_non_changed}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    assert (perc_changed == perc_max_changes)

    return


def check_rebalance_shift(x, y, priors):
    labels, counts = np.unique(y, return_counts=True)
    priors_prev = counts / y.shape[0]
    n_labels = len(labels)
    print(f'labels: {labels}')
    print(f'priors: {priors_prev}')
    print(f'counts: {counts}')

    x_new, y_new = Rebalance(priors).transform(x, y)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    n_labels_new = len(labels_new)

    n_removed_labels = np.count_nonzero(np.array(priors) == 0.)

    print(f'desired_priors: {priors}')

    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    padded_priors_new = []
    count = 0
    for lab in labels:
        if lab not in labels_new:
            padded_priors_new.append(0)
        else:
            padded_priors_new.append(priors_new[count])
            count += 1
    padded_priors_new = np.array(padded_priors_new)

    assert (np.max(np.fabs(padded_priors_new - priors)) < TEST_TOLERANCE)
    assert ((n_labels - n_labels_new) == n_removed_labels)

    return


def check_knockout_shift(x, y, cl, delta):
    labels, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    priors_prev = counts / n_samples
    n_labels = len(labels)
    n_cl = len(np.where(y == cl)[0])

    x_new, y_new = KnockOut(cl, delta).transform(x, y)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    n_samples_new = y_new.shape[0]
    priors_new = counts_new / n_samples_new
    n_labels_new = len(labels_new)

    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors_prev}')
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(priors_new[cl] - delta * priors_prev[cl]) < TEST_TOLERANCE)
    assert (n_samples_new == n_samples)

    return


def check_only_one_shift(x, y, keep_cl):
    labels, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    priors_prev = counts / n_samples
    n_labels = len(labels)
    n_cl = len(np.where(y == keep_cl)[0])

    x_new, y_new = OnlyOne(keep_cl).transform(x, y)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    n_samples_new = y_new.shape[0]
    priors_new = counts_new / n_samples_new
    n_labels_new = len(labels_new)

    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors_prev}')
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    expected_priors_new_keep_cl = 1.

    assert (n_labels_new == 1)
    assert (labels_new[0] == keep_cl)
    assert (np.fabs(expected_priors_new_keep_cl - priors_new[keep_cl]) < TEST_TOLERANCE)
    assert (n_samples_new == n_samples)

    return


def check_subsample_joint_shift(x, y, gamma=0.8, shift_features=None):
    x_new, y_new = SubsampleJoint(gamma=gamma).transform(x, y)

    labels, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    priors_prev = counts / y.shape[0]
    n_labels = len(labels)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    n_samples_new = y_new.shape[0]
    priors_new = counts_new / y_new.shape[0]
    n_labels_new = len(labels_new)

    print(f'n_samples: {n_samples}')
    print(f'labels: {labels}')
    print(f'n_labels: {n_labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors_prev}')
    print(f'n_samples_new: {n_samples_new}')
    print(f'labels_new: {labels_new}')
    print(f'n_labels_new: {n_labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    assert (n_labels == n_labels_new)

    return


def check_subsample_one_feature_shift(x, y, f, f_split, p=0.5, one_side=True):
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]

    x_new, y_new = subsample_one_feature_shift(x, y, f, f_split, p=p, one_side=one_side)

    n_samples = y.shape[0]
    smaller_than_split_indices = np.where(x[:, f] <= f_split)[0]
    n_smaller = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(x[:, f] > f_split)[0]
    n_larger = len(larger_than_split_indices)

    n_samples_new = y_new.shape[0]
    smaller_than_split_indices_new = np.where(x_new[:, f] <= f_split)[0]
    n_smaller_new = len(smaller_than_split_indices_new)
    larger_than_split_indices_new = np.where(x_new[:, f] > f_split)[0]
    n_larger_new = len(larger_than_split_indices_new)

    print(f'feature: {f}')
    print(f'f_split: {f_split}')
    print(f'n_samples: {n_samples}')
    print(f'n_smaller: {n_smaller}')
    print(f'n_larger: {n_larger}')
    print(f'n_samples_new: {n_samples_new}')
    print(f'n_smaller_new: {n_smaller_new}')
    print(f'n_larger_new: {n_larger_new}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    perc_smaller = float(n_smaller) / n_samples
    perc_larger = float(n_larger) / n_samples

    perc_smaller_new = float(n_smaller_new) / n_samples_new
    perc_larger_new = float(n_larger_new) / n_samples_new

    print(f'perc_smaller_new: {perc_smaller_new}')
    print(f'perc_larger_new: {perc_larger_new}')

    if one_side:
        expected_perc_smaller_new = p * perc_smaller / (p * perc_smaller + perc_larger)
        print(f'expected_perc_smaller_new: {expected_perc_smaller_new}')

        assert (np.fabs(perc_smaller_new - expected_perc_smaller_new) < TEST_TOLERANCE)
    else:
        expected_perc_smaller_new = p * perc_smaller / (p * perc_smaller + (1-p) * perc_larger)
        expected_perc_larger_new = 1 - expected_perc_smaller_new
        print(f'expected_perc_larger_new: {expected_perc_larger_new}')

        assert (np.fabs(perc_larger_new - expected_perc_larger_new) < TEST_TOLERANCE)

    return x_new, y_new


def check_subsample_feature_shift(x, y, feat_delta=0.5, p=0.5, numerical_features=None, one_side=True):
    if numerical_features is not None:
        n_numerical = len(numerical_features)
        feat_indices = np.random.choice(n_numerical, ceil(n_numerical * feat_delta), replace=False)
        feat_indices = np.array(numerical_features)[feat_indices]
    else:
        feat_indices = np.random.choice(x.shape[1], ceil(x.shape[1] * feat_delta), replace=False)

    for f in feat_indices:
        f_split = np.median(x[:, f])
        x, y = check_subsample_one_feature_shift(x, y, f, f_split, p=p, one_side=one_side)

    return x, y


def check_subsample_categorical_feature_shift(x, y, f, p=0.5, one_side=True):
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]

    x_new, y_new, group_1, group_2 = subsample_categorical_feature_shift(x, y, f, p=p,
                                                                         return_groups=True,
                                                                         one_side=one_side)

    n_samples = y.shape[0]
    is_in_group_1 = lambda val: (val in group_1)
    is_in_group_1_vec = np.vectorize(is_in_group_1)
    is_in_group_2 = lambda val: (val in group_2)
    is_in_group_2_vec = np.vectorize(is_in_group_2)

    smaller_than_split_indices = np.where(is_in_group_1_vec(x[:, f]))[0]
    n_smaller = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(is_in_group_2_vec(x[:, f]))[0]
    n_larger = len(larger_than_split_indices)

    n_samples_new = y_new.shape[0]
    smaller_than_split_indices = np.where(is_in_group_1_vec(x_new[:, f]))[0]
    n_smaller_new = len(smaller_than_split_indices)
    larger_than_split_indices = np.where(is_in_group_2_vec(x_new[:, f]))[0]
    n_larger_new = len(larger_than_split_indices)

    print(f'n_samples: {n_samples}')
    print(f'n_smaller: {n_smaller}')
    print(f'n_larger: {n_larger}')
    print(f'n_samples_new: {n_samples_new}')
    print(f'n_smaller_new: {n_smaller_new}')
    print(f'n_larger_new: {n_larger_new}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    perc_smaller = float(n_smaller) / n_samples
    perc_larger = float(n_larger) / n_samples

    perc_smaller_new = float(n_smaller_new) / n_samples_new
    perc_larger_new = float(n_larger_new) / n_samples_new

    print(f'perc_smaller_new: {perc_smaller_new}')
    print(f'perc_larger_new: {perc_larger_new}')

    if one_side:
        expected_perc_smaller_new = p * perc_smaller / (p * perc_smaller + perc_larger)
        print(f'expected_perc_smaller_new: {expected_perc_smaller_new}')

        assert (np.fabs(perc_smaller_new - expected_perc_smaller_new) < TEST_TOLERANCE)
    else:
        expected_perc_smaller_new = p * perc_smaller / (p * perc_smaller + (1 - p) * perc_larger)
        expected_perc_larger_new = 1 - expected_perc_smaller_new
        print(f'expected_perc_larger_new: {expected_perc_larger_new}')

        assert (np.fabs(perc_larger_new - expected_perc_larger_new) < TEST_TOLERANCE)

    return x_new, y_new


def check_subsample_all_categorical_feature_shift(x, y, p=0.5):
    for f in range(x.shape[1]):
        x, y = check_subsample_categorical_feature_shift(x, y, f, p=p)
    return


def check_subsample_all_feature_shift(x, y, p=0.5, is_categorical=None, one_side=True):

    x[:, ~is_categorical], y = check_subsample_feature_shift(x[:, ~is_categorical].astype(float), y, feat_delta=1.0,
                                                             p=p, one_side=one_side)
    if np.any(is_categorical):
        check_subsample_all_categorical_feature_shift(x[:, is_categorical], y, p=p)

    return


def check_gaussian_noise_shift(x, y, noise_key='medium', delta=1.0, feat_delta=1.0, clip=True, ceil_int=True):
    x_prev = x.copy()
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'priors: {priors}')

    x_new, y_new = GaussianNoise(
        noise_key=noise_key,
        samples_fraction=delta,
        features_fraction=feat_delta,
        clip=clip,
        ceil_int=ceil_int
    ).transform(x, y)
    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    n_feat_changed = np.count_nonzero(np.any(x_new != x_prev, axis=0))
    print(f'n_feat_changed: {n_feat_changed}')

    n_features = x.shape[1]

    n_feat_expected_to_change = ceil(n_features * feat_delta)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)
    assert (n_feat_changed == n_feat_expected_to_change)

    return


def check_adversarial_shift(x, y, delta=1.0, feat_delta=1.0, attack_type='boundary', model=RandomForestClassifier()):
    x_prev = x.copy()
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'priors: {priors}')

    x_new, y_new = Adversarial(
        samples_fraction=delta,
        features_fraction=feat_delta,
        attack_type=attack_type,
        model=model
    ).transform(x, y)
    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    n_feat_changed = np.count_nonzero(np.any(x_new != x_prev, axis=0))
    print(f'n_feat_changed: {n_feat_changed}')

    n_features = x.shape[1]

    n_feat_expected_to_change = ceil(n_features * feat_delta)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - (delta * 0.5)) < TEST_TOLERANCE)
    assert (n_feat_changed == n_feat_expected_to_change)

    return


def check_switch_categorical_features_shift(x, y, delta=1.0, feat_delta=1.0):
    x_prev = x.copy()
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'priors: {priors}')

    x_new, y_new = SwitchCategorical(
        samples_fraction=delta,
        features_fraction=feat_delta,
    ).transform(x, y)
    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    n_categorical = x.shape[1]
    n_expected_to_change = ceil(n_categorical * feat_delta)
    n_feat_changed = np.count_nonzero(np.any(x_new != x_prev, axis=0))

    print(f'n_feat_changed: {n_feat_changed}')
    print(f'n_expected_to_change: {n_expected_to_change}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)
    assert (n_feat_changed == n_expected_to_change)

    return


def check_under_sampling_shift(x, y, delta=0.5):
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')

    x_new, y_new = UnderSample(delta).transform(x, y)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    assert (np.all(np.fabs(priors_new - priors) < TEST_TOLERANCE))
    assert (counts.sum() == counts_new.sum())

    return


def check_over_sampling_shift(x, y, delta=0.5):

    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors}')

    x_new, y_new = OverSample(delta).transform(x, y)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    assert (np.all(counts_new == counts))

    # This check should be done, but we lose information of the permutaiton on x
    # thus the input and output rows do not correspond
    # The whole data will result as changed
    #n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    #perc_changed = float(n_changed) / x.shape[0]

    #print(f'n_changed: {n_changed}')
    #print(f'perc_changed: {perc_changed}')

    #assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)

    return


def check_constant_shift(x, y, delta=1.0, feat_delta=1.0):
    x_prev = x.copy()
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'priors: {priors}')

    shift = ConstantNumeric(
        samples_fraction=delta,
        features_fraction=feat_delta,
    )
    x_new, y_new = shift.transform(x, y)
    shifted_features = shift.shifted_features
    shifted_indices = shift.shifted_indices

    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    n_categorical = x.shape[1]
    n_expected_to_change = ceil(n_categorical * feat_delta)
    n_feat_changed = np.count_nonzero(np.any(x_new != x_prev, axis=0))

    print(f'n_feat_changed: {n_feat_changed}')
    print(f'n_expected_to_change: {n_expected_to_change}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)
    assert (n_feat_changed == n_expected_to_change)

    for f in shifted_features:
        unique_values = np.unique(x_new[shifted_indices, f])
        assert(len(unique_values) == 1)

    return


def check_constant_categorical_shift(x, y, delta=1.0, feat_delta=1.0):
    x_prev = x.copy()
    labels, counts = np.unique(y, return_counts=True)
    priors = counts / y.shape[0]
    print(f'labels: {labels}')
    print(f'priors: {priors}')

    shift = ConstantCategorical(
        samples_fraction=delta,
        features_fraction=feat_delta,
    )
    x_new, y_new = shift.transform(x, y)
    shifted_features = shift.shifted_features
    shifted_indices = shift.shifted_indices

    n_changed = np.count_nonzero(np.any(x_new != x_prev, axis=1))
    perc_changed = float(n_changed) / x.shape[0]

    print(f'n_changed: {n_changed}')
    print(f'perc_changed: {perc_changed}')

    n_categorical = x.shape[1]
    n_expected_to_change = ceil(n_categorical * feat_delta)
    n_feat_changed = np.count_nonzero(np.any(x_new != x_prev, axis=0))

    print(f'n_feat_changed: {n_feat_changed}')
    print(f'n_expected_to_change: {n_expected_to_change}')

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    priors_new = counts_new / y_new.shape[0]
    print(f'labels_new: {labels_new}')
    print(f'priors_new: {priors_new}')

    assert (np.fabs(perc_changed - delta) < TEST_TOLERANCE)
    assert (n_feat_changed == n_expected_to_change)

    for f in shifted_features:
        unique_values = np.unique(x_new[shifted_indices, f])
        assert(len(unique_values) == 1)

    return


def check_error_sampling_shift(x, y, error_fraction=0.5):
    labels, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    priors_prev = counts / n_samples

    model = RandomForestClassifier(n_estimators=2).fit(x, y)
    y_pred = model.predict(x)
    n_errors = np.sum(y_pred != y)

    x_new, y_new = ErrorBasedSampling(error_fraction, model).transform(x, y)

    y_new_pred = model.predict(x_new)

    error_rate = 1 - accuracy_score(y_new, y_new_pred)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    n_samples_new = y_new.shape[0]
    priors_new = counts_new / n_samples_new

    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors_prev}')
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')
    print(f'n_errors: {n_errors}')
    print(f'n_errors_new: {int(error_rate * n_samples)}')

    assert ((error_rate - error_fraction) < TEST_TOLERANCE)
    assert (n_samples_new == n_samples)

    return


def check_nearest_neighbors(x, y, fraction_to_remove=0.5, near_far_probability=0.5):
    labels, counts = np.unique(y, return_counts=True)
    n_samples = y.shape[0]
    priors_prev = counts / n_samples

    x_new, y_new = NearestNeighbors(fraction_to_remove, near_far_probability).transform(x, y)

    labels_new, counts_new = np.unique(y_new, return_counts=True)
    n_samples_new = y_new.shape[0]
    priors_new = counts_new / n_samples_new

    print(f'labels: {labels}')
    print(f'counts: {counts}')
    print(f'priors: {priors_prev}')
    print(f'labels_new: {labels_new}')
    print(f'counts_new: {counts_new}')
    print(f'priors_new: {priors_new}')

    assert (n_samples_new == n_samples)

    # rm duplicates and check fraction_to_remove has been removed
    unique_rows = np.unique(x_new, axis=0)
    # not strict equality as initial x could contain duplicates itself
    assert(unique_rows.shape[0] == int(np.ceil(fraction_to_remove * x.shape[0])))

    return
