from drift_dac.perturbation_shared_utils import *

__all__ = ['apply_shift']


def apply_shift(X_te_orig, y_te_orig, shift, is_categorical=None):

    if is_categorical is None:
        is_categorical = [False] * X_te_orig.shape[1]

    if isinstance(shift, Shift):
        shift_obj = shift
        print(shift.name.replace('_', ' ').title())
    else:
        raise ValueError("Input shift must be a Shift object.")

    X_te_1 = copy.deepcopy(X_te_orig)
    if shift_obj.feature_type == PerturbationConstants.NUMERIC:
        (X_te_1[:, ~is_categorical], y_te_1) = shift_obj.transform(X_te_orig[:, ~is_categorical].astype(float), y_te_orig)
    elif shift_obj.feature_type == PerturbationConstants.CATEGORICAL:
        (X_te_1[:, is_categorical], y_te_1) = shift_obj.transform(X_te_orig[:, is_categorical], y_te_orig)
    else:
        (X_te_1, y_te_1) = shift_obj.transform(X_te_orig, y_te_orig)

    indices, feat_indices = shift_obj.shifted_indices, shift_obj.shifted_features

    return (X_te_1, y_te_1), indices, feat_indices
