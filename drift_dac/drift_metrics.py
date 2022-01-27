import scipy.stats as stats
import scipy.sparse as sparse
import numpy as np
from drift_dac.domain_classifier import DomainClassifier, DomainClassifierModel
from sklearn.base import clone
from sklearn.calibration import CalibratedClassifierCV
import random


def test_chi2(y_pred_src, y_pred_tar, nb_classes):
    # Calculate observed and expected counts
    freq_exp = np.zeros(nb_classes)
    freq_obs = np.zeros(nb_classes)

    unique_tr, counts_tr = np.unique(y_pred_src, return_counts=True)
    unique_te, counts_te = np.unique(y_pred_tar, return_counts=True)

    for i in range(len(unique_tr)):
        freq_exp[i] = counts_tr[i]

    for i in range(len(unique_te)):
        freq_obs[np.where(unique_tr == unique_te[i])[0]] = counts_te[i]

    if np.amin(freq_exp) < 5 or np.amin(freq_obs) < 5:
        # The chi-squared test using contingency tables is not well defined if less than 5 samples per category exist,
        # which might happen in the low-sample regime.
        t = None
        p_val = random.uniform(0, 1)
    else:
        # In almost all cases, we resort to obtaining a p-value from the chi-squared test's contingency table.
        freq_conc = np.array([freq_exp, freq_obs])
        t, p_val, _, _ = stats.chi2_contingency(freq_conc)

    return t, p_val


def prediction_drift(predictions_src, predictions_tar):
    y_predicted_src = np.argmax(predictions_src, axis=1)
    y_predicted_tar = np.argmax(predictions_tar, axis=1)
    nb_classes = predictions_src.shape[1]

    bbse_soft_metric = 1.
    bbse_hard_metric = 1.
    bbse_soft_stat = 1.
    bbse_hard_stat = 1.

    for i in range(nb_classes):
        bbse_soft_t, bbse_soft_p_val = stats.ks_2samp(predictions_src[:, i], predictions_tar[:, i])
        if bbse_soft_p_val < bbse_soft_metric:
            bbse_soft_metric = bbse_soft_p_val
            bbse_soft_stat = bbse_soft_t

        bbse_hard_t, bbse_hard_p_val = test_chi2(y_predicted_src, y_predicted_tar, nb_classes)
        if bbse_hard_p_val < bbse_hard_metric:
            bbse_hard_metric = bbse_hard_p_val
            bbse_hard_stat = bbse_hard_t

    # scale to use the same significance level as for single test
    bbse_soft_metric = min(bbse_soft_metric * nb_classes, 1.0)
    bbse_hard_metric = min(bbse_hard_metric * nb_classes, 1.0)

    return bbse_soft_stat, bbse_soft_metric, bbse_hard_stat, bbse_hard_metric


def univariate_drift(X_src, X_tar, is_categorical=None):
    n_features = X_src.shape[1]
    univariate_t = []
    univariate_p_val = []

    if is_categorical is None:
        is_categorical = [False] * n_features

    for i, is_cat in enumerate(is_categorical):

        if is_cat:
            f_src = X_src[:, i].astype(str)
            f_src = f_src[np.where(f_src != 'nan')[0]]
            f_tar = X_tar[:, i].astype(str)
            f_tar = f_tar[np.where(f_tar != 'nan')[0]]

            if f_tar.size == 0 or f_src.size == 0:
                x_univ_t, x_univ_p_val = None, None
            else:
                nb_categories = np.unique(f_src).shape[0]
                x_univ_t, x_univ_p_val = test_chi2(f_src, f_tar, nb_categories)
        else:
            f_src = X_src[:, i].astype(float)
            f_src = f_src[np.logical_not(np.isnan(f_src))]
            f_tar = X_tar[:, i].astype(float)
            f_tar = f_tar[np.logical_not(np.isnan(f_tar))]

            if f_tar.size == 0 or f_src.size == 0:
                x_univ_t, x_univ_p_val = None, None
            else:
                x_univ_t, x_univ_p_val = stats.ks_2samp(f_src, f_tar)

        univariate_t.append(x_univ_t)
        univariate_p_val.append(x_univ_p_val)

    return univariate_t, univariate_p_val


def domain_classifier_accuracy(X_src, X_tar, dc_model=DomainClassifierModel.RF, return_model=False):
    # Characterize shift via domain classifier.
    orig_dims = X_src.shape[1]
    domain_clf = DomainClassifier(orig_dims, dc=dc_model)
    model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = \
        domain_clf.build_model([X_src, X_tar])

    p_val, _, _, _ = domain_clf.accuracy_binomial_test(model, X_te_dcl, y_te_dcl)

    if return_model:
        return score, p_val, model
    else:
        return score, p_val


def confidence_drop(predictions_src, predictions_tar):
    avg_confidence_src = np.mean(np.max(predictions_src, axis=1))
    avg_confidence_tar = np.mean(np.max(predictions_tar, axis=1))

    conf_drop = avg_confidence_tar - avg_confidence_src
    return conf_drop


def reverse_classification_accuracy(X_src, y_src, predictions_src, X_tar, predictions_tar, model):
    nb_classes = predictions_src.shape[1]

    y_src_pred = np.argmax(predictions_src, axis=1)
    direct_score = np.mean(y_src_pred == y_src)

    pseudo_labels = np.argmax(predictions_tar, axis=1)

    # if there are not pseudo labels for each class, skip rsa computation
    if len(np.unique(pseudo_labels)) == nb_classes:
        if isinstance(model, CalibratedClassifierCV):
            reverse_model = clone(model.calibrated_classifiers_[0].base_estimator)
        else:
            reverse_model = clone(model)
        reverse_model.fit(X_tar, pseudo_labels)
        reverse_score = reverse_model.score(X_src, y_src)

        rca = reverse_score - direct_score
    else:
        rca = None

    return rca


def compute_drift_metrics(X_src, y_src, predictions_src, X_tar, predictions_tar, model,
                          dc_model=DomainClassifierModel.RF, return_dc_model=False, preprocessor=None,
                          is_categorical=None):
    bbse_metrics = prediction_drift(predictions_src, predictions_tar)
    bbse_soft_stat, bbse_soft_metric, bbse_hard_stat, bbse_hard_metric = bbse_metrics

    if preprocessor is None:
        X_src_pproc = X_src
        X_tar_pproc = X_tar
    else:
        X_src_pproc = preprocessor.transform(X_src)
        X_tar_pproc = preprocessor.transform(X_tar)
        if sparse.issparse(X_src_pproc):
            X_src_pproc = X_src_pproc.todense()
        if sparse.issparse(X_tar_pproc):
            X_tar_pproc = X_tar_pproc.todense()

    dc_out = domain_classifier_accuracy(X_src_pproc, X_tar_pproc, dc_model, return_dc_model)
    if return_dc_model:
        dc_acc, dc_p_val, dc_model = dc_out
    else:
        dc_acc, dc_p_val = dc_out

    conf_drop = confidence_drop(predictions_src, predictions_tar)

    rca = reverse_classification_accuracy(X_src_pproc, y_src, predictions_src, X_tar_pproc, predictions_tar, model)

    univariate_t, univariate_p_val = univariate_drift(X_src, X_tar, is_categorical)

    metrics = dict()
    metrics['bbsd_soft_t'] = bbse_soft_stat
    metrics['bbsd_soft_p_val'] = bbse_soft_metric
    metrics['bbsd_hard_t'] = bbse_hard_stat
    metrics['bbsd_hard_p_val'] = bbse_hard_metric
    metrics['dc_acc'] = dc_acc
    metrics['dc_p_val'] = dc_p_val
    metrics['confidence_drop'] = conf_drop
    metrics['rca'] = rca
    metrics['x_univ_t'] = univariate_t
    metrics['x_univ_p_val'] = univariate_p_val

    if return_dc_model:
        return metrics, dc_model
    else:
        return metrics
