from drift_dac.tests.perturbations_tests_utils import *
from drift_dac.perturbation_shared_utils import PerturbationConstants
from drift_dac.drift_metrics import compute_drift_metrics
from drift_dac.domain_classifier import DomainClassifier, MultiDomainClassifier, DomainClassifierModel

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix
import scipy.sparse as sparse

import copy


def test_perturbation_random_class_subset():
    x, y, is_categorical = generate_synthetic_data()
    check_random_class_subset_shift(x, y, perc_max_changes=0.3)


def test_perturbation_rebalance():
    x, y, is_categorical = generate_synthetic_data()
    n_classes = len(np.unique(y))
    desired_priors = np.random.rand(n_classes)
    desired_priors /= desired_priors.sum()
    check_rebalance_shift(x, y, priors=desired_priors)


def test_perturbation_knockout():
    x, y, is_categorical = generate_synthetic_data()
    check_knockout_shift(x, y, cl=0, delta=0.5)


def test_perturbation_only_one():
    x, y, is_categorical = generate_synthetic_data()
    check_only_one_shift(x, y, keep_cl=0)


def test_perturbation_subsample_joint():
    x, y, is_categorical = generate_synthetic_data()
    check_subsample_joint_shift(x[:, ~is_categorical].astype(float), y, gamma=0.8, shift_features=None)


def test_perturbation_subsample_one_feature():
    x, y, is_categorical = generate_synthetic_data()
    f = np.where(~is_categorical)[0][0]
    f_split = np.median(x[:, f].astype(float))
    check_subsample_one_feature_shift(x[:, ~is_categorical].astype(float), y, f, f_split, p=0.5, one_side=True)


def test_perturbation_subsample_feature():
    x, y, is_categorical = generate_synthetic_data()
    check_subsample_feature_shift(x[:, ~is_categorical].astype(float), y, feat_delta=0.5, p=0.5, one_side=True)


def test_perturbation_subsample_categorical_feature():
    x, y, is_categorical = generate_synthetic_data()
    f = np.where(is_categorical)[0][0]
    check_subsample_categorical_feature_shift(x, y, f, p=0.5, one_side=True)


def test_perturbation_subsample_all_categorical_feature():
    x, y, is_categorical = generate_synthetic_data()
    check_subsample_all_categorical_feature_shift(x[:, is_categorical], y, p=0.5)


def test_perturbation_subsample_all_feature():
    x, y, is_categorical = generate_synthetic_data()
    check_subsample_all_feature_shift(x, y, p=0.5, is_categorical=is_categorical, one_side=True)


def test_perturbation_gaussian_noise():
    x, y, is_categorical = generate_synthetic_data()
    check_gaussian_noise_shift(x[:, ~is_categorical].astype(float), y, noise_key='medium', delta=1.0, feat_delta=1.0,
                               clip=True, ceil_int=True)


def test_perturbation_switch_categorical_features():
    x, y, is_categorical = generate_synthetic_data()
    check_switch_categorical_features_shift(x[:, is_categorical], y, delta=1.0, feat_delta=1.0)


def test_perturbation_under_sampling():
    x, y, is_categorical = generate_synthetic_data()
    check_under_sampling_shift(x[:, ~is_categorical].astype(float), y, delta=0.5)


def test_perturbation_over_sampling():
    x, y, is_categorical = generate_synthetic_data()
    check_over_sampling_shift(x[:, ~is_categorical].astype(float), y, delta=0.5)


def test_perturbation_constant_numeric():
    x, y, is_categorical = generate_synthetic_data()
    check_constant_shift(x[:, ~is_categorical].astype(float), y, delta=0.5, feat_delta=0.5)


def test_perturbation_constant_categorical():
    x, y, is_categorical = generate_synthetic_data()
    check_constant_categorical_shift(x[:, is_categorical], y, delta=0.5, feat_delta=0.5)


def test_perturbation_adversarial():
    x, y, is_categorical = generate_synthetic_data()
    y[:len(y)-1] = 0
    print(np.unique(y, return_counts=True))
    check_adversarial_shift(x[:, ~is_categorical].astype(float), y, delta=0.25, feat_delta=1.0, attack_type='boundary')


def test_perturbation_error_sampling():
    x, y, is_categorical = generate_synthetic_data()
    check_error_sampling_shift(x[:, ~is_categorical].astype(float), y, error_fraction=0.5)


def test_perturbation_replace_word():
    x, y, is_text = generate_synthetic_text_data()
    check_replace_word_shift(x[:, is_text], y, delta=1., pct_words_to_swap=1.)

    
def test_perturbation_typos_shift():
    x, y, is_text = generate_synthetic_text_data()
    check_typos_shift(x[:, is_text], y, delta=1., pct_words_to_swap=1.)


def test_perturbation_delete_word():
    x, y, is_text = generate_synthetic_text_data()
    check_delete_word_shift(x[:, is_text], y, delta=0.5)


def test_perturbation_nearest_neighbors():
    x, y, is_categorical = generate_synthetic_data()
    check_nearest_neighbors(x[:, ~is_categorical].astype(float), y, fraction_to_remove=0.5, near_far_probability=0.5)
    

def test_drift_metrics():
    X, y, is_categorical = generate_synthetic_data()

    X_src, X_tar, y_src, y_tar = train_test_split(X, y, test_size=0.5)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, np.where(is_categorical)[0]),
            ('num', numeric_transformer, np.where(~is_categorical)[0])])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_src, y_src)
    predictions_src = model.predict_proba(X_src)

    list_of_shifts = [ConstantCategorical(0.5, 0.5), FlipSign(0.5, 0.5)]

    for shift in list_of_shifts:
        X_perturbed = copy.deepcopy(X_tar)
        if shift.feature_type == PerturbationConstants.NUMERIC:
            (X_perturbed[:, ~is_categorical], y_perturbed) = shift.transform(X_tar[:, ~is_categorical].astype(float),
                                                                             y_tar)
        elif shift.feature_type == PerturbationConstants.CATEGORICAL:
            (X_perturbed[:, is_categorical], y_perturbed) = shift.transform(X_tar[:, is_categorical], y_tar)
        else:
            (X_perturbed, y_perturbed) = shift.transform(X_tar, y_tar)

        predictions_tar = model.predict_proba(X_perturbed)
        drift_metrics = compute_drift_metrics(X_src, y_src, predictions_src,
                                              X_perturbed, predictions_tar,
                                              model=model.steps[1][1], preprocessor=model.steps[0][1],
                                              is_categorical=is_categorical)

        print(shift.name)
        print(drift_metrics)


def test_domain_classifier():
    X, y, is_categorical = generate_synthetic_data()

    X_src, X_tar, y_src, y_tar = train_test_split(X, y, test_size=0.5)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, np.where(is_categorical)[0]),
            ('num', numeric_transformer, np.where(~is_categorical)[0])])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_src, y_src)

    shift = GaussianNoise('medium', 0.8)

    X_perturbed = copy.deepcopy(X_tar)
    (X_perturbed[:, ~is_categorical], y_perturbed) = shift.transform(X_tar[:, ~is_categorical].astype(float), y_tar)

    X_src_pproc = preprocessor.transform(X_src)
    X_tar_pproc = preprocessor.transform(X_tar)
    X_perturbed_pproc = preprocessor.transform(X_perturbed)
    if sparse.issparse(X_src_pproc):
        X_src_pproc = X_src_pproc.todense()
    if sparse.issparse(X_tar_pproc):
        X_tar_pproc = X_tar_pproc.todense()
    if sparse.issparse(X_perturbed_pproc):
        X_perturbed_pproc = X_perturbed_pproc.todense()

    orig_dims = X_src.shape[1]
    domain_clf = DomainClassifier(orig_dims, dc=DomainClassifierModel.RF)

    model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = \
        domain_clf.build_model([X_src_pproc, X_tar_pproc], [y_src, y_tar])
    p_val, _, _, _ = domain_clf.accuracy_binomial_test(model, X_te_dcl, y_te_dcl)

    print("no shift")
    print("DC accuracy: %.2f" % score)
    print("DC p-value: %.2f" % p_val)

    model, score, (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = \
        domain_clf.build_model([X_src_pproc, X_perturbed_pproc], [y_src, y_perturbed])
    p_val, _, _, _ = domain_clf.accuracy_binomial_test(model, X_te_dcl, y_te_dcl)

    print(shift.name)
    print("DC accuracy: %.2f" % score)
    print("DC p-value: %.2f" % p_val)


def test_multi_domain_classifier():
    X, y, is_categorical = generate_synthetic_data()

    X_src, X_tar, y_src, y_tar = train_test_split(X, y, test_size=0.5)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, np.where(is_categorical)[0]),
            ('num', numeric_transformer, np.where(~is_categorical)[0])])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_src, y_src)

    X_src_pproc = preprocessor.transform(X_src)
    if sparse.issparse(X_src_pproc):
        X_src_pproc = X_src_pproc.todense()

    list_of_shift = [GaussianNoise('medium', 0.8), FlipSign(0.8), SwappedValues(0.8)]

    list_of_domains_X = [X_src_pproc]
    list_of_domains_y = [y_src]
    print(" ")
    for shift in list_of_shift:
        print(shift.name)
        X_perturbed = copy.deepcopy(X_tar)
        if shift.feature_type == PerturbationConstants.NUMERIC:
            (X_perturbed[:, ~is_categorical], y_perturbed) = shift.transform(X_tar[:, ~is_categorical].astype(float),
                                                                             y_tar)
        elif shift.feature_type == PerturbationConstants.CATEGORICAL:
            (X_perturbed[:, is_categorical], y_perturbed) = shift.transform(X_tar[:, is_categorical], y_tar)
        else:
            (X_perturbed, y_perturbed) = shift.transform(X_tar, y_tar)

        X_perturbed_pproc = preprocessor.transform(X_perturbed)
        if sparse.issparse(X_perturbed_pproc):
            X_perturbed_pproc = X_perturbed_pproc.todense()

        list_of_domains_X.append(X_perturbed_pproc)
        list_of_domains_y.append(y_perturbed)

    orig_dims = X_src.shape[1]
    domain_clf = MultiDomainClassifier(orig_dims, dc=DomainClassifierModel.RF)

    model, score, train_test_data = domain_clf.build_model(list_of_domains_X, list_of_domains_y)
    (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = train_test_data

    y_te_dcl_pred = model.predict(X_te_dcl)

    print("DC reference accuracy: %.2f" % domain_clf.ratio)
    print("DC observed  accuracy: %.2f" % score)
    print("Confusion Matrix")
    print(confusion_matrix(y_te_dcl, y_te_dcl_pred))


def test_multi_domain_classifier_without_target():
    X, y, is_categorical = generate_synthetic_data()

    X_src, X_tar, y_src, y_tar = train_test_split(X, y, test_size=0.5)

    numeric_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='median')),
        ('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
        ('onehot', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('cat', categorical_transformer, np.where(is_categorical)[0]),
            ('num', numeric_transformer, np.where(~is_categorical)[0])])

    model = Pipeline(steps=[('preprocessor', preprocessor),
                            ('classifier', RandomForestClassifier())])

    model.fit(X_src, y_src)

    X_src_pproc = preprocessor.transform(X_src)
    if sparse.issparse(X_src_pproc):
        X_src_pproc = X_src_pproc.todense()

    list_of_shift = [GaussianNoise('medium', 0.8), FlipSign(0.8), SwappedValues(0.8)]

    list_of_domains_X = [X_src_pproc]

    print(" ")
    for shift in list_of_shift:
        print(shift.name)
        X_perturbed = copy.deepcopy(X_tar)
        if shift.feature_type == PerturbationConstants.NUMERIC:
            (X_perturbed[:, ~is_categorical], _) = shift.transform(X_tar[:, ~is_categorical].astype(float))
        elif shift.feature_type == PerturbationConstants.CATEGORICAL:
            (X_perturbed[:, is_categorical], _) = shift.transform(X_tar[:, is_categorical])
        else:
            (X_perturbed, _) = shift.transform(X_tar)

        X_perturbed_pproc = preprocessor.transform(X_perturbed)
        if sparse.issparse(X_perturbed_pproc):
            X_perturbed_pproc = X_perturbed_pproc.todense()

        list_of_domains_X.append(X_perturbed_pproc)

    orig_dims = X_src.shape[1]
    domain_clf = MultiDomainClassifier(orig_dims, dc=DomainClassifierModel.RF)

    model, score, train_test_data = domain_clf.build_model(list_of_domains_X, list_y=None)
    (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = train_test_data

    y_te_dcl_pred = model.predict(X_te_dcl)

    print("DC reference accuracy: %.2f" % domain_clf.ratio)
    print("DC observed  accuracy: %.2f" % score)
    print("Confusion Matrix")
    print(confusion_matrix(y_te_dcl, y_te_dcl_pred))

    assert (y_tr_old is None)
    assert (y_te_old is None)



