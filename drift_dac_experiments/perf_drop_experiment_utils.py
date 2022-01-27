from drift_dac_experiments.shifter import apply_shift
from drift_dac_experiments.viz_utils import name2type, name2severity
from drift_dac.drift_metrics import compute_drift_metrics
from drift_dac.domain_classifier import DomainClassifierModel
from drift_dac.covariate_shift import *
from drift_dac.prior_shift import *
from drift_dac.perturbation_shared_utils import NoShift
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import f1_score, accuracy_score
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from scipy.sparse import issparse
from scipy import stats

import pandas as pd
import numpy as np
import pickle
import os
import itertools


def get_X_y(df, target):
    X = np.array(df.loc[:, df.columns != target])
    y = np.array(df.loc[:, df.columns == target])
    return X, y


def retrieve_domain_df(dataset_df, split_variable, value, n_min_samples=500, n_max_samples=10000, seed=0):
    domain_dataset_df = dataset_df[dataset_df[split_variable] == value].drop(split_variable, axis=1)
    n_samples = domain_dataset_df.shape[0]
    if n_samples >= n_min_samples:
        if n_samples > n_max_samples:
            domain_dataset_df = domain_dataset_df.sample(n=n_max_samples, random_state=seed).reset_index(drop=True)
    else:
        domain_dataset_df = None  # not enough samples

    return domain_dataset_df


def split_into_domains(dataset_df, split_variable, n_min_samples=500, n_max_samples=10000, seed=0):
    split_values = dataset_df[split_variable].unique()
    domain_values = []
    for value in split_values:
        domain_dataset_df = retrieve_domain_df(dataset_df, split_variable, value, n_min_samples, n_max_samples, seed)
        if not (domain_dataset_df is None):
            domain_values.append((len(domain_values), split_variable, value, domain_dataset_df.shape[0]))

    return domain_values


def split_train_test_valid(dss_df, target, test_valid_size=0.6, test_size=0.5, seed=0):
    ## Shuffle and do a stratified split

    dss_df = dss_df[dss_df[target].notna()].reset_index(drop=True)

    if test_valid_size == 0:
        return dss_df, None, None

    dss_X = dss_df.drop(target, axis=1)
    dss_Y = dss_df[target]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_valid_size, random_state=seed)
    for train_index, valid_test_index in sss.split(dss_X, dss_Y):
        X_train, X_valid_test = dss_X.iloc[train_index], dss_X.iloc[valid_test_index]
        Y_train, Y_valid_test = dss_Y.iloc[train_index], dss_Y.iloc[valid_test_index]

    dss_valid_test = pd.concat([X_valid_test, Y_valid_test], axis=1)
    dss_valid_test.reset_index(inplace=True, drop=True)
    X_valid_test = dss_valid_test.drop(target, axis=1)
    Y_valid_test = dss_valid_test[target]

    sss = StratifiedShuffleSplit(n_splits=1, test_size=test_size, random_state=seed)
    for valid_index, test_index in sss.split(X_valid_test, Y_valid_test):
        X_valid, X_test = X_valid_test.iloc[valid_index], X_valid_test.iloc[test_index]
        Y_valid, Y_test = Y_valid_test.iloc[valid_index], Y_valid_test.iloc[test_index]

    ## Write train and test datasets
    train_df = pd.concat([X_train, Y_train], axis=1)
    test_df = pd.concat([X_test, Y_test], axis=1)
    valid_df = pd.concat([X_valid, Y_valid], axis=1)
    return train_df, valid_df, test_df


def get_rand_domain_split(dss_df, target, num_domain, rand_run,
                          out_path, save_dataset=True, test_valid_size=0.6, test_size=0.5):
    # print("Random split of domain data...")
    # Load data.

    df_filename = f'{out_path}/domain_{num_domain}_dfs_{rand_run}.pkl'

    if os.path.exists(df_filename):
        print("Loading from file: ", df_filename)
        with open(df_filename, 'rb') as f:
            [df_train, df_valid, df_test] = pickle.load(f)
    else:
        df_train, df_valid, df_test = split_train_test_valid(dss_df, target, test_valid_size, test_size, seed=rand_run)

        if save_dataset:
            with open(df_filename, 'wb') as f:
                pickle.dump([df_train, df_valid, df_test], f)

    features = [f for f in df_train.columns if f != target]
    categorical_features = df_train.select_dtypes(include=['object', 'category']).columns.tolist()
    is_categorical = np.array([True if feat in categorical_features else False for feat in features])

    return df_train, df_valid, df_test, is_categorical


def generate_shift(df_test, target, is_categorical, shift, rand_run, out_path, save_dataset=True, verbose=True):
    X_te_unproc, y_te_unproc = get_X_y(df_test, target)

    shifted_data_filename = f'{out_path}/shifted_data_{shift}_{rand_run}.pkl'

    if os.path.exists(shifted_data_filename):

        print("Loading from file: ", shifted_data_filename)
        with open(shifted_data_filename, 'rb') as f:
            [(X_te_unproc, y_te_unproc), shifted_indices, shifted_feat_indices] = pickle.load(f)

    else:

        (X_te_unproc, y_te_unproc), shifted_indices, shifted_feat_indices = apply_shift(
            X_te_unproc, y_te_unproc, shift, is_categorical)

        if save_dataset:
            with open(shifted_data_filename, 'wb') as f:
                pickle.dump([(X_te_unproc, y_te_unproc), shifted_indices, shifted_feat_indices], f)

    if verbose:
        print("%%%%%%%%%%%%%%%%% Summary - Shift - Before preprocessing")
        print(X_te_unproc.shape)
        print(np.unique(y_te_unproc, return_counts=True))
        if shifted_indices is not None:
            print('Len shifted: ', len(shifted_indices))
        if shifted_feat_indices is not None:
            print("N feat shifted: ", len(shifted_feat_indices))

    feature_names = [f for f in df_test.columns if f != target]

    df_test_shift = pd.DataFrame(data=np.concatenate([X_te_unproc, y_te_unproc], axis=1),
                                 columns=feature_names + [target],
                                 index=df_test.index)

    return df_test_shift, shifted_indices, shifted_feat_indices


def evaluate_model(model, X, y):
    predictions = model.predict_proba(X)
    y_predicted = np.argmax(predictions, axis=1)
    accuracy = accuracy_score(y, y_predicted)
    f1 = f1_score(y, y_predicted, average='macro')
    return accuracy, f1, predictions


def compute_statistics(predictions, n_percentiles=100):
    # they used only one class probability
    features = np.percentile(predictions[:, :-1], np.arange(0, n_percentiles + 1, 1), axis=0)

    return features


def fill_result_dataframe(dataset_name, model_type, domain_src, domain_tar, shift_name, shift_type,
                          accuracy_src, accuracy_tar, f1_src, f1_tar, features_tar,
                          metrics, rand_run, result_df=None):
    percentiles = range(features_tar.shape[0])
    classes = range(features_tar.shape[1])
    n_features = len(metrics['x_univ_t'])
    features_idx = range(n_features)

    if result_df is None:
        columns = ['dataset', 'model', 'id_domain_src', 'domain_src', 'size_src',
                   'id_domain_tar', 'domain_tar', 'size_tar', 'shift', 'shift_type', 'run',
                   'accuracy_src', 'accuracy_tar', 'accuracy_drop',
                   'f1_src', 'f1_tar', 'f1_drop',
                   'bbsd_soft_t', 'bbsd_soft_p_val', 'bbsd_hard_t', 'bbsd_hard_p_val',
                   'dc_acc', 'dc_p_val', 'confidence_drop', 'rca']

        columns.extend(['prediction_percentile_%s_%s' % (cl, perc)
                        for cl, perc in itertools.product(classes, percentiles)])

        columns.extend(['x_univ_t_%d' % feat_idx for feat_idx in features_idx])
        columns.extend(['x_univ_p_val_%d' % feat_idx for feat_idx in features_idx])

        result_df = pd.DataFrame(columns=columns)

    id_domain_src, split_variable, domain_value_src, size_src = domain_src
    id_domain_tar, split_variable, domain_value_tar, size_tar = domain_tar
    row_result = {'dataset': dataset_name,
                  'model': model_type,
                  'id_domain_src': id_domain_src,
                  'domain_src': split_variable + ":" + str(domain_value_src),
                  'size_src': size_src,
                  'id_domain_tar': id_domain_tar,
                  'domain_tar': split_variable + ":" + str(domain_value_tar),
                  'size_tar': size_tar,
                  'shift': shift_name,
                  'shift_type': shift_type,
                  'run': rand_run,
                  'accuracy_src': accuracy_src,
                  'accuracy_tar': accuracy_tar,
                  'accuracy_drop': accuracy_tar - accuracy_src,
                  'f1_src': f1_src,
                  'f1_tar': f1_tar,
                  'f1_drop': f1_tar - f1_src,
                  'bbsd_soft_t': metrics['bbsd_soft_t'],
                  'bbsd_soft_p_val': metrics['bbsd_soft_p_val'],
                  'bbsd_hard_t': metrics['bbsd_hard_t'],
                  'bbsd_hard_p_val': metrics['bbsd_hard_p_val'],
                  'dc_acc': metrics['dc_acc'],
                  'dc_p_val': metrics['dc_p_val'],
                  'confidence_drop': metrics['confidence_drop'],
                  'rca': metrics['rca']}

    for cl, perc in itertools.product(classes, percentiles):
        row_result['prediction_percentile_%s_%s' % (cl, perc)] = features_tar[perc, cl]

    for feat_idx in features_idx:
        row_result['x_univ_t_%d' % feat_idx] = metrics['x_univ_t'][feat_idx]
        row_result['x_univ_p_val_%d' % feat_idx] = metrics['x_univ_p_val'][feat_idx]

    return result_df.append(row_result, ignore_index=True)


class SimplePreprocessor(object):
    def __init__(self, is_categorical):
        numeric_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())])
        categorical_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
            ('ordinal', OrdinalEncoder(handle_unknown='use_encoded_value', unknown_value=-1))])

        self.preprocessor = ColumnTransformer(
            transformers=[
                ('cat', categorical_transformer, np.where(is_categorical)[0]),
                ('num', numeric_transformer, np.where(~is_categorical)[0])])

    def fit(self, x):
        return self.preprocessor.fit(x)

    def fit_transform(self, x):
        return self.preprocessor.fit_transform(x)

    def transform(self, x):
        return self.preprocessor.transform(x)


def add_drop_prediction_to_results(metrics_id, true_types, pred_drops, pred_types,
                                   result_df, set_type, pred_prefix):
    if pred_prefix + '_pred_drop' not in result_df.columns:
        result_df[pred_prefix + '_pred_drop'] = None
        result_df[pred_prefix + '_pred_type'] = None

    if 'set_type' not in result_df.columns:
        result_df['set_type'] = None
        result_df['true_type'] = None

    for i, df_id in enumerate(metrics_id):

        result_df.loc[df_id, pred_prefix + '_pred_drop'] = pred_drops[i]
        result_df.loc[df_id, 'set_type'] = set_type
        if pred_types is not None:
            result_df.loc[df_id, pred_prefix + '_pred_type'] = pred_types[i]
        if true_types is not None:
            result_df.loc[df_id, 'true_type'] = true_types[i]

    return result_df


class MetaDataset(object):
    def __init__(self):
        self.datasets = []
        self.datasets_orig = []
        self.primary_y = []
        self.drift_types = []
        self.drops = []
        self.metrics_id = []
        self.meta_features = []
        self.shift_names = []
        self.shift_types = []

    def append(self, dataset, dataset_orig, primary_y, drift_type, drop, metrics_id, meta_features, shift_name=None,
               shift_type=None):
        self.datasets.append(dataset)
        self.datasets_orig.append(dataset_orig)
        self.primary_y.append(primary_y)
        self.drift_types.append(drift_type)
        self.drops.append(drop)
        self.metrics_id.append(metrics_id)
        self.meta_features.append(meta_features)
        self.shift_names.append(shift_name)
        self.shift_types.append(shift_type)

    def arrayfy(self):
        self.datasets = np.array(self.datasets)
        self.datasets_orig = np.array(self.datasets_orig)
        self.primary_y = np.array(self.primary_y)
        self.drift_types = np.array(self.drift_types)
        self.drops = np.array(self.drops)
        self.metrics_id = np.array(self.metrics_id)
        self.meta_features = np.array(self.meta_features)
        self.shift_names = np.array(self.shift_names)
        self.shift_types = np.array(self.shift_types)

    def shuffle(self):
        shuffled_indices = np.random.permutation(len(self.drift_types))

        self.datasets = self.datasets[shuffled_indices]
        self.datasets_orig = self.datasets_orig[shuffled_indices]
        self.primary_y = self.primary_y[shuffled_indices]
        self.drops = self.drops[shuffled_indices]
        self.drift_types = self.drift_types[shuffled_indices]
        self.metrics_id = self.metrics_id[shuffled_indices]
        self.shift_names = self.shift_names[shuffled_indices]
        self.shift_types = self.shift_types[shuffled_indices]

    def filter_label_preserving(self, true_model):
        label_preserving_idx = [accuracy_score(true_model.predict(X), y) == 1. for X, y in
                                zip(self.datasets, self.primary_y)]

        self.datasets = self.datasets[label_preserving_idx]
        self.datasets_orig = self.datasets_orig[label_preserving_idx]
        self.primary_y = self.primary_y[label_preserving_idx]
        self.drift_types = self.drift_types[label_preserving_idx]
        self.drops = self.drops[label_preserving_idx]
        self.metrics_id = self.metrics_id[label_preserving_idx]
        self.meta_features = self.meta_features[label_preserving_idx]
        self.shift_names = self.shift_names[label_preserving_idx]
        self.shift_types = self.shift_types[label_preserving_idx]


class ReferenceTask(object):
    # drop is zero
    def __init__(self, model, X_tr_orig, y_tr, X_src_orig, y_src, preprocess, is_categorical):
        self.model = model
        self.X_tr_orig = X_tr_orig
        self.y_tr = y_tr
        self.X_src_orig = X_src_orig
        self.y_src = y_src
        self.preprocess = preprocess
        self.is_categorical = is_categorical
        self.ref_accuracy = self.model.score(self.preprocess.transform(X_src_orig), self.y_src)


def multi_domain_data_generation(id_domain, split_value, domain_value, model,
                                 list_of_shifts, list_of_train_shifts, rand_run,
                                 list_of_drift_types, dataset_params, exp_params,
                                 result_df):
    model_type = type(model).__name__

    target_domain_values = split_into_domains(dataset_params.dataset_df, dataset_params.split_variable,
                                              dataset_params.n_min_samples_target, dataset_params.n_max_samples,
                                              seed=rand_run)

    train = MetaDataset()
    test = MetaDataset()
    test_unseen = MetaDataset()
    test_natural = MetaDataset()

    domain_df = retrieve_domain_df(dataset_params.dataset_df, dataset_params.split_variable, split_value,
                                   dataset_params.n_min_samples, dataset_params.n_max_samples,
                                   seed=rand_run)

    rand_domain_data = get_rand_domain_split(domain_df, dataset_params.target, id_domain, rand_run,
                                             exp_params.out_path, exp_params.save_dataset, exp_params.test_valid_size,
                                             exp_params.test_size)

    df_train, df_target, df_test_large, is_categorical = rand_domain_data

    df_test = df_test_large.sample(n=df_target.shape[0], random_state=rand_run)

    print(df_train.shape)
    print(df_target.shape)
    print(df_test_large.shape)

    if exp_params.preprocess is None:
        preprocess = SimplePreprocessor(is_categorical)

    if exp_params.labelenc is None:
        labelenc = LabelEncoder()

    X_tr_orig, y_tr_orig = get_X_y(df_train, dataset_params.target)
    X_val_orig, y_val_orig = get_X_y(df_test, dataset_params.target)

    X_tr = preprocess.fit_transform(X_tr_orig)
    X_val = preprocess.transform(X_val_orig)

    if issparse(X_tr):
        X_tr = X_tr.todense()
    if issparse(X_val):
        X_val = X_val.todense()

    y_tr = labelenc.fit_transform(y_tr_orig)
    y_val = labelenc.transform(y_val_orig)

    if exp_params.param_grid is None:
        model.fit(X_tr, y_tr.ravel())
    else:
        gs_clf = GridSearchCV(model, exp_params.param_grid, cv=5)
        gs_clf.fit(X_tr, y_tr.ravel())
        model = gs_clf.best_estimator_

    X_src_orig, X_src, y_src = X_val_orig, X_val, y_val

    # calibrate base model
    model_calibrated = CalibratedClassifierCV(base_estimator=model, cv="prefit")
    model_calibrated.fit(X_src, y_src)
    model = model_calibrated

    accuracy_src, f1_src, predictions_src = evaluate_model(model, X_src, y_src)
    features_src = compute_statistics(predictions_src)

    ref_task = ReferenceTask(model, X_tr_orig, y_tr, X_src_orig, y_src, preprocess, is_categorical)

    n_x_features = X_val.shape[1]
    percentiles = range(features_src.shape[0])
    classes = range(features_src.shape[1])
    features = {
        'data': ['size_tar'],
        'drift': ['bbsd_soft_t',
                  # 'bbsd_soft_p_val',
                  # 'bbsd_hard_t',
                  # 'bbsd_hard_p_val',
                  'dc_acc',
                  # 'dc_p_val',
                  'confidence_drop',
                  'rca'] + ['x_univ_t_%d' % col for col in range(n_x_features)],
        'model': ['prediction_percentile_%s_%s' % (cl, perc) for cl, perc in itertools.product(classes, percentiles)]
    }
    all_features = features['drift'] + features['model']

    print('Generating synthetic shifts and metrics...')

    for sh_i, shift in enumerate(list_of_shifts + list_of_train_shifts):

        if sh_i >= len(list_of_shifts):  # training shifts
            # bootstrap from test pool
            df_test = df_test_large.sample(n=df_target.shape[0], random_state=rand_run + sh_i)
        else:
            # Fixed target dataset as in Amazon paper.
            df_test = df_target.copy(deep=True)

        if isinstance(shift, str):
            shift_name = shift
        else:
            shift_name = shift.name
            # hack for error-based sampling shift
            if isinstance(shift, ErrorBasedSampling):
                shift.model = Pipeline([('preprocessor', preprocess), ('clf', model)])
                shift.labelenc = labelenc

        try:

            df_test_shift, _, _ = generate_shift(df_test, dataset_params.target, is_categorical, shift,
                                                 rand_run, exp_params.out_path, exp_params.save_dataset,
                                                 verbose=exp_params.verbose)

        except Exception as e:
            print('Error --> Skip shift')
            print(shift_name)
            print(e)
            continue

        X_tar_orig, y_tar_orig = get_X_y(df_test_shift, dataset_params.target)
        X_tar = preprocess.transform(X_tar_orig)
        y_tar = labelenc.transform(y_tar_orig)

        if issparse(X_tar):
            X_tar = X_tar.todense()

        accuracy_tar, f1_tar, predictions_tar = evaluate_model(model, X_tar, y_tar)
        features_tar = compute_statistics(predictions_tar)
        metrics = compute_drift_metrics(X_src_orig, y_src, predictions_src, X_tar_orig, predictions_tar,
                                        model,
                                        dc_model=DomainClassifierModel.RF, return_dc_model=False,
                                        preprocessor=preprocess, is_categorical=is_categorical)
        shifted_domain_value = (id_domain, dataset_params.split_variable, split_value, X_tar.shape[0])

        result_df = fill_result_dataframe(dataset_params.dataset_name, model_type, domain_value, shifted_domain_value,
                                          shift_name,
                                          'synthetic', accuracy_src, accuracy_tar, f1_src, f1_tar, features_tar,
                                          metrics, rand_run, result_df=result_df)

        if sh_i >= len(list_of_shifts):  # training shifts

            drift_type_idx = list_of_drift_types.index(name2type(shift_name))
            soft_drift_type = np.zeros((len(list_of_drift_types),))
            soft_drift_type[drift_type_idx] = 1.

            drift_type = np.argmax(soft_drift_type)
            drop = accuracy_tar - accuracy_src
            metrics_id = result_df.shape[0] - 1
            meta_features = result_df.iloc[metrics_id][all_features]
            train.append(X_tar, X_tar_orig, y_tar, drift_type, drop, metrics_id, meta_features, shift_name,
                         name2type(shift_name))

        elif name2type(shift_name) in list_of_drift_types:

            drift_type_idx = list_of_drift_types.index(name2type(shift_name))
            soft_drift_type = np.zeros((len(list_of_drift_types),))
            if drift_type_idx == 0:
                soft_drift_type = np.zeros((len(list_of_drift_types),))
                soft_drift_type[drift_type_idx] = 1.
            else:
                sev = name2severity(shift_name)
                soft_drift_type[drift_type_idx] = sev
                soft_drift_type[0] = 1 - sev

            drift_type = np.argmax(soft_drift_type)
            drop = accuracy_tar - accuracy_src
            metrics_id = result_df.shape[0] - 1
            meta_features = result_df.iloc[metrics_id][all_features]
            test.append(X_tar, X_tar_orig, y_tar, drift_type, drop, metrics_id, meta_features, shift_name,
                        name2type(shift_name))

        else:
            drop = accuracy_tar - accuracy_src
            metrics_id = result_df.shape[0] - 1
            meta_features = result_df.iloc[metrics_id][all_features]
            test_unseen.append(X_tar, X_tar_orig, y_tar, None, drop, metrics_id, meta_features, shift_name,
                               name2type(shift_name))

    print('Generating natural shifts and metrics...')

    for other_domain_value in target_domain_values:
        id_other_domain, _, split_other_value, other_domain_size = other_domain_value
        if split_other_value == split_value:
            continue

        other_domain_df = retrieve_domain_df(dataset_params.dataset_df, dataset_params.split_variable,
                                             split_other_value,
                                             n_min_samples=dataset_params.n_min_samples_target,
                                             n_max_samples=dataset_params.n_min_samples_target,
                                             seed=rand_run)

        rand_domain_data = get_rand_domain_split(other_domain_df, dataset_params.target, id_other_domain, rand_run,
                                                 exp_params.out_path, exp_params.save_dataset, test_valid_size=0)

        df_train, _, _, is_categorical = rand_domain_data

        X_tar_orig, y_tar_orig = get_X_y(df_train, dataset_params.target)
        X_tar = preprocess.transform(X_tar_orig)
        y_tar = labelenc.transform(y_tar_orig)

        if issparse(X_tar):
            X_tar = X_tar.todense()

        accuracy_tar, f1_tar, predictions_tar = evaluate_model(model, X_tar, y_tar)
        features_tar = compute_statistics(predictions_tar)
        metrics = compute_drift_metrics(X_src_orig, y_src, predictions_src, X_tar_orig, predictions_tar,
                                        model,
                                        dc_model=DomainClassifierModel.RF, return_dc_model=False,
                                        preprocessor=preprocess, is_categorical=is_categorical)

        shift_name = '->'.join([str(split_value), str(split_other_value)])
        result_df = fill_result_dataframe(dataset_params.dataset_name, model_type, domain_value, other_domain_value,
                                          shift_name,
                                          'natural', accuracy_src, accuracy_tar, f1_src, f1_tar, features_tar,
                                          metrics, rand_run, result_df=result_df)

        drop = accuracy_tar - accuracy_src
        metrics_id = result_df.shape[0] - 1
        meta_features = result_df.iloc[metrics_id][all_features]
        test_natural.append(X_tar, X_tar_orig, y_tar, None, drop, metrics_id, meta_features)

    # to array
    train.arrayfy()
    test.arrayfy()
    test_unseen.arrayfy()
    test_natural.arrayfy()

    # shuffle
    train.shuffle()

    return train, test, test_unseen, test_natural, ref_task, result_df


def shift_generation_setting(dataset_params, exp_params):
    pos_class = dataset_params.pos_class
    n_training_datasets_per_shift = exp_params.n_training_datasets_per_shift
    n_test_datasets_per_shift = exp_params.n_test_datasets_per_shift
    n_unseen_datasets_per_shift = exp_params.n_unseen_datasets_per_shift
    train_on_subpop = exp_params.train_on_subpop

    # Define shift types.
    severities = np.random.uniform(low=0.25, high=0.99, size=n_unseen_datasets_per_shift)
    severities_feat = np.random.uniform(low=0.25, high=0.99, size=n_unseen_datasets_per_shift)

    list_of_unseen_shifts = []
    list_of_unseen_shifts += [ConstantNumeric(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
    list_of_unseen_shifts += [PlusMinusSomePercent(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
    list_of_unseen_shifts += [GaussianNoise('medium', sev_i, sev_j) for sev_i, sev_j in
                              zip(severities, severities_feat)]
    list_of_unseen_shifts += [GaussianNoise('small', sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
    list_of_unseen_shifts += [FlipSign(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]

    # training shifts
    severities_train = np.random.uniform(low=0.75, high=0.99, size=n_training_datasets_per_shift)
    severities_feat_train = np.random.uniform(low=0.25, high=0.99, size=n_training_datasets_per_shift)

    list_of_train_shifts = n_training_datasets_per_shift * [NoShift()]  # negatives

    # test shifts
    severities = np.random.uniform(low=0.25, high=0.74, size=n_test_datasets_per_shift)
    severities_feat = np.random.uniform(low=0.25, high=0.99, size=n_test_datasets_per_shift)

    list_of_test_shifts = n_test_datasets_per_shift * [NoShift()]  # negatives

    if train_on_subpop:
        list_of_train_shifts += [SubsampleNumeric(sev_i, sev_j) for sev_i, sev_j in
                                 zip(severities_train, severities_feat_train)]
        list_of_train_shifts += [SubsampleCategorical(sev_i, sev_j) for sev_i, sev_j in
                                 zip(severities_train, severities_feat_train)]
        list_of_train_shifts += [KnockOut(pos_class, sev) for sev in severities_train]
        list_of_train_shifts += [SubsampleJoint(sev) for sev in severities_train]
        # list_of_train_shifts += [ErrorBasedSampling(sev) for sev in severities_train]

        list_of_test_shifts += [SubsampleNumeric(sev_i, sev_j) for sev_i, sev_j in
                                zip(severities, severities_feat)]
        list_of_test_shifts += [SubsampleCategorical(sev_i, sev_j) for sev_i, sev_j in
                                zip(severities, severities_feat)]
        list_of_test_shifts += [KnockOut(pos_class, sev) for sev in severities]
        list_of_test_shifts += [SubsampleJoint(sev) for sev in severities]
        # list_of_test_shifts += [ErrorBasedSampling(sev) for sev in severities]

        list_of_unseen_shifts += [Scaling(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
        list_of_unseen_shifts += [SwappedValues(sev) for sev in severities]
        list_of_unseen_shifts += [MissingValues(sev_i, sev_j) for sev_i, sev_j in
                                  zip(severities, severities_feat)]
        list_of_unseen_shifts += [Outliers(sev_i, sev_j) for sev_i, sev_j in
                                  zip(severities, severities_feat)]
    else:

        list_of_train_shifts += [Scaling(sev_i, sev_j) for sev_i, sev_j in zip(severities_train, severities_feat_train)]
        list_of_train_shifts += [SwappedValues(sev) for sev in severities_train]
        list_of_train_shifts += [MissingValues(sev_i, sev_j) for sev_i, sev_j in
                                 zip(severities_train, severities_feat_train)]
        list_of_train_shifts += [Outliers(sev_i, sev_j) for sev_i, sev_j in
                                 zip(severities_train, severities_feat_train)]

        list_of_test_shifts += [Scaling(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
        list_of_test_shifts += [SwappedValues(sev) for sev in severities]
        list_of_test_shifts += [MissingValues(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
        list_of_test_shifts += [Outliers(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]

        list_of_unseen_shifts += [SubsampleNumeric(sev_i, sev_j) for sev_i, sev_j in zip(severities, severities_feat)]
        list_of_unseen_shifts += [SubsampleCategorical(sev_i, sev_j) for sev_i, sev_j in
                                  zip(severities, severities_feat)]
        list_of_unseen_shifts += [KnockOut(pos_class, sev) for sev in severities]
        list_of_unseen_shifts += [SubsampleJoint(sev) for sev in severities]
        # list_of_unseen_shifts += [ErrorBasedSampling(sev) for sev in severities]

    return list_of_train_shifts, list_of_test_shifts, list_of_unseen_shifts


def within_ci_accuracy(y_true, y_pred, ref_accuracy, n_samples, alpha):
    acc_pred = ref_accuracy + y_pred
    acc_true = ref_accuracy + y_true
    sigma = np.sqrt(acc_true * (1 - acc_true) / n_samples)
    ci = stats.norm.ppf(1 - alpha / 2) * sigma

    acc_inf = acc_true - ci
    acc_sup = acc_true + ci

    pred = (acc_pred <= acc_sup) * (acc_inf <= acc_pred)
    return np.mean(pred)


def within_ci_mae(y_true, y_pred, ref_accuracy, n_samples, alpha):
    acc_pred = ref_accuracy + y_pred
    acc_true = ref_accuracy + y_true
    sigma = np.sqrt(acc_true * (1 - acc_true) / n_samples)
    ci = stats.norm.ppf(1 - alpha / 2) * sigma
    corrected_error = np.abs(acc_pred - acc_true) - ci
    corrected_error[corrected_error < 0] = 0
    return np.mean(corrected_error)


def within_ci_binning_mae(y_true, y_pred, ref_accuracy, cifb):
    acc_pred = ref_accuracy + y_pred
    acc_true = ref_accuracy + y_true
    ci = cifb.get_ci(acc_true)
    corrected_error = np.abs(acc_pred - acc_true) - ci
    corrected_error[corrected_error < 0] = 0
    return np.mean(corrected_error)


def likelihood(y_true, y_pred, ref_accuracy, n_samples):
    acc_pred = ref_accuracy + y_pred
    acc_true = ref_accuracy + y_true
    sigma = np.sqrt(acc_true * (1 - acc_true) / n_samples)
    # if use the sum here (instead of mean) lik won't be comparable across test sets having different n. of datasets
    lik = np.mean((acc_true - acc_pred) ** 2 / sigma ** 2)
    return lik


def auc_sigma_accuracy(y_pred, y_true, ref_accuracy, n_samples):
    z_range = 3 * np.arange(0, 1., 0.01)
    acc_pred = ref_accuracy + y_pred
    acc_true = ref_accuracy + y_true
    sigma = np.sqrt(acc_true * (1 - acc_true) / n_samples)

    auc = []
    for z in z_range:
        acc_inf = acc_true - z * sigma
        acc_sup = acc_true + z * sigma
        pred = (acc_pred < acc_sup) * (acc_inf < acc_pred)

        auc.append(np.mean(pred))

    return np.mean(auc)


def auc_sigma_mae(y_pred, y_true, ref_accuracy, n_samples):
    z_range = 3 * np.arange(0, 1., 0.01)
    acc_pred = ref_accuracy + y_pred
    acc_true = ref_accuracy + y_true
    sigma = np.sqrt(acc_true * (1 - acc_true) / n_samples)

    auc = []
    for z in z_range:
        ci = z * sigma
        corrected_error = np.abs(acc_pred - acc_true) - ci
        corrected_error[corrected_error < 0] = 0

        auc.append(np.mean(corrected_error))

    return np.mean(auc)
