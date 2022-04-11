import os
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.dummy import DummyRegressor
from sklearn.metrics import r2_score
from sklearn.base import clone
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()
tf.experimental.output_all_intermediates(True)

import pickle
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import GridSearchCV
from drift_dac_experiments.multi_domain_performance_predictor import MultiDomainPerformancePredictor, \
    MultiDomainPerformancePredictorDNN
from drift_dac_experiments.perf_drop_experiment_utils import within_ci_accuracy, likelihood, within_ci_mae, \
    auc_sigma_accuracy, auc_sigma_mae

import matplotlib.colors as mcolors


class SetOfDatasets(object):
    def __init__(self, X, y, primary_y, drift_types, meta_features_orig):
        self.X = X
        self.y = y
        self.primary_y = primary_y
        self.drift_types = drift_types
        self.meta_features_orig = meta_features_orig
        self.meta_features = None
        self.meta_features_amazon = None
        self.meta_features_naver = None

        self.X_with_pred = None
        self.y_hat_mdl = None
        self.y_hat_meta_proba = None
        self.X_mdl = None

    def select(self, keep_idx):
        self.X = self.X[keep_idx]
        self.y = self.y[keep_idx]
        self.meta_features_orig = self.meta_features_orig[keep_idx]
        self.drift_types = self.drift_types[keep_idx]
        self.primary_y = self.primary_y[keep_idx]

    def fit_preproc(self, expert_feat_start, expert_feat_idx):
        imp = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                        ('scaler', StandardScaler())])

        imp.fit(self.meta_features_orig)

        imp_amazon = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                               ('scaler', StandardScaler())])

        imp_amazon.fit(self.meta_features_orig[:, expert_feat_start:])

        imp_naver = Pipeline([('imputer', SimpleImputer(missing_values=np.nan, strategy='mean')),
                              ('scaler', StandardScaler())])

        imp_naver.fit(self.meta_features_orig[:, expert_feat_idx])

        return imp, imp_amazon, imp_naver

    def process_data(self, imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx, base_model):
        self.meta_features = imp.transform(self.meta_features_orig)
        self.meta_features_amazon = imp_amazon.transform(self.meta_features_orig[:, expert_feat_start:])
        self.meta_features_naver = imp_naver.transform(self.meta_features_orig[:, expert_feat_idx])

        self.y_hat_meta_proba = np.array([base_model.predict_proba(x) for x in self.X])

        self.X_with_pred = np.concatenate([self.X, self.y_hat_meta_proba], axis=2)

        # determine correct-incorrect outcome - these are targets for the meta model trainer
        self.y_hat_mdl = np.asarray((self.primary_y == np.argmax(self.y_hat_meta_proba, axis=-1)), dtype=np.int)
        # get input features for meta training
        faux1 = np.sort(self.y_hat_meta_proba, axis=-1)
        # add delta between top and second candidate
        faux2 = np.expand_dims(faux1[:, :, -1] - faux1[:, :, -2], axis=-1)
        self.X_mdl = np.concatenate([self.X, faux1, faux2], axis=-1)


def pp_comparison(n_runs, pp_list, fld, out_fld, keep_ratio_train_shifts=1., update=True):
    all_runs_mae = dict()
    all_runs_r2 = dict()
    all_runs_within_ci_accuracy = dict()
    all_runs_likelihood = dict()
    all_runs_within_ci_mae = dict()
    all_runs_auc_sigma_acc = dict()
    all_runs_auc_sigma_mae = dict()

    for seed in range(n_runs):
        print('%d/%d' % (seed + 1, n_runs))

        np.random.seed(seed)

        with open(os.path.join(fld, 'data_%d.pkl' % seed), 'rb') as f:
            train, test, test_unseen, test_natural, ref_task, _ = pickle.load(f)

        with open(os.path.join(fld, 'shifts_%d.pkl' % seed), 'rb') as f:
            list_of_drift_types, filtered_list_of_train_shifts, filtered_list_of_test_shifts, filtered_list_of_unseen_shifts = pickle.load(
                f)

        data = dict()

        # filter away subsample shift
        perturbation_shifts_ids = [i for i, sh in enumerate(filtered_list_of_unseen_shifts) if
                                   not any([sh_type in sh for sh_type in ('subsample', 'ko_shift')])]
        subpop_shifts_ids = [i for i, sh in enumerate(filtered_list_of_unseen_shifts) if
                             any([sh_type in sh for sh_type in ('subsample', 'ko_shift')])]
        no_shifts_ids = [i for i, sh in enumerate(filtered_list_of_test_shifts) if 'no_shift' in sh]
        shifts_ids = [i for i, sh in enumerate(filtered_list_of_test_shifts) if not ('no_shift' in sh)]

        data['train'] = SetOfDatasets(train.datasets, train.drops, train.primary_y, train.drift_types,
                                      train.meta_features)
        if keep_ratio_train_shifts < 1.:
            # OOM issues: reduce n. of training shifts to use
            keep_idx = np.random.choice(train.datasets.shape[0],
                                        size=int(np.ceil(keep_ratio_train_shifts * train.datasets.shape[0])),
                                        replace=False)
            data['train'].select(keep_idx)
        expert_feat_start = 4 + train.datasets[0].shape[1]  # use prediction percentiles only
        expert_feat_idx = [1, 2, 3]  # use dc_acc, confidence_drop and rca only
        imp, imp_amazon, imp_naver = data['train'].fit_preproc(expert_feat_start, expert_feat_idx)

        data['test_no_shift'] = SetOfDatasets(test.datasets, test.drops, test.primary_y, test.drift_types,
                                              test.meta_features)
        data['test_no_shift'].select(no_shifts_ids)

        data['test'] = SetOfDatasets(test.datasets, test.drops, test.primary_y, test.drift_types,
                                     test.meta_features)
        data['test'].select(shifts_ids)

        data['test_unseen'] = SetOfDatasets(test_unseen.datasets, test_unseen.drops, test_unseen.primary_y,
                                            test_unseen.drift_types,
                                            test_unseen.meta_features)
        data['test_unseen'].select(perturbation_shifts_ids)

        data['test_unseen_subpop'] = SetOfDatasets(test_unseen.datasets, test_unseen.drops, test_unseen.primary_y,
                                                   test_unseen.drift_types,
                                                   test_unseen.meta_features)
        data['test_unseen_subpop'].select(subpop_shifts_ids)

        data['test_natural'] = SetOfDatasets(test_natural.datasets, test_natural.drops, test_natural.primary_y,
                                             test_natural.drift_types,
                                             test_natural.meta_features)

        X_src = ref_task.preprocess.transform(ref_task.X_src_orig)
        y_src = ref_task.y_src
        ref_accuracy = ref_task.ref_accuracy

        base_model = ref_task.model

        data['train'].process_data(imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx, base_model)
        data['test_no_shift'].process_data(imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx, base_model)
        data['test'].process_data(imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx, base_model)
        data['test_unseen'].process_data(imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx, base_model)
        data['test_unseen_subpop'].process_data(imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx,
                                                base_model)
        data['test_natural'].process_data(imp, imp_amazon, imp_naver, expert_feat_start, expert_feat_idx, base_model)

        y_hat_meta_proba = base_model.predict_proba(X_src)
        # determine correct-incorrect outcome - these are targets for the meta model trainer
        y_hat_meta_targets = np.asarray((y_src == np.argmax(y_hat_meta_proba, axis=-1)), dtype=np.int)
        # get input features for meta training
        faux1 = np.sort(y_hat_meta_proba, axis=-1)
        # add delta between top and second candidate
        faux2 = np.expand_dims(faux1[:, -1] - faux1[:, -2], axis=-1)
        X_meta_in = np.concatenate([X_src, faux1, faux2], axis=-1)

        all_models_file = os.path.join(out_fld, 'all_models_%d.pkl' % seed)
        if os.path.exists(all_models_file) and update:
            with open(all_models_file, 'rb') as f:
                all_models = pickle.load(f)
        else:
            all_models = dict()

        for pp_name in pp_list:
            print(pp_name)
            y_pred = dict()
            if pp_name == 'Dummy':
                pp = DummyRegressor().fit(data['train'].meta_features, data['train'].y)
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = pp.predict(data[set_name].meta_features)
            elif pp_name == 'ExpertRF':
                param_grid = {
                    'n_estimators': [5, 10, 20, 50, 100],
                    'criterion': ['mae']
                }

                pp = GridSearchCV(RandomForestRegressor(criterion='mae'),
                                  param_grid,
                                  scoring='neg_mean_absolute_error').fit(data['train'].meta_features,
                                                                         data['train'].y).best_estimator_
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = pp.predict(data[set_name].meta_features)
            elif pp_name == 'ExpertRF (amazon)':

                param_grid = {
                    'n_estimators': [5, 10, 20, 50, 100],
                    'criterion': ['mae']
                }

                pp = GridSearchCV(RandomForestRegressor(criterion='mae'),
                                  param_grid, scoring='neg_mean_absolute_error').fit(data['train'].meta_features_amazon,
                                                                                     data['train'].y).best_estimator_
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = pp.predict(data[set_name].meta_features_amazon)
            elif pp_name == 'ExpertRF (naver)':
                param_grid = {
                    'n_estimators': [5, 10, 20, 50, 100],
                    'criterion': ['mae']
                }

                pp = GridSearchCV(RandomForestRegressor(criterion='mae'),
                                  param_grid, scoring='neg_mean_absolute_error').fit(data['train'].meta_features_naver,
                                                                                     data['train'].y).best_estimator_
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = pp.predict(data[set_name].meta_features_naver)
            elif pp_name == 'MultiDomainClassifier':
                pp = MultiDomainPerformancePredictor(n_features=data['train'].X_with_pred.shape[2],
                                                     n_domains=len(np.unique(data['train'].drift_types)),
                                                     multi_task=True)
                pp.fit(data['train'].X_with_pred, data['train'].y, data['train'].drift_types)
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = pp.predict(data[set_name].X_with_pred)[0]
            elif pp_name == 'ErrorPredictorKNN':
                pp = KNeighborsClassifier(3)
                pp.fit(data['train'].X_mdl.reshape(-1, data['train'].X_mdl.shape[-1]),
                       data['train'].y_hat_mdl.reshape(-1, 1))
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.array(
                        [pp.predict_proba(x)[:, 1].mean() for x in data[set_name].X_mdl]) - ref_accuracy
            elif pp_name == 'ErrorPredictorRF':
                pp = RandomForestClassifier()
                pp.fit(data['train'].X_mdl.reshape(-1, data['train'].X_mdl.shape[-1]),
                       data['train'].y_hat_mdl.reshape(-1, 1))
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.array(
                        [pp.predict_proba(x)[:, 1].mean() for x in data[set_name].X_mdl]) - ref_accuracy
            elif pp_name == 'ErrorPredictorRF_no_shift':
                pp = RandomForestClassifier()
                pp.fit(X_meta_in, y_hat_meta_targets)
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.array(
                        [pp.predict_proba(x)[:, 1].mean() for x in data[set_name].X_mdl]) - ref_accuracy
            elif pp_name == 'ErrorPredictorGradBoost':
                meta_config = {'n_estimators': 200,
                               'max_depth': 3,
                               'learning_rate': 0.001,
                               'min_samples_leaf': 10,
                               'min_samples_split': 10,
                               'random_state': 42}

                pp = GradientBoostingClassifier(**meta_config)
                pp.fit(data['train'].X_mdl.reshape(-1, data['train'].X_mdl.shape[-1]),
                       data['train'].y_hat_mdl.reshape(-1, 1))
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.array(
                        [pp.predict_proba(x)[:, 1].mean() for x in data[set_name].X_mdl]) - ref_accuracy
            elif pp_name == 'DNN':
                pp = MultiDomainPerformancePredictorDNN(n_samples=data['train'].X_with_pred.shape[1],
                                                        n_features=data['train'].X_with_pred.shape[2],
                                                        encoded_ds_size=100,
                                                        n_layers=5,
                                                        encoder_type='mlp',
                                                        n_domains=len(np.unique(data['train'].drift_types)))

                pp.fit(data['train'].X_with_pred, data['train'].y, None, epochs=100, batch_size=400,
                       validation_split=0.2, verbose=False, early_stop_patience=50, lr=0.001, list_X_meta=None)

                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.atleast_1d(pp.predict(data[set_name].X_with_pred)[0].squeeze())
            elif pp_name == 'DNN+Expert':
                pp = MultiDomainPerformancePredictorDNN(n_samples=data['train'].X_with_pred.shape[1],
                                                        n_features=data['train'].X_with_pred.shape[2],
                                                        encoded_ds_size=100,
                                                        n_layers=5,
                                                        encoder_type='mlp',
                                                        n_meta_features=data['train'].meta_features.shape[1],
                                                        n_domains=len(np.unique(data['train'].drift_types)))

                pp.fit(data['train'].X_with_pred, data['train'].y, None, list_X_meta=data['train'].meta_features,
                       verbose=False, lr=.01, early_stop_patience=50, epochs=100,
                       batch_size=200, validation_split=0.2)

                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.atleast_1d(
                        pp.predict(data[set_name].X_with_pred, list_X_meta=data[set_name].meta_features)[0].squeeze())

            elif pp_name == 'MetaErrorPred':
                x_train, x_meta, y_train, y_meta = train_test_split(
                    np.moveaxis(data['train'].X, 1, 0), np.moveaxis(data['train'].primary_y, 1, 0), train_size=0.5
                )

                x_train = np.moveaxis(x_train, 0, 1).reshape(-1, x_train.shape[2])
                x_meta = np.moveaxis(x_meta, 0, 1)

                y_train = np.moveaxis(y_train, 0, 1).reshape(-1, 1).ravel()
                #y_meta = np.moveaxis(y_meta, 0, 1)

                drops_meta = data['train'].y.reshape(-1, 1)
                meta_features = data['train'].meta_features

                y_pred = ref_task.model.predict(x_train)
                correct_train = (y_train == y_pred)

                errpred = RandomForestClassifier()

                errpred.fit(x_train, correct_train)

                pp = LinearRegression()  # RandomForestRegressor() #Ridge(alpha=1e-3) #LinearRegression() #Lasso(alpha=0.01)

                acc_pred = np.concatenate([errpred.predict_proba(x)[:, 1].mean().reshape(-1, 1) for x in x_meta], axis=0)
                pp.fit(meta_features, ((ref_accuracy + drops_meta) - acc_pred).ravel())

                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.array([errpred.predict_proba(x)[:, 1].mean() +
                                                 pp.predict(meta.reshape(1, -1)).reshape(-1)[0] - ref_accuracy for
                                                 x, meta in zip(data[set_name].X, data[set_name].meta_features)])
                    # hack to limit bound
                    y_pred[set_name][y_pred[set_name] < -ref_accuracy] = -ref_accuracy
                    y_pred[set_name][y_pred[set_name] > 1-ref_accuracy] = 1-ref_accuracy

            elif pp_name == 'ATC':
                negative_entropy = np.sum(y_hat_meta_proba * np.log(y_hat_meta_proba), axis=1)
                negative_entropy = np.nan_to_num(negative_entropy)
                t_levels = np.sort(np.unique(negative_entropy))
                error_rate_src = 1 - np.mean(y_hat_meta_targets)
                min_diff = 1.0
                best_thresh = t_levels[0]
                for t in t_levels:
                    diff = np.abs(np.mean(negative_entropy < t) - error_rate_src)
                    if diff < min_diff:
                        min_diff = diff
                        best_thresh = t
                    elif diff > min_diff:
                        break

                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = np.array(
                        [np.mean(np.sum(y_hat_meta_proba * np.log(y_hat_meta_proba), axis=1) >= best_thresh) - ref_accuracy for
                         y_hat_meta_proba in data[set_name].y_hat_meta_proba])

            elif pp_name == 'LODOErrorPredictor (oracle)':
                model = RandomForestClassifier()

                n_max_domains = data['test_natural'].meta_features.shape[0]
                if n_max_domains > 1:
                    y_pred = dict()
                    for set_name in data:
                        y_pred[set_name] = []

                    for i in range(n_max_domains):
                        x = data['test_natural'].X_mdl[i]

                        X_other_mdl = np.concatenate(
                            [data['test_natural'].X_mdl[:i], data['test_natural'].X_mdl[i + 1:]])
                        y_other_mdl = np.concatenate(
                            [data['test_natural'].y_hat_mdl[:i], data['test_natural'].y_hat_mdl[i + 1:]])

                        pp = clone(model)  # reset model
                        pp.fit(X_other_mdl.reshape(-1, X_other_mdl.shape[-1]), y_other_mdl.reshape(-1, 1))

                        for set_name in data:
                            if set_name == 'test_natural':
                                y_pred[set_name].append(pp.predict_proba(x)[:, 1].mean() - ref_accuracy)
                            else:
                                y_pred[set_name].append(
                                    [pp.predict_proba(x)[:, 1].mean() - ref_accuracy for x in data[set_name].X_mdl])

                    for set_name in data:
                        if set_name == 'test_natural':
                            y_pred[set_name] = np.array(y_pred[set_name])
                        else:
                            y_pred[set_name] = np.mean(np.array(y_pred[set_name]), axis=0)

                else:
                    # pass cannot be computed
                    y_pred = dict()
                    for set_name in data:
                        y_pred[set_name] = data[set_name].y
            elif pp_name == 'TargetDummy (oracle)':
                y_pred = dict()
                for set_name in data:
                    y_pred[set_name] = DummyRegressor().fit(data[set_name].meta_features, data[set_name].y).predict(
                        data[set_name].meta_features)

            elif pp_name == 'TargetErrorPredictor (oracle)':
                y_pred = dict()
                for set_name in data:
                    pp = RandomForestClassifier().fit(data[set_name].X_mdl.reshape(-1, data[set_name].X_mdl.shape[-1]),
                                                      data[set_name].y_hat_mdl.reshape(-1, 1))
                    y_pred[set_name] = np.array(
                        [pp.predict_proba(x)[:, 1].mean() for x in data[set_name].X_mdl]) - ref_accuracy

            y_pred['test_natural'] = np.atleast_1d(np.array(y_pred['test_natural']))

            all_models[pp_name] = y_pred

        with open(all_models_file, 'wb') as f:
            pickle.dump(all_models, f)

        labels = list(all_models.keys())

        n_pp = len(labels)

        if seed == 0:
            pp_colors = np.random.choice(list(mcolors.CSS4_COLORS), size=n_pp, replace=False)

        mae = dict()
        r2 = dict()
        within_ci_acc = dict()
        lik = dict()
        wci_mae = dict()
        auc_sigma_acc = dict()
        auc_sigma_maerror = dict()

        if seed == 0:
            for set_name in data:
                all_runs_mae[set_name] = []
                all_runs_r2[set_name] = []
                all_runs_within_ci_accuracy[set_name] = []
                all_runs_likelihood[set_name] = []
                all_runs_within_ci_mae[set_name] = []
                all_runs_auc_sigma_acc[set_name] = []
                all_runs_auc_sigma_mae[set_name] = []

        alpha = 0.05
        n_samples = data['train'].X[0].shape[0]
        for set_name in data:
            mae[set_name] = [np.abs(data[set_name].y - all_models[label][set_name]) for label in labels]
            r2[set_name] = [r2_score(data[set_name].y, all_models[label][set_name]) for label in labels]
            within_ci_acc[set_name] = [
                within_ci_accuracy(data[set_name].y, all_models[label][set_name], ref_accuracy, n_samples, alpha) for
                label in labels]
            lik[set_name] = [likelihood(data[set_name].y, all_models[label][set_name], ref_accuracy, n_samples) for
                             label in labels]
            wci_mae[set_name] = [
                within_ci_mae(data[set_name].y, all_models[label][set_name], ref_accuracy, n_samples, alpha) for label
                in labels]
            auc_sigma_acc[set_name] = [
                auc_sigma_accuracy(data[set_name].y, all_models[label][set_name], ref_accuracy, n_samples) for label in
                labels]
            auc_sigma_maerror[set_name] = [
                auc_sigma_mae(data[set_name].y, all_models[label][set_name], ref_accuracy, n_samples) for label in
                labels]

            all_runs_mae[set_name].append(np.array(mae[set_name]))
            all_runs_r2[set_name].append(np.array(r2[set_name]))
            all_runs_within_ci_accuracy[set_name].append(np.array(within_ci_acc[set_name]))
            all_runs_likelihood[set_name].append(np.array(lik[set_name]))
            all_runs_within_ci_mae[set_name].append(np.array(wci_mae[set_name]))
            all_runs_auc_sigma_acc[set_name].append(np.array(auc_sigma_acc[set_name]))
            all_runs_auc_sigma_mae[set_name].append(np.array(auc_sigma_maerror[set_name]))

        plt.figure(figsize=(8, 5));
        for i, lab in enumerate(labels):
            r2_all = []
            for set_name in r2:
                r2_all.append(r2[set_name][i])
            plt.plot(r2_all, label=labels[i], color=pp_colors[i], marker='o', linewidth=2, markersize=8);
        plt.legend(bbox_to_anchor=(1.4, 1.05));
        plt.xticks(range(len(r2.keys())), r2.keys());
        plt.ylim(-0.1, 1.1);
        plt.hlines(y=0.0, xmin=0, xmax=len(r2.keys()), linestyles='dashed');
        plt.title("R2 Score of Performance Predictors");
        plt.ylabel('r2 score');

        plt.tight_layout()
        plt.savefig(os.path.join(out_fld, 'r2_score_%d.png' % seed), bbox_inches='tight')
        plt.close()

        for set_name in mae:
            plt.figure(figsize=(8, 5));
            plt.boxplot(mae[set_name], labels=labels)
            plt.hlines(y=0.02, xmin=0, xmax=n_pp + 1, linestyles='dashed')
            plt.title("Performance Predictors Absolute Error on %s" % set_name);
            plt.xticks(rotation=45);
            plt.ylim(bottom=-0.01, top=0.1);
            plt.ylabel('absolute error');

            plt.tight_layout()
            plt.savefig(os.path.join(out_fld, './abs_error_%s_%d.png' % (set_name, seed)), bbox_inches='tight')
            plt.close()

    for set_name in all_runs_mae:
        all_runs_mae[set_name] = np.array(all_runs_mae[set_name])
        all_runs_r2[set_name] = np.array(all_runs_r2[set_name])
        all_runs_within_ci_accuracy[set_name] = np.array(all_runs_within_ci_accuracy[set_name])
        all_runs_likelihood[set_name] = np.array(all_runs_likelihood[set_name])
        all_runs_within_ci_mae[set_name] = np.array(all_runs_within_ci_mae[set_name])
        all_runs_auc_sigma_acc[set_name] = np.array(all_runs_auc_sigma_acc[set_name])
        all_runs_auc_sigma_mae[set_name] = np.array(all_runs_auc_sigma_mae[set_name])

        np.save(os.path.join(out_fld, 'r2_score_%s.npy' % set_name), all_runs_r2[set_name])
        np.save(os.path.join(out_fld, 'within_ci_accuracy_%s.npy' % set_name), all_runs_within_ci_accuracy[set_name])
        np.save(os.path.join(out_fld, 'likelihood_%s.npy' % set_name), all_runs_likelihood[set_name])
        np.save(os.path.join(out_fld, 'abs_error_%s.npy' % set_name), all_runs_mae[set_name])
        np.save(os.path.join(out_fld, 'within_ci_mae_%s.npy' % set_name), all_runs_within_ci_mae[set_name])
        np.save(os.path.join(out_fld, 'auc_sigma_accuracy_%s.npy' % set_name), all_runs_auc_sigma_acc[set_name])
        np.save(os.path.join(out_fld, 'auc_sigma_mae_%s.npy' % set_name), all_runs_auc_sigma_mae[set_name])

    model_names_files = os.path.join(out_fld, 'model_names.npy')
    if os.path.exists(model_names_files) and update:
        model_names = list(np.load(model_names_files))
        for n in pp_list:
            if n not in model_names:
                model_names += [n]
        np.save(model_names_files, np.array(model_names))
    else:
        np.save(model_names_files, np.array(pp_list))

    all_runs_mae_mean = dict()

    for set_name in all_runs_mae:
        all_runs_mae_mean[set_name] = np.mean(all_runs_mae[set_name], axis=2)

        plt.figure(figsize=(8, 5));
        plt.boxplot(all_runs_mae_mean[set_name], labels=labels)
        plt.hlines(y=0.02, xmin=0, xmax=n_pp + 1, linestyles='dashed')
        plt.title("Performance Predictors Absolute Error on %s" % set_name);
        plt.xticks(rotation=45);
        plt.ylim(bottom=-0.01, top=0.1);
        plt.ylabel('absolute error');

        plt.tight_layout()
        plt.savefig(os.path.join(out_fld, 'abs_error_%s.png' % set_name), bbox_inches='tight')
        plt.close()

    metrics = [('r2_score', all_runs_r2),
               ('likelihood', all_runs_likelihood),
               ('within_ci_mae', all_runs_within_ci_mae),
               ('within_ci_accuracy', all_runs_within_ci_accuracy),
               ('auc_sigma_accuracy', all_runs_auc_sigma_acc),
               ('auc_sigma_mae', all_runs_auc_sigma_mae)]

    for name, all_runs_metric in metrics:
        plt.figure(figsize=(12, 8));
        for i, lab in enumerate(labels):
            metric_all = []
            for set_name in all_runs_metric:
                metric_all.append(all_runs_metric[set_name][:, i])
            y = np.mean(metric_all, axis=1)
            lo = np.quantile(metric_all, 0.1, axis=1)
            hi = np.quantile(metric_all, 0.9, axis=1)
            plt.plot(y, label=labels[i], color=pp_colors[i], marker='o', linewidth=2, markersize=8);
            plt.fill_between(np.arange(len(all_runs_metric.keys())), lo, hi, alpha=0.3, color=pp_colors[i])
        plt.legend(bbox_to_anchor=(1.4, 1.05));
        plt.xticks(range(len(all_runs_metric.keys())), all_runs_metric.keys());
        # plt.ylim(-0.1, 1.1);
        plt.hlines(y=0.0, xmin=0, xmax=len(all_runs_metric.keys()), linestyles='dashed');
        plt.title("%s of Performance Predictors" % name);
        plt.ylabel('%s' % name);

        plt.tight_layout()
        plt.savefig(os.path.join(out_fld, '%s.png' % name), bbox_inches='tight')
        plt.close()