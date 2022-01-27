# -*- coding: utf-8 -*-
import pandas as pd, numpy as np
from matplotlib import pyplot as plt
import os
import seaborn as sns
import matplotlib.colors as mcolors


def plot_metrics(out_fld, selected_ids, pp_colors, labels, metrics, all_runs_mae, figsize=(8, 5), suffix='',
                 ylim_top=0.1, disaggregate=False):
    plt.rcParams.update({'font.size': 16})
    all_runs_mae_mean = dict()
    for set_name in all_runs_mae:
        all_runs_mae_mean[set_name] = np.mean(all_runs_mae[set_name], axis=2)
        n_runs = all_runs_mae_mean[set_name].shape[0]

    palette = [pp_colors[i] for i in selected_ids]

    series_1 = pd.Series()
    series_2 = pd.Series()
    series_3 = pd.Series()
    series_4 = pd.Series()

    for i, pp in enumerate(labels[selected_ids]):
        for set_name in all_runs_mae_mean:
            series_1 = series_1.append(pd.Series(all_runs_mae_mean[set_name][:, selected_ids[i]]), ignore_index=True)
            series_2 = series_2.append(pd.Series([set_name] * n_runs), ignore_index=True)
            series_3 = series_3.append(pd.Series(np.arange(n_runs)), ignore_index=True)
            series_4 = series_4.append(pd.Series([pp.replace('(amazon)', '[3]').replace('(naver)', '[2]')] * n_runs),
                                       ignore_index=True)

    display_df = pd.DataFrame({'absolute error': series_1,
                               'set': series_2,
                               'run': series_3,
                               'Performance Predictor': series_4})

    plt.figure(figsize=figsize);
    # plt.title('Absolute Error of Performance Predictors')
    sns.boxplot(x="set", y="absolute error", hue="Performance Predictor",
                data=display_df, palette=palette, fliersize=0)
    plt.ylim(bottom=-0.001);  # , ylim_top);
    plt.legend();  # bbox_to_anchor=(1.4, 1.05));
    plt.tight_layout()
    plt.savefig(os.path.join(out_fld, 'abs_error' + suffix + '.png'), bbox_inches='tight')

    for name, all_runs_metric in metrics:
        plt.figure(figsize=(12, 8));
        for i, lab in enumerate(labels):
            if i in selected_ids:
                metric_all = []
                for set_name in all_runs_metric:
                    metric_all.append(all_runs_metric[set_name][:, i])
                y = np.mean(metric_all, axis=1)
                lo = np.quantile(metric_all, 0.1, axis=1)
                hi = np.quantile(metric_all, 0.9, axis=1)
                plt.plot(y, label=labels[i].replace('(amazon)', '[3]').replace('(naver)', '[2]'), color=pp_colors[i],
                         marker='o', linewidth=2, markersize=8);
                plt.fill_between(np.arange(len(all_runs_metric.keys())), lo, hi, alpha=0.3, color=pp_colors[i])
        plt.legend();  # bbox_to_anchor=(1.2, 1.05));
        plt.xticks(range(len(all_runs_metric.keys())), all_runs_metric.keys());
        # plt.ylim(-0.1, 1.1);
        # plt.hlines(y=0.0, xmin=0, xmax=len(all_runs_metric.keys()), linestyles='dashed');
        # plt.title("%s of Performance Predictors" % name);
        plt.ylabel('%s' % name.replace('auc_sigma_mae', r'$MAE_{CI}$').replace('within_ci_mae', r'$MAE_{CI_{0.05}}$'));

        plt.tight_layout()
        plt.savefig(os.path.join(out_fld, '%s%s.png' % (name, suffix)), bbox_inches='tight')

    # desaggregated results
    if disaggregate:

        for run in range(n_runs):

            series_1 = pd.Series()
            series_2 = pd.Series()
            series_3 = pd.Series()
            series_4 = pd.Series()

            for i, pp in enumerate(labels[selected_ids]):
                for set_name in all_runs_mae:
                    series_1 = series_1.append(pd.Series(all_runs_mae[set_name][run, selected_ids[i], :]),
                                               ignore_index=True)
                    series_2 = series_2.append(pd.Series([set_name] * all_runs_mae[set_name].shape[2]),
                                               ignore_index=True)
                    series_3 = series_3.append(pd.Series(np.arange(all_runs_mae[set_name].shape[2])), ignore_index=True)
                    series_4 = series_4.append(pd.Series(
                        [pp.replace('(amazon)', '[3]').replace('(naver)', '[2]')] * all_runs_mae[set_name].shape[2]),
                                               ignore_index=True)

            display_df = pd.DataFrame({'absolute error': series_1,
                                       'set': series_2,
                                       'run': series_3,
                                       'Performance Predictor': series_4})

            plt.figure(figsize=figsize);
            # plt.title('Absolute Error of Performance Predictors %d' % run)
            sns.boxplot(x="set", y="absolute error", hue="Performance Predictor",
                        data=display_df, palette=palette)
            plt.legend();  # bbox_to_anchor=(1.2, 1.05));
            plt.tight_layout()
            plt.savefig(os.path.join(out_fld, 'abs_error' + suffix + '_%d.png' % run), bbox_inches='tight')


def pp_viz(fld, out_fld, subpop=True, ylim_top=0.1, figsize=(8, 5), seed=42):
    labels = np.load(os.path.join(fld, 'model_names.npy'))
    n_pp = len(labels)

    all_runs_mae = dict()
    all_runs_r2 = dict()
    all_runs_within_ci_accuracy = dict()
    all_runs_likelihood = dict()
    all_runs_within_ci_mae = dict()
    all_runs_auc_sigma_acc = dict()
    all_runs_auc_sigma_mae = dict()

    np.random.seed(12345)

    # pp_colors = np.random.choice(list(mcolors.CSS4_COLORS), size=n_pp, replace=False)
    pp_colors = list(mcolors.CSS4_COLORS)[10::7]

    if subpop:
        # set_names = ['train', 'test_no_shift', 'test', 'test_unseen', 'test_unseen_subpop', 'test_natural']
        set_names = ['test_no_shift', 'test', 'test_unseen', 'test_unseen_subpop', 'test_natural']
    else:
        # set_names = ['train', 'test_no_shift', 'test', 'test_unseen', 'test_natural']
        set_names = ['test_no_shift', 'test', 'test_unseen', 'test_natural']

    new_set_names = dict()
    new_set_names['train'] = 'train'
    new_set_names['test_no_shift'] = 'no_shift'
    new_set_names['test'] = 'unseen_severity'
    new_set_names['test_unseen'] = 'perturbation_shift'
    new_set_names['test_unseen_subpop'] = 'subpop_shift'
    new_set_names['test_natural'] = 'natural'

    for set_name in set_names:
        all_runs_r2[new_set_names[set_name]] = np.load(os.path.join(fld, 'r2_score_%s.npy' % set_name))
        all_runs_within_ci_accuracy[new_set_names[set_name]] = np.load(
            os.path.join(fld, 'within_ci_accuracy_%s.npy' % set_name))
        all_runs_likelihood[new_set_names[set_name]] = np.load(os.path.join(fld, 'likelihood_%s.npy' % set_name))
        all_runs_mae[new_set_names[set_name]] = np.load(os.path.join(fld, 'abs_error_%s.npy' % set_name))
        all_runs_within_ci_mae[new_set_names[set_name]] = np.load(os.path.join(fld, 'within_ci_mae_%s.npy' % set_name))
        all_runs_auc_sigma_acc[new_set_names[set_name]] = np.load(
            os.path.join(fld, 'auc_sigma_accuracy_%s.npy' % set_name))
        all_runs_auc_sigma_mae[new_set_names[set_name]] = np.load(os.path.join(fld, 'auc_sigma_mae_%s.npy' % set_name))

    metrics = [('r2_score', all_runs_r2),
               ('likelihood', all_runs_likelihood),
               ('within_ci_mae', all_runs_within_ci_mae),
               ('within_ci_accuracy', all_runs_within_ci_accuracy),
               ('auc_sigma_accuracy', all_runs_auc_sigma_acc),
               ('auc_sigma_mae', all_runs_auc_sigma_mae)]

    # pp_list = ['Dummy', 'ExpertRF', 'ExpertRF (amazon)', 'ExpertRF (naver)', 'MultiDomainClassifier', 'ErrorPredictorRF', 'ErrorPredictorRF_no_shift', 'ErrorPredictorKNN',
    #       'ErrorPredictorGradBoost', 'DNN', 'DNN+Expert', 'ATC', 'MetaErrorPred', 'LODOErrorPredictor (oracle)', 'TargetDummy (oracle)', 'TargetErrorPredictor (oracle)']

    # Expert Models

    expert_models_ids = [1, 2, 3]

    plot_metrics(out_fld, expert_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_expert', ylim_top=ylim_top)

    # Error Predictor Models

    error_predictor_models_ids = [6, 8, 7, 5]

    plot_metrics(out_fld, error_predictor_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_errpred', ylim_top=ylim_top)

    # PP Comparison

    error_predictor_models_ids = [0, 1, 4, 9, 10, 11, 5, 12, 13, 14, 15]

    plot_metrics(out_fld, error_predictor_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_all_plus_oracle', ylim_top=ylim_top)

    # PP Comparison

    error_predictor_models_ids = [0, 1, 4, 9, 10, 11, 5, 12]

    plot_metrics(out_fld, error_predictor_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_all', ylim_top=ylim_top)

    # PP Comparison

    error_predictor_models_ids = [5, 12, 13, 14, 15]

    plot_metrics(out_fld, error_predictor_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_best_plus_oracle', ylim_top=ylim_top)

    # Paper

    paper_models_ids = [11, 1, 5, 12]

    plot_metrics(out_fld, paper_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_paper', ylim_top=ylim_top)

    # Paper 2

    paper_models_ids = [11, 2, 3, 5]

    plot_metrics(out_fld, paper_models_ids, pp_colors, labels, metrics, all_runs_mae, figsize=figsize,
                 suffix='_paper2', ylim_top=ylim_top)

