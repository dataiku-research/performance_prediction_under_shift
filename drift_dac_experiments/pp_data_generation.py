from drift_dac_experiments.viz_utils import name2type
import os
import numpy as np
import matplotlib.pyplot as plt
from drift_dac_experiments.perf_drop_experiment_utils import shift_generation_setting
from drift_dac_experiments.perf_drop_experiment_utils import multi_domain_data_generation, split_into_domains
import matplotlib.patches as mpatches
from scipy import stats
import pickle


def pp_data_generation(dataset_params, exp_params, model, n_runs, out_dir):

    for seed in range(n_runs):
        print('%d/%d' % (seed + 1, n_runs))

        # ## Synthetic Data Generation
        print("Generating Data")

        np.random.seed(seed)

        lists = shift_generation_setting(dataset_params, exp_params)
        list_of_train_shifts, list_of_test_shifts, list_of_unseen_shifts = lists

        list_of_shifts = list_of_test_shifts + list_of_unseen_shifts

        domain_values = split_into_domains(dataset_params.dataset_df, dataset_params.split_variable,
                                           dataset_params.n_min_samples, dataset_params.n_max_samples, seed=seed)

        list_of_drift_types = [name2type(sh.name) for sh in list_of_train_shifts]
        seen = set()
        list_of_drift_types = [x for x in list_of_drift_types if not (x in seen or seen.add(x))]

        # change here the source domain
        domain_value = domain_values[0]

        id_domain, split_variable, split_value, domain_size = domain_value

        if id_domain == exp_params.n_train_domains:
            break

        print('Domain Source: %s' % split_value)

        result_df = None
        out = multi_domain_data_generation(id_domain, split_value, domain_value, model,
                                           list_of_shifts, list_of_train_shifts, seed,
                                           list_of_drift_types, dataset_params, exp_params,
                                           result_df=result_df)
        train, test, test_unseen, test_natural, ref_task, result_df = out

        ref_accuracy = ref_task.model.score(ref_task.preprocess.transform(ref_task.X_src_orig), ref_task.y_src)

        alpha = 0.05
        n_samples = ref_task.y_src.shape[0]
        sigma = np.sqrt(ref_accuracy * (1 - ref_accuracy) / n_samples)
        ci_drop = stats.norm.ppf(1 - alpha / 2) * sigma

        print('Ref Acc: %.2f' % ref_accuracy)
        print('CI drop: %.2f' % ci_drop)

        filtered_list_of_train_shifts = train.shift_names
        filtered_list_of_test_shifts = test.shift_names
        filtered_list_of_unseen_shifts = test_unseen.shift_names

        list_of_drift_types = train.shift_types

        print("Saving")

        out = (train, test, test_unseen, test_natural, ref_task, None)
        with open(os.path.join(out_dir, './data_%d.pkl' % seed), 'wb') as f:
            pickle.dump(out, f)

        out = (
            list_of_drift_types, filtered_list_of_train_shifts, filtered_list_of_test_shifts,
            filtered_list_of_unseen_shifts)
        with open(os.path.join(out_dir, './shifts_%d.pkl' % seed), 'wb') as f:
            pickle.dump(out, f)

        print('Saving pictures')

        plt.boxplot([train.drops,
                     test.drops,
                     test_unseen.drops,
                     test_natural.drops],
                    labels=['train', 'test', 'unseen', 'natural']);
        plt.hlines(y=0.0, xmin=0, xmax=5, linestyles='dashed')
        plt.hlines(y=ci_drop, xmin=0, xmax=5, linestyles='dotted', color='darkred', alpha=0.6)
        plt.hlines(y=-ci_drop, xmin=0, xmax=5, linestyles='dotted', color='darkred', alpha=0.6)

        plt.savefig(os.path.join(out_dir, './true_drops_%d.png' % seed))
        plt.close()

        train_shifts_names = train.shift_types
        test_shifts_names = test.shift_types
        unseen_shifts_names = test_unseen.shift_types

        perturbation_shifts_ids = [i for i, sh in enumerate(filtered_list_of_unseen_shifts) if
                                   not any([sh_type in sh for sh_type in ('subsample', 'ko_shift')])]
        subpop_shifts_ids = [i for i, sh in enumerate(filtered_list_of_unseen_shifts) if
                             any([sh_type in sh for sh_type in ('subsample', 'ko_shift')])]

        all_unseen_shifts_names = [name2type(s) for s in filtered_list_of_unseen_shifts]
        unseen_shifts_names = [name2type(filtered_list_of_unseen_shifts[s]) for s in perturbation_shifts_ids]
        unseen_subpop_shifts_names = [name2type(filtered_list_of_unseen_shifts[s]) for s in subpop_shifts_ids]

        colors = ['lightblue', 'gold', 'lightgreen', 'darkgreen', 'darkred']

        plt.boxplot([train.drops[np.array(train_shifts_names) == t] for t in np.unique(train_shifts_names)],
                    labels=[s for s in np.unique(train_shifts_names)],
                    positions=[i for i in range(1, len(np.unique(train_shifts_names)) + 1)],
                    patch_artist=True, boxprops=dict(facecolor=colors[0], color=colors[0]));
        last_pos = len(np.unique(train_shifts_names))

        labels = [s.replace('_shift', '').replace('no', 'no_shift') for s in np.unique(train_shifts_names)]
        plt.boxplot([test.drops[np.array(test_shifts_names) == t] for t in np.unique(test_shifts_names)],
                    labels=[s + ' (test)' for s in np.unique(test_shifts_names)],
                    positions=[i + last_pos for i in range(1, len(np.unique(test_shifts_names)) + 1)],
                    patch_artist=True, boxprops=dict(facecolor=colors[1], color=colors[1]));

        last_pos += len(np.unique(test_shifts_names))
        labels += [s.replace('_shift', '').replace('no', 'no_shift') + ' (test)' for s in np.unique(test_shifts_names)]

        plt.boxplot([test_unseen.drops[np.array(all_unseen_shifts_names) == t] for t in np.unique(unseen_shifts_names)],
                    labels=[s for s in np.unique(unseen_shifts_names)],
                    positions=[i + last_pos for i in range(1, len(np.unique(unseen_shifts_names)) + 1)],
                    patch_artist=True, boxprops=dict(facecolor=colors[2], color=colors[2]));

        last_pos += len(np.unique(unseen_shifts_names))
        plt.xlim(0, last_pos + 1)
        labels += [s.replace('_shift', '').replace('gn', 'gaussian') for s in np.unique(unseen_shifts_names)]

        if unseen_subpop_shifts_names:
            plt.boxplot([test_unseen.drops[np.array(all_unseen_shifts_names) == t] for t in
                         np.unique(unseen_subpop_shifts_names)],
                        labels=[s for s in np.unique(unseen_subpop_shifts_names)],
                        positions=[i + last_pos for i in range(1, len(np.unique(unseen_subpop_shifts_names)) + 1)],
                        patch_artist=True, boxprops=dict(facecolor=colors[3], color=colors[3]));

            last_pos += len(np.unique(unseen_subpop_shifts_names))
            plt.xlim(0, last_pos + 1)
            labels += [s.replace('_shift', '').replace('_feature', '') for s in np.unique(unseen_subpop_shifts_names)]

        plt.boxplot([test_natural.drops],
                    labels=['natural'],
                    positions=[last_pos + 1],
                    patch_artist=True, boxprops=dict(facecolor=colors[4], color=colors[4]));

        last_pos += 1
        plt.xlim(0, last_pos + 1)

        labels += ['natural']

        plt.xticks(range(1, last_pos + 1), labels, rotation=60)

        plt.legend(
            handles=[mpatches.Patch(color=c, label=l) for c, l in
                     zip(colors, ['train', 'unseen_severity', 'unseen_shift', 'unseen_subpop_shift', 'natural'])]);

        # plt.title('True Drops by Type %d' % seed)
        plt.hlines(y=0.0, xmin=0, xmax=last_pos + 1, linestyles='dashed', alpha=0.6)
        plt.hlines(y=ci_drop, xmin=0, xmax=last_pos + 1, linestyles='dotted', color='darkred', alpha=0.6)
        plt.hlines(y=-ci_drop, xmin=0, xmax=last_pos + 1, linestyles='dotted', color='darkred', alpha=0.6)

        plt.tight_layout()
        plt.savefig(os.path.join(out_dir, './true_drops_by_type_%d.png' % seed), bbox_inches='tight')
        plt.close()