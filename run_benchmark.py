from drift_dac_experiments.base_parameters import DatasetParams, ExperimentParams
from drift_dac_experiments.pp_data_generation import pp_data_generation
from drift_dac_experiments.pp_comparison import pp_comparison
from drift_dac_experiments.pp_viz import pp_viz
import pandas as pd
import os


def run_benchmark(dataset_name):
    # create output folders

    data_fld = dataset_name
    pp_fld = dataset_name + '_pp'
    viz_fld = dataset_name + '_viz'

    os.makedirs(data_fld)
    os.makedirs(pp_fld)
    os.makedirs(viz_fld)

    # load experiment setting
    exp_params = ExperimentParams()

    dataset_df = pd.read_csv(dataset_name)
    dataset_params = DatasetParams(dataset_name + '.csv', dataset_df)

    # base model RF
    model = exp_params.list_of_models[1]

    n_runs = 10

    # generate shifted datasets (train, test_no_shift, test_unseen_severity, test_unseen_perturbation_shift,
    # test_unseen_subpop_shift, test_natural
    pp_data_generation(dataset_params, exp_params, model, n_runs, data_fld)

    # compare Performance Predictors
    pp_list = ['Dummy', 'ExpertRF', 'ExpertRF (amazon)', 'ExpertRF (naver)', 'MultiDomainClassifier',
               'ErrorPredictorRF', 'ErrorPredictorRF_no_shift', 'ErrorPredictorKNN', 'ErrorPredictorGradBoost', 'DNN',
               'DNN+Expert', 'ATC', 'MetaErrorPred', 'LODOErrorPredictor (oracle)', 'TargetDummy (oracle)',
               'TargetErrorPredictor (oracle)']

    pp_comparison(n_runs, pp_list, data_fld, pp_fld, keep_ratio_train_shifts=1., update=False)

    # generate plots
    pp_viz(pp_fld, viz_fld, subpop=True, ylim_top=0.1, figsize=(12, 5), seed=40)  # seed fixes the color palette


if __name__ == "__main__":

    datasets = [
        'adult',
        'video_games',
        'heart',
        'bank',
        'dont_get_kicked',
        'Churn_Modelling',
        'bng_zoo',
        'jsbach_chorals_modified',
        'SDSS',
        'bng_ionosphere',
        'network_intrusion_detection',
        'artificial_characters',
        'default_of_credit_card_clients'
    ]

    for dataset_name in datasets:
        run_benchmark(dataset_name)
