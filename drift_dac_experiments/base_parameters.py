from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression


class DatasetParams(object):
    # dataset-specific parameters
    def __init__(self, dataset_name, dataset_df):
        self.dataset_name = dataset_name
        self.dataset_df = dataset_df

        if dataset_name == 'adult':
            self.n_min_samples = 2000
            self.n_max_samples = 2000
            self.n_min_samples_target = 500
            self.n_samples_subdataset_augmentation = 100
            self.pos_class = '<=50K'
            self.target = 'class'
            self.split_variable = 'race'
            self.true_model = None
        elif dataset_name == 'video_games':
            self.n_min_samples = 1900
            self.n_max_samples = 1900
            self.n_min_samples_target = 475
            self.n_samples_subdataset_augmentation = 95
            self.pos_class = 'high'
            self.target = 'Global_Sales'
            self.split_variable = 'Genre'
            self.true_model = None
        elif dataset_name == 'bank':
            self.n_min_samples = 4000
            self.n_max_samples = 4000
            self.n_min_samples_target = 1000
            self.n_samples_subdataset_augmentation = 200
            self.pos_class = 'yes'
            self.target = 'default'
            self.split_variable = 'marital'
            self.true_model = None
        elif dataset_name == 'heart':
            self.n_min_samples = 4000
            self.n_max_samples = 4000
            self.n_min_samples_target = 1000
            self.n_samples_subdataset_augmentation = 200
            self.pos_class = 1
            self.target = 'cardio'
            self.split_variable = 'gender'
            self.true_model = None
        elif dataset_name == 'bng_zoo':
            self.n_min_samples = 4000
            self.n_max_samples = 4000
            self.n_min_samples_target = 1000
            self.n_samples_subdataset_augmentation = 200
            self.pos_class = 'mammal'
            self.target = 'type'
            self.split_variable = 'catsize'
            self.true_model = None
        elif dataset_name == 'jsbach_chorals_modified':
            self.n_min_samples = 1800
            self.n_max_samples = 1800
            self.n_min_samples_target = 450
            self.n_samples_subdataset_augmentation = 90
            self.pos_class = 'D_M'
            self.target = 'chord_label'
            self.split_variable = 'meter'
            self.true_model = None
        elif dataset_name == 'SDSS':
            self.n_min_samples = 1700
            self.n_max_samples = 1700
            self.n_min_samples_target = 425
            self.n_samples_subdataset_augmentation = 85
            self.pos_class = 'GALAXY'
            self.target = 'class'
            self.split_variable = 'camcol'
            self.true_model = None
        elif dataset_name == 'bng_ionosphere':
            self.n_min_samples = 2000
            self.n_max_samples = 2000
            self.n_min_samples_target = 500
            self.n_samples_subdataset_augmentation = 100
            self.pos_class = 'g'
            self.target = 'class'
            self.split_variable = 'a20'
            self.true_model = None
        elif dataset_name == 'artificial_characters':
            self.n_min_samples = 1900
            self.n_max_samples = 1900
            self.n_min_samples_target = 475
            self.n_samples_subdataset_augmentation = 95
            self.pos_class = 1
            self.target = 'Class'
            self.split_variable = 'V1'
            self.true_model = None
        elif dataset_name == 'default_of_credit_card_clients':
            self.n_min_samples = 4000
            self.n_max_samples = 4000
            self.n_min_samples_target = 1000
            self.n_samples_subdataset_augmentation = 200
            self.pos_class = 1
            self.target = 'default payment next month'
            self.split_variable = 'SEX'
            self.true_model = None
        else:
            raise NotImplementedError()

        self.dataset_df = self.dataset_df[self.dataset_df[self.target].notna()].reset_index(drop=True)


class ExperimentParams(object):
    # experiment-specific parameters
    def __init__(self):
        # define models
        self.list_of_models = [LogisticRegression(), RandomForestClassifier()]

        # runs
        self.n_rand_runs = 5

        # n source domains from which to run the whole experiment
        self.n_train_domains = 2

        self.verbose = True
        self.out_path = None
        self.save_dataset = False
        self.preprocess = None
        self.labelenc = None
        self.param_grid = None
        self.test_valid_size = 3. / 4
        self.test_size = 2. / 3

        self.n_training_datasets_per_shift = 100
        self.n_test_datasets_per_shift = 100
        self.n_unseen_datasets_per_shift = 100

        self.augment = False
        self.n_train_shift_types = 5
        self.train_many_severities = False

        self.all_training_shifts_combinations = False
        self.baseline_only = True
        self.train_on_subpop = False
