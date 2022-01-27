import tensorflow.compat.v1 as tf

tf.disable_v2_behavior()

from drift_dac.domain_classifier import DomainClassifierModel, MultiDomainClassifier
from sklearn.base import BaseEstimator
from sklearn.metrics import mean_absolute_error, accuracy_score
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, LearningRateScheduler, ReduceLROnPlateau
from tensorflow.keras.layers import Input, LSTM, Dropout, Dense, Bidirectional, BatchNormalization
from tensorflow.keras.layers import GlobalAveragePooling1D, Layer, Concatenate
from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model as KerasModel
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.optimizers import Adam
from tensorflow.keras import regularizers

from tensorflow.random import set_seed as set_random_seed
from node.networks.layer import ObliviousDecisionTree
import numpy as np
import os
import math
import copy
from functools import partial


class MultiDomainPerformancePredictor(BaseEstimator):
    def __init__(self, n_features, n_domains, dc=None, multi_task=False):
        if dc is None:
            dc = DomainClassifierModel.RF
        self.n_features = n_features
        self.domain_clf = MultiDomainClassifier(n_features, dc=dc)
        self.domain_drop = np.zeros((n_domains,))
        self.n_domains = n_domains
        self.domain_clf_model = None
        self.multi_task = multi_task
        self.y_type_values = None

    def fit(self, list_X, list_y_drop, list_y_type, n_max_samples_per_domain=50000):

        # group X by drift type
        list_X_train = []
        self.y_type_values = np.unique(list_y_type)
        for i in range(self.n_domains):
            domain_idx = list_y_type == self.y_type_values[i]
            list_X_domain = np.vstack(list_X[domain_idx])

            n_samples = len(list_X_domain)
            if n_samples > n_max_samples_per_domain:
                selected_idx = np.random.choice(n_samples, size=n_max_samples_per_domain)
                list_X_domain = list_X_domain[selected_idx]

            list_X_train.append(list_X_domain)

        list_X_train = np.array(list_X_train)

        # build the Multi-Domain-Classifier to predict sample-level drift type
        self.domain_clf_model, score, train_test_data = self.domain_clf.build_model(list_X_train)
        (X_tr_dcl, y_tr_dcl, _, X_te_dcl, y_te_dcl, _) = train_test_data

        # fixed performance drop by drift type
        for i in range(self.n_domains):
            self.domain_drop[i] = list_y_drop[list_y_type == self.y_type_values[i]].mean()

    def predict_proba(self, list_X):

        n_datasets = len(list_X)
        y_pred_ds = []
        y_probas_ds = []
        for ds_i in range(n_datasets):
            y_proba_sample = self.domain_clf_model.predict_proba(list_X[ds_i])
            # probabilities for drift types
            y_probas_ds_i = np.mean(y_proba_sample, axis=0)
            # weighted average of performance drop by drift type (weights are the drift types probabilities)
            y_pred_ds_i = np.sum(y_probas_ds_i * self.domain_drop)
            y_pred_ds.append(y_pred_ds_i)
            y_probas_ds.append(y_probas_ds_i)

        if not self.multi_task:
            return y_pred_ds
        else:
            return y_pred_ds, y_probas_ds

    def predict(self, list_X):

        if not self.multi_task:
            return self.predict_proba(list_X)
        else:
            y_pred_ds, y_probas_ds = self.predict_proba(list_X)
            return y_pred_ds, self.y_type_values[np.argmax(y_probas_ds, axis=1)]

    def score(self, list_X, list_y_drop, list_y_type=None):
        if not self.multi_task:
            y_pred_ds = self.predict(list_X)
            return mean_absolute_error(list_y_drop, y_pred_ds)
        else:
            [y_pred_ds, y_type_ds] = self.predict(list_X)
            if list_y_type is None:
                return mean_absolute_error(list_y_drop, y_pred_ds)
            else:
                return [mean_absolute_error(list_y_drop, y_pred_ds),
                        accuracy_score(list_y_type, self.y_type_values[np.argmax(y_type_ds, axis=1)])]


class RandomSamplePermutation(Layer):
    def __init__(self, n_samples=500, **kwargs):
        super(RandomSamplePermutation, self).__init__(**kwargs)
        self.n_samples = n_samples

    def call(self, datasets, training=None):
        if not training:
            return datasets

        permuted_indices = np.random.permutation(self.n_samples)

        return datasets[:, permuted_indices, :]

    def get_config(self):
        return {"n_samples": self.n_samples,
                "name": 'rnd_permutation'}


class VectorizedODT(Layer):
    def __init__(self, n_trees=1,
                 depth=1,
                 units=100,
                 threshold_init_beta=1.,
                 name='vec_odt',
                 **kwargs):
        super(VectorizedODT, self).__init__(name=name, **kwargs)
        self.odt_layer = ObliviousDecisionTree(n_trees=n_trees,
                                               depth=depth,
                                               units=units,
                                               threshold_init_beta=threshold_init_beta)

    def call(self, input_numeric, training=None):
        dataset_encoder = tf.vectorized_map(self.odt_layer, input_numeric)

        return dataset_encoder

    def get_config(self):
        return {"n_trees": self.odt_layer.n_trees,
                "depth": self.odt_layer.depth,
                "units": self.odt_layer.units,
                "threshold_init_beta": self.odt_layer.threshold_init_beta,
                "name": self.name}


class MultiDomainPerformancePredictorDNN(MultiDomainPerformancePredictor):
    def __init__(self, n_samples, n_features, n_domains, n_layers=1, encoded_ds_size=100, hidden_size=100,
                 multi_task=False, loss_weight_perf_drop=1., loss_weight_drift_type=1., encoder_type='lstm',
                 n_meta_features=None, drop_branch=True, n_trees=3, depth=3, threshold_init_beta=1., lr=0.01):
        self.n_features = n_features
        self.n_samples = n_samples
        self.n_domains = n_domains
        self.n_layers = n_layers

        self.n_samples_per_dataset = n_samples
        self.n_features = n_features
        self.encoded_ds_size = encoded_ds_size
        self.hidden_size = hidden_size

        self.n_trees = n_trees
        self.depth = depth
        self.threshold_init_beta = threshold_init_beta

        self.multi_task = multi_task
        self.history = None
        self.loss_weight_perf_drop = loss_weight_perf_drop
        self.loss_weight_drift_type = loss_weight_drift_type
        self.lr = lr

        self.y_type_values = None

        valid_encoder_types = ['mlp', 'lstm', 'odt']
        if encoder_type not in valid_encoder_types:
            raise ValueError('Encoder type must be one of: %s.' % str(valid_encoder_types))
        self.encoder_type = encoder_type

        self.n_meta_features = n_meta_features
        self.drop_branch = drop_branch

        self.domain_drop = np.zeros((n_domains,))

        self.model = self._make_model()
        self._compile(self.lr)

    def _make_encoder(self, input_numeric):

        input_shape = (self.n_samples_per_dataset, self.n_features)

        if self.encoder_type == 'lstm':
            dataset_encoder = Bidirectional(
                LSTM(units=self.encoded_ds_size, return_sequences=False,
                     input_shape=input_shape), name='encoded_dataset')(input_numeric)

        elif self.encoder_type == 'mlp':
            dataset_encoder = input_numeric
            for i in range(self.n_layers):
                dataset_encoder = Dense(units=self.encoded_ds_size, kernel_initializer='normal',
                                        activation='relu', name='ds_dense_%d'%i)(dataset_encoder)

            dataset_encoder = GlobalAveragePooling1D()(dataset_encoder)
            #dataset_encoder = Dense(self.hidden_size, kernel_initializer='normal',
            #                        activation='relu', name='ds_avg_dense')(dataset_encoder)

        elif self.encoder_type == 'odt':

            dataset_encoder = VectorizedODT(n_trees=self.n_trees, depth=self.depth, units=self.encoded_ds_size,
                                            threshold_init_beta=self.threshold_init_beta)(input_numeric)
            dataset_encoder = GlobalAveragePooling1D()(dataset_encoder)

        else:
            raise NotImplementedError('Only lstm, mlp and odt are supported.')

        return dataset_encoder

    def _make_branches(self, encoded_data):

        if self.drop_branch:
            if not self.multi_task:
                # single branch for drop prediction
                encoded_data = Dropout(0.2, name='perf_drop_dropout')(encoded_data)
                performance_drop = Dense(1, kernel_initializer='normal', name='perf_drop')(encoded_data)
                output_model = performance_drop
            else:
                # two branches, one for drop; the other for drift type

                drift_type = Dense(units=self.n_domains, activation='softmax', name='drift_type')(encoded_data)
                performance_drop = Dense(1, activation='tanh', name='perf_drop', use_bias=False)(encoded_data)
                output_model = [performance_drop, drift_type]

        else:
            self.multi_task = True
            # single branch for drift type

            drift_type = Dense(units=self.n_domains, activation='softmax', name='drift_type')(encoded_data)

            output_model = drift_type

        return output_model

    def _compile(self, lr=0.01):
        if self.drop_branch:
            if not self.multi_task:
                self.model.compile(
                    loss='mean_absolute_error',
                    optimizer=Adam(lr),
                    metrics=['mean_absolute_error']
                )
            else:
                self.model.compile(
                    loss={'perf_drop': 'mean_squared_error', 'drift_type': 'categorical_crossentropy'},
                    loss_weights={'perf_drop': self.loss_weight_perf_drop,
                                  'drift_type': self.loss_weight_drift_type},
                    optimizer=Adam(lr),
                    metrics={'perf_drop': 'mean_absolute_error', 'drift_type': 'accuracy'}
                )
        else:
            self.model.compile(
                loss='categorical_crossentropy',
                optimizer=Adam(lr),
                metrics=['accuracy']
            )

    def _make_model(self) -> KerasModel:
        """
        This method is used to generate a Keras Model containing the DNN performance predictor
        :return: a compiled Keras Model object
        """

        input_numeric = Input(shape=(self.n_samples_per_dataset, self.n_features), name='dataset')

        input_numeric_permuted = RandomSamplePermutation(n_samples=self.n_samples_per_dataset, name='permuted_dataset')(
            input_numeric)

        dataset_encoder = self._make_encoder(input_numeric_permuted)

        if self.n_meta_features is not None:
            input_meta = Input(shape=(self.n_meta_features,), name='meta_features')
            encoded_meta = Dropout(0.5, name='meta_input_dropout')(input_meta)
            encoded_meta = Dense(self.hidden_size, kernel_initializer='normal',
                                 activation='relu', name='meta_dense',
                                 kernel_regularizer=regularizers.l2(0.1),
                                 bias_regularizer=regularizers.l2(0.1),
                                 activity_regularizer=regularizers.l2(0.1))(encoded_meta)
            encoded_data = Concatenate(axis=1, name='concat_dataset_meta_features')([dataset_encoder, encoded_meta])
        else:
            encoded_data = dataset_encoder

        output_model = self._make_branches(encoded_data)

        if self.n_meta_features is not None:
            model = KerasModel(inputs=[input_numeric, input_meta], outputs=output_model)
        else:
            model = KerasModel(inputs=input_numeric, outputs=output_model)

        return model

    def _step_decay(self, epoch):
        initial_lrate = self.lr
        drop = 0.5
        epochs_drop = 10.0
        lrate = initial_lrate * math.pow(drop, math.floor((1 + epoch) / epochs_drop))
        return lrate

    def _fixed_lr(self, epoch):
        return self.lr

    def fit(self, list_X, list_y_drop, list_y_type=None, random_state=1234, epochs=100, batch_size=256,
            validation_split=0.2, verbose=0, early_stop_patience=10, lr=None, list_X_meta=None,
            reduce_lr_plateau=True, validation_data=None):

        np.random.seed(random_state)
        set_random_seed(random_state)

        if list_y_type is not None:
            self.y_type_values = np.unique(list_y_type)
            if len(list_y_type.shape) == 1:
                list_y_type_ohe = to_categorical(list_y_type)
            else:  # else is a soft label
                list_y_type_ohe = copy.deepcopy(list_y_type)
                list_y_type = np.argmax(list_y_type_ohe, axis=1)

        if os.path.exists('./mdc_net.h5'):
            os.remove('./mdc_net.h5')

        if lr is not None:
            self.lr = lr
            self._compile(self.lr)

        if validation_split != 0 or (validation_data is not None):
            callbacks = [
                ModelCheckpoint(filepath='./mdc_net.h5', monitor='val_loss', verbose=verbose,
                                save_best_only=True, mode='min'),
                EarlyStopping(monitor='val_loss', mode='min', verbose=verbose, patience=early_stop_patience)]

            if reduce_lr_plateau:
                callbacks += [
                    ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=5, min_lr=self.lr * 0.01,
                                      verbose=verbose)]
            else:
                callbacks += [LearningRateScheduler(self._fixed_lr)]
        else:
            callbacks = [LearningRateScheduler(self._fixed_lr)]

        if self.n_meta_features is not None:
            input_X = [list_X, list_X_meta]
        else:
            input_X = list_X

        if self.drop_branch:
            if not self.multi_task:
                input_y = list_y_drop
            else:
                input_y = {'perf_drop': list_y_drop, 'drift_type': list_y_type_ohe}
        else:
            input_y = list_y_type_ohe

        self.history = self.model.fit(
            input_X,
            input_y,
            epochs=epochs,
            batch_size=batch_size,
            validation_split=validation_split,
            validation_data=validation_data,
            callbacks=callbacks,
            verbose=verbose
        )

        if validation_split != 0 or (validation_data is not None):
            self.model = load_model('./mdc_net.h5', custom_objects={'RandomSamplePermutation': RandomSamplePermutation,
                                                                    'VectorizedODT': VectorizedODT})

        if not self.drop_branch:
            # fixed performance drop by drift type
            for i in range(self.n_domains):
                self.domain_drop[i] = list_y_drop[list_y_type == self.y_type_values[i]].mean()

    def predict_proba(self, list_X, list_X_meta=None):
        if self.n_meta_features is not None:
            input_X = [list_X, list_X_meta]
        else:
            input_X = list_X

        if self.drop_branch:
            return self.model.predict(input_X)
        else:
            type_pred_proba = self.model.predict(input_X)
            # weighted average of performance drop by drift type (weights are the drift types probabilities)
            drop_pred = np.sum(type_pred_proba * np.reshape(self.domain_drop, (1, self.n_domains)), axis=1)

            return np.clip(drop_pred, -1., 1.), type_pred_proba

    def predict(self, list_X, list_X_meta=None):

        if not self.multi_task:
            drop_pred = self.predict_proba(list_X, list_X_meta)
            type_pred_proba = None
            type_pred = None
        else:
            drop_pred, type_pred_proba = self.predict_proba(list_X, list_X_meta)
            type_pred = np.argmax(type_pred_proba, axis=1)

        return np.clip(drop_pred, -1., 1.), type_pred


def resize_dataset(X, y=None, n_samples_out=500):
    # duplicate the n_samples till the size n_samples_out
    k = n_samples_out // X.shape[0]
    if k > 0 and X.shape[0] == n_samples_out // k:
        X_out = np.repeat(X, k, axis=0)
        if y is not None:
            y_out = np.tile(y, k)
    else:
        replication_ids = np.random.choice(X.shape[0], size=n_samples_out, replace=True)
        X_out = X[replication_ids, :]
        if y is not None:
            y_out = y[replication_ids]

    if y is not None:
        return X_out, y_out
    else:
        return X_out


def predict_performance_drop_type(performance_drop_predictor, X, list_X_meta=None):
    X = np.atleast_3d(X)
    if list_X_meta is not None:
        list_X_meta = np.atleast_2d(list_X_meta)

    if isinstance(performance_drop_predictor, MultiDomainPerformancePredictorDNN):
        n_samples = performance_drop_predictor.n_samples_per_dataset
        if X[0].shape[0] < n_samples:
            X = np.vectorize(partial(resize_dataset, y=None, n_samples_out=n_samples),
                             signature='(m,k)->(n,k)')(X)

        drops_pred, type_pred = performance_drop_predictor.predict(X, list_X_meta=list_X_meta)  # output drop/type
    elif isinstance(performance_drop_predictor, MultiDomainPerformancePredictor):
        drops_pred, type_pred = performance_drop_predictor.predict(X)  # output drop/type
    elif isinstance(performance_drop_predictor, BaseEstimator):
        drops_pred = performance_drop_predictor.predict(list_X_meta)
        type_pred = None
    else:
        raise NotImplementedError(
            "Supported models are sklearn, MultiDomainPerformancePredictor, MultiDomainPerformancePredictorDNN")
    return drops_pred, type_pred


def fit_performance_predictor(performance_drop_predictor, X, y_drop, y_type, list_X_meta=None,
                              **kwargs):
    X = np.atleast_3d(X)
    if list_X_meta is not None:
        list_X_meta = np.atleast_2d(list_X_meta)

    if isinstance(performance_drop_predictor, MultiDomainPerformancePredictorDNN):
        n_samples = performance_drop_predictor.n_samples_per_dataset
        if X[0].shape[0] < n_samples:
            X = np.vectorize(partial(resize_dataset, y=None, n_samples_out=n_samples),
                             signature='(m,k)->(n,k)')(X)

        performance_drop_predictor.fit(X, y_drop, y_type, list_X_meta=list_X_meta, **kwargs)
    elif isinstance(performance_drop_predictor, MultiDomainPerformancePredictor):
        performance_drop_predictor.fit(X, y_drop, y_type)
    elif isinstance(performance_drop_predictor, BaseEstimator):
        performance_drop_predictor.fit(list_X_meta, y_drop)
    else:
        raise NotImplementedError(
            "Supported models are sklearn, MultiDomainPerformancePredictor, MultiDomainPerformancePredictorDNN")