import numpy as np
from enum import Enum
from scipy.stats import binom_test

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.svm import OneClassSVM
from sklearn.ensemble import RandomForestClassifier
from keras.utils import to_categorical
from keras.callbacks import ReduceLROnPlateau, EarlyStopping

import keras
import keras_resnet.models
from keras import optimizers
from keras.layers import Dense, Dropout
from keras.models import Sequential


class DomainClassifierModel(Enum):
    FFNNDCL = 1
    FLDA = 2
    RF = 3
    OCSVM = 4 # valid for two domains only


class MultiDomainClassifier(object):
    def __init__(self, orig_dims, dc=None):
        self.orig_dims = orig_dims
        self.dc = dc
        self.ratio = -1.0

        self.is_tabular = True
        if isinstance(self.orig_dims, tuple):
            if len(self.orig_dims) > 2:
                self.is_tabular = False
            else:
                self.orig_dims = self.orig_dims[0]

    # Shuffle two sets in unison, specifically used for data points and labels.
    def __unison_shuffled_copies(self, a, b, c=None, return_p=False):
        assert len(a) == len(b)
        p = np.random.permutation(len(a))
        a_out = a[p]
        b_out = b[p]
        c_out = None if c is None else c[p]
        if return_p:
            return a_out, b_out, c_out, list(p)
        else:
            return a_out, b_out, c_out

    # Partition the data set(s) for the difference classifier.
    def __prepare_difference_detector(self, list_x, list_y=None, balanced=True):

        n_domains = len(list_x)

        # Balancing makes testing easier.
        if balanced:
            min_len = np.min([len(x) for x in list_x])
            for i in range(n_domains):
                list_x[i] = list_x[i][:min_len]
                if list_y is not None:
                    list_y[i] = list_y[i][:min_len]

        # Extract halves from all sets
        list_x_first_half = []
        list_x_second_half = []
        list_y_first_half = []
        list_y_second_half = []
        for i, x in enumerate(list_x):
            list_x_first_half.append(x[:len(x) // 2, :])
            list_x_second_half.append(x[len(x) // 2:, :])
            if list_y is not None:
                y = list_y[i]
                list_y_first_half.append(y[:len(y) // 2])
                list_y_second_half.append(y[len(y) // 2:])

        # Recombine halves into new dataset, where samples from different domains are labeled with domain_id (0, 1, ...)
        x_train_new = np.concatenate(list_x_first_half, axis=0)
        y_train_new = np.concatenate([domain_id * np.ones(len(x)) for domain_id, x in enumerate(list_x_first_half)],
                                     axis=0)

        x_test_new = np.concatenate(list_x_second_half, axis=0)
        y_test_new = np.concatenate([domain_id * np.ones(len(x)) for domain_id, x in enumerate(list_x_second_half)],
                                    axis=0)

        if list_y is not None:
            y_train_old = np.concatenate(list_y_first_half, axis=0)
            y_test_old = np.concatenate(list_y_second_half, axis=0)
        else:
            y_train_old = None
            y_test_old = None

        self.ratio = len(list_x_first_half[0]) / len(x_train_new)

        x_train_new, y_train_new, y_train_old = self.__unison_shuffled_copies(x_train_new, y_train_new, y_train_old)

        x_test_new, y_test_new, y_test_old = self.__unison_shuffled_copies(x_test_new, y_test_new, y_test_old)

        train_test_data = (x_train_new, y_train_new, y_train_old, x_test_new, y_test_new, y_test_old)

        return train_test_data

    def build_model(self, list_X, list_y=None, balanced=True):
        training_data = self.__prepare_difference_detector(list_X, list_y, balanced=balanced)
        if self.dc == DomainClassifierModel.FFNNDCL:
            return self.neural_network_difference_detector(training_data)
        elif self.dc == DomainClassifierModel.FLDA:
            return self.fisher_lda_difference_detector(training_data)
        elif self.dc == DomainClassifierModel.RF:
            return self.random_forest_difference_detector(training_data)
        else:
            raise NotImplementedError('Invalid MultiDomainClassifierModel input.')

    def fisher_lda_difference_detector(self, train_test_data):
        (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = train_test_data

        lda = LinearDiscriminantAnalysis()
        lda.fit(X_tr_dcl, y_tr_dcl)
        score = lda.score(X_te_dcl, y_te_dcl)
        return lda, score, train_test_data

    def neural_network_difference_detector(self, train_test_data):
        (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = train_test_data

        if self.is_tabular:  # tabular data
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            early_stopper = EarlyStopping(min_delta=0.001, patience=10)
            batch_size = 128
            nb_classes = 2
            epochs = 100

            model = Sequential()
            model.add(Dense(64, input_dim=self.orig_dims, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(64, activation='relu'))
            model.add(Dropout(0.5))
            model.add(Dense(nb_classes, activation='softmax'))

            model.compile(loss='categorical_crossentropy',
                          optimizer='rmsprop',
                          metrics=['accuracy'])

            model.fit(X_tr_dcl, to_categorical(y_tr_dcl),
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(X_te_dcl, to_categorical(y_te_dcl)),
                      shuffle=True,
                      callbacks=[lr_reducer, early_stopper])

            score = model.evaluate(X_te_dcl, to_categorical(y_te_dcl))

        else:
            lr_reducer = ReduceLROnPlateau(factor=np.sqrt(0.1), cooldown=0, patience=5, min_lr=0.5e-6)
            early_stopper = EarlyStopping(min_delta=0.001, patience=10)
            batch_size = 128
            nb_classes = 2
            epochs = 200

            model = keras_resnet.models.ResNet18(keras.layers.Input(self.orig_dims), classes=nb_classes)
            model.compile(loss='categorical_crossentropy',
                          optimizer=optimizers.SGD(lr=0.001, decay=1e-6, momentum=0.9),
                          metrics=['accuracy'])

            model.fit(X_tr_dcl.reshape(len(X_tr_dcl), self.orig_dims[0], self.orig_dims[1], self.orig_dims[2]),
                      to_categorical(y_tr_dcl),
                      batch_size=batch_size,
                      epochs=epochs,
                      validation_data=(X_te_dcl.reshape(len(X_te_dcl), self.orig_dims[0], self.orig_dims[1],
                                                        self.orig_dims[2]), to_categorical(y_te_dcl)),
                      shuffle=True,
                      callbacks=[lr_reducer, early_stopper])

            score = model.evaluate(X_te_dcl.reshape(len(X_te_dcl), self.orig_dims[0], self.orig_dims[1],
                                                    self.orig_dims[2]), to_categorical(y_te_dcl))

        score = score[1]  # 0: loss 1: accuracy

        return model, score, train_test_data

    def random_forest_difference_detector(self, train_test_data):
        (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = train_test_data

        if self.is_tabular:  # tabular data

            n_estimators = 100
            criterion = 'gini'
            max_depth = None

            model = RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth, criterion=criterion)

            model.fit(X_tr_dcl, y_tr_dcl)

            score = model.score(X_te_dcl, y_te_dcl)
        else:
            raise NotImplementedError

        return model, score, train_test_data


class DomainClassifier(MultiDomainClassifier):
    def __init__(self, orig_dims, dc=None, sign_level=0.05):
        super().__init__(orig_dims, dc)
        self.sign_level = sign_level

    def build_model(self, list_X, list_y=None, balanced=True):
        if self.dc == DomainClassifierModel.OCSVM:
            train_test_data = self.__prepare_difference_detector(list_X, list_y, balanced=balanced)
            return self.one_class_svm(train_test_data)
        else:
            return super().build_model(list_X, list_y, balanced)

    def _test_accuracy_significance(self, y_true, y_pred_prob):
        # Get most anomalous indices sorted in descending order.
        # '1' is the new domain class
        most_conf_test_indices = np.argsort(y_pred_prob[:, 1])[::-1]
        most_conf_test_perc = np.sort(y_pred_prob[:, 1])[::-1]

        # Test whether classification accuracy is statistically significant.
        y_te_new_pred_argm = np.argmax(y_pred_prob, axis=1)
        errors = np.count_nonzero(y_true - y_te_new_pred_argm)
        successes = len(y_te_new_pred_argm) - errors

        p_val = binom_test(successes, len(y_te_new_pred_argm), self.ratio, alternative='greater')

        return p_val, most_conf_test_indices, most_conf_test_perc

    def accuracy_binomial_test(self, model, X_te_new, y_te_new):
        if self.dc == DomainClassifierModel.FFNNDCL:

            # Predict class assignments.
            if not self.is_tabular:
                X_te_new_res = X_te_new.reshape(len(X_te_new),
                                                self.orig_dims[0],
                                                self.orig_dims[1],
                                                self.orig_dims[2])
            else:
                X_te_new_res = X_te_new

            y_te_new_pred_probs = model.predict(X_te_new_res)

            p_val, most_conf_test_indices, most_conf_test_perc = self._test_accuracy_significance(y_te_new,
                                                                                                  y_te_new_pred_probs)

            return p_val, p_val < self.sign_level, most_conf_test_indices, most_conf_test_perc

        if self.dc == DomainClassifierModel.FLDA:
            y_te_new_pred_probs = model.predict_proba(X_te_new)

            p_val, most_conf_test_indices, most_conf_test_perc = self._test_accuracy_significance(y_te_new,
                                                                                                  y_te_new_pred_probs)

            return p_val, p_val < self.sign_level, most_conf_test_indices, most_conf_test_perc

        elif self.dc == DomainClassifierModel.OCSVM:
            y_pred_te = model.predict(X_te_new)
            novelties = X_te_new[y_pred_te == -1]
            return -1, novelties, None, len(novelties) > 0, -1

        elif self.dc == DomainClassifierModel.RF:

            # Predict class assignments.

            y_te_new_pred_probs = model.predict_proba(X_te_new)

            p_val, most_conf_test_indices, most_conf_test_perc = self._test_accuracy_significance(y_te_new,
                                                                                                  y_te_new_pred_probs)

            return p_val, p_val < self.sign_level, most_conf_test_indices, most_conf_test_perc

    def one_class_svm(self, train_test_data):
        (X_tr_dcl, y_tr_dcl, y_tr_old, X_te_dcl, y_te_dcl, y_te_old) = train_test_data
        svm = OneClassSVM()
        svm.fit(X_tr_dcl)
        return svm, -1, train_test_data

