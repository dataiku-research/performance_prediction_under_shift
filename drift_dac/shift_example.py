import numpy as np

from drift_dac.covariate_shift import ConstantCategorical, FlipSign
from drift_dac.prior_shift import KnockOut
from drift_dac.tests.perturbations_tests_utils import generate_synthetic_data
from drift_dac.perturbation_shared_utils import PerturbationConstants
from drift_dac.drift_metrics import compute_drift_metrics

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline

import copy


X, y, is_categorical = generate_synthetic_data()
# x is the samples x features array and contains both numeric and categorical variables (before preprocessing)
# y is the target variable
# is_categorical is a list of bool indicating whether a feature is categorical

# Split in source and target domain
X_src, X_tar, y_src, y_tar = train_test_split(X, y, test_size=0.5)

numeric_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())])
categorical_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore'))])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', numeric_transformer, np.where(~is_categorical)[0]),
        ('cat', categorical_transformer, np.where(is_categorical)[0])])

model = Pipeline(steps=[('preprocessor', preprocessor),
                        ('classifier', RandomForestClassifier())])

model.fit(X_src, y_src)
predictions_src = model.predict_proba(X_src)

list_of_shifts = [KnockOut(0, 0.4), ConstantCategorical(0.5, 0.5), FlipSign(0.5, 0.5)]

for shift in list_of_shifts:
    # Perturb the target domain
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
                                          X_perturbed, y_perturbed, predictions_tar,
                                          model, preprocessor=model.steps[0][1])

    print(shift.name)
    print(drift_metrics)

