import numpy as np
import pandas as pd
from sklearn.metrics import f1_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier


data_paths = [r'C:\Users\Hassan\Documents\sampling_experiment\data.csv'] # list of paths to data sets
clf_cols = ['y'] # list of class col for each data set
sampling_tech_keys = ['random', 'stratified'] # holds names (sampling_keys) of sampling techs
sampling_percentages = [10, 20, 40, 80, 100]
classifiers = [LogisticRegression(),
               RandomForestClassifier(n_estimators=100, max_depth=2, random_state=0),
               SVC(C=1.0, cache_size=200, class_weight=None, coef0=0.0,
                   decision_function_shape='ovr', degree=3, gamma='auto', kernel='rbf',
                   max_iter=-1, probability=False, random_state=None, shrinking=True,
                   tol=0.001, verbose=False),
               GradientBoostingClassifier(),
               MLPClassifier(activation='relu', alpha=1e-05, batch_size='auto',
                              beta_1=0.9, beta_2=0.999, early_stopping=False,
                              epsilon=1e-08, hidden_layer_sizes=(10, 4),
                              learning_rate='constant', learning_rate_init=0.001,
                              max_iter=200, momentum=0.9,
                              nesterovs_momentum=True, power_t=0.5, random_state=1,
                              shuffle=True, solver='lbfgs', tol=0.0001,
                              validation_fraction=0.1, verbose=False, warm_start=False)
] # classification models to be tested

def sample_test_radom_fn(data, clf_col):
    y = data.loc[:, clf_col]
    X = data.loc[:, data.columns != clf_col]
    return train_test_split(X, y, test_size=0.2)

def sample_train_fn(X, y, sampling_key, percentage):
    if percentage == 100:
        return X, y
    if sampling_key == 'random' or sampling_key is None:
        x_train, _, y_train, _ = train_test_split(X, y, test_size=(100 -percentage)/100)
        return x_train, y_train
    elif sampling_key == 'stratified':
        x_train, _, y_train, _ = train_test_split(X, y, stratify=y, test_size=(100 - percentage) / 100)
        return x_train, y_train

def log_fn(clf, percentage, sampling_key,  f_score):
    #print('{} trained on {}% of data which is sampled by {} technique has achieved accuracy of {}'.format(type(clf).__name__, percentage, sampling_key, f_score))
    print('{}    {}%    {}    {}'.format(type(clf).__name__, percentage, sampling_key, f_score))

for path, clf_col in zip(data_paths, clf_cols):
    data = pd.read_csv(path, index_col=0)
    X, x_test, y, y_test = sample_test_radom_fn(data, clf_col)
    for clf in classifiers:
        for sampling_key in sampling_tech_keys:
            for percentage in sampling_percentages:
                x_train, y_train = sample_train_fn(X, y, sampling_key, percentage)
                clf.fit(x_train, y_train)
                y_pred = clf.predict(x_test)
                f_score = f1_score(y_test, y_pred, average='micro')
                log_fn(clf, percentage, sampling_key,  f_score)