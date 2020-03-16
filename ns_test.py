""" test """

import warnings

from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

import lightgbm as lgb

from timer import MultiLayerTimer
from null_seletor import NullSeletor


warnings.filterwarnings("ignore")

def run_test():
    """ run """
    timer = MultiLayerTimer()
    random_state = 7
    dataset, target = make_classification(n_samples=200000, n_features=64, n_informative=16,
                                          n_redundant=4, random_state=random_state, flip_y=0.1)

    features = [f'feature_{i}' for i in range(dataset.shape[1])]
    train_length = int(dataset.shape[0] * 0.8)
    train_x, train_y = dataset[:train_length], target[:train_length]
    valid_x, valid_y = dataset[train_length:], target[train_length:]

    with timer.timer("random forest model"):
        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(train_x, train_y)
        train_score = roc_auc_score(train_y, rf_model.predict_proba(train_x)[:, 1])
        valid_score = roc_auc_score(valid_y, rf_model.predict_proba(valid_x)[:, 1])
        print('RF model', "train_score: ", train_score, "valid_score: ", valid_score)

        rf_model = RandomForestClassifier(random_state=random_state)
        selector = NullSeletor(rf_model)
        selector.fit(train_x, train_y, features)
        train_x = selector.transform(train_x)
        valid_x = selector.transform(valid_x)
        print('Selected Features', train_x.shape[1])

        rf_model = RandomForestClassifier(random_state=random_state)
        rf_model.fit(train_x, train_y)
        train_score = roc_auc_score(train_y, rf_model.predict_proba(train_x)[:, 1])
        valid_score = roc_auc_score(valid_y, rf_model.predict_proba(valid_x)[:, 1])
        print('Selected RF model', "train_score: ", train_score, "valid_score: ", valid_score)


    train_x, train_y = dataset[:train_length], target[:train_length]
    valid_x, valid_y = dataset[train_length:], target[train_length:]

    with timer.timer("LGB model"):
        clf = lgb.LGBMClassifier()
        clf.fit(train_x, train_y)
        train_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
        valid_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
        print('LGB model', "train_score: ", train_score, "valid_score: ", valid_score)

        clf = lgb.LGBMClassifier()
        selector = NullSeletor(clf)
        selector.fit(train_x, train_y, features)
        train_x = selector.transform(train_x)
        valid_x = selector.transform(valid_x)
        print('Selected Features', train_x.shape[1])

        clf = lgb.LGBMClassifier()
        clf.fit(train_x, train_y)
        train_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
        valid_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
        print('Selected LGB model', "train_score: ", train_score, "valid_score: ", valid_score)


if __name__ == '__main__':
    run_test()
