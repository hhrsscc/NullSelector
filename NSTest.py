import lightgbm as lgb
from NullSeletor import NullSeletor
from sklearn.metrics import roc_auc_score
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier

if __name__ == '__main__':
    random_state = 7
    rate = 0.8
    x, y = make_classification(n_samples=200000, n_features=64, n_informative=16, n_redundant=4, random_state=random_state, flip_y=0.1)

    features = [f'feature_{i}' for i in range(x.shape[1])]
    train_length = int(x.shape[0] * rate)
    train_x, train_y = x[:train_length], y[:train_length]
    valid_x, valid_y = x[train_length:], y[train_length:]


    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(train_x, train_y)
    train_score = roc_auc_score(train_y, rf.predict_proba(train_x)[:, 1])
    valid_score = roc_auc_score(valid_y, rf.predict_proba(valid_x)[:, 1])
    print('RF model', train_score, valid_score)

    rf = RandomForestClassifier(random_state=random_state)
    selector = NullSeletor(rf)
    selector.fit(train_x, train_y, features)
    train_x = selector.transform(train_x)
    valid_x = selector.transform(valid_x)
    print('Selected Features', train_x.shape[1])

    rf = RandomForestClassifier(random_state=random_state)
    rf.fit(train_x, train_y)
    train_score = roc_auc_score(train_y, rf.predict_proba(train_x)[:, 1])
    valid_score = roc_auc_score(valid_y, rf.predict_proba(valid_x)[:, 1])
    print('Selected RF model', train_score, valid_score)


    train_x, train_y = x[:train_length], y[:train_length]
    valid_x, valid_y = x[train_length:], y[train_length:]

    clf = lgb.LGBMClassifier()
    clf.fit(train_x, train_y)
    train_score = roc_auc_score(train_y, clf.predict_proba(train_x)[:, 1])
    valid_score = roc_auc_score(valid_y, clf.predict_proba(valid_x)[:, 1])
    print('LGB model', train_score, valid_score)

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
    print('Selected LGB model', train_score, valid_score)