""" Null Seletor """
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.feature_selection.base import SelectorMixin
from sklearn.utils import check_X_y, shuffle
from sklearn.utils.validation import check_is_fitted


class NullSeletor(BaseEstimator, SelectorMixin):
    """ Feature selector use null importance """
    def __init__(self, model, choose_func=None, threshold=50, random_state=7):
        self.model = model
        self.random_state = random_state
        self.threshold = threshold
        if choose_func:
            self.choose_func = choose_func
        else:
            self.choose_func = self._get_support_features
        self.features = None
        self.actual_dict = None
        self.null_dict = None

    def get_importance_info(self, dataset, target, *model_args, **model_kwargs):
        """ Get model importance info """
        self.model.fit(dataset, target, *model_args, **model_kwargs)

        info_dict = {}
        for key, value in zip(self.features, self.model.feature_importances_):
            info_dict[key] = value

        return info_dict

    def fit(self, dataset, target, features, *model_args, n_iter=10, **model_kwargs):
        """ Learn Null Importance """

        dataset, target = check_X_y(dataset, target, accept_sparse=['csr', 'csc'], dtype=np.float32)
        assert dataset.shape[1] == len(features), "Invalid features length"

        self.features = features
        self.actual_dict = self.get_importance_info(dataset, target, *model_args, **model_kwargs)
        self.null_dict = defaultdict(list)
        shuffled_y = target.copy()
        for _ in range(n_iter):
            shuffled_y = shuffle(shuffled_y, random_state=self.random_state)
            info_dict = self.get_importance_info(dataset, shuffled_y, *model_args, **model_kwargs)
            for key, value in info_dict.items():
                self.null_dict[key].append(value)

        for key, value in self.null_dict.items():
            self.null_dict[key] = sorted(value)

    def _get_support_features(self, actual_importance, null_importances):
        return actual_importance > np.percentile(null_importances, self.threshold)


    def _get_support_mask(self):
        check_is_fitted(self, ['actual_dict', 'null_dict', 'features'])

        return np.array([self.choose_func(self.actual_dict[k], self.null_dict[k]) \
                         for k in self.features])
