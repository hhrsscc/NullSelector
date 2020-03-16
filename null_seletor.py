""" Null Seletor """
from warnings import warn
from collections import defaultdict

import numpy as np
from sklearn.base import BaseEstimator
from sklearn.utils import check_array, check_X_y
from sklearn.utils import shuffle, safe_mask
from sklearn.utils.validation import check_is_fitted


class NullSeletor(BaseEstimator):
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


    def get_selected_features(self):
        """ Get selected features """
        selected_features = []
        mask = self._get_support_mask()
        for index, support in enumerate(mask):
            if support:
                selected_features.append(self.features[index])
        return selected_features


    def transform(self, dataset):
        """
        Instead of from sklearn.feature_selection.base import SelectorMixin.
        SelectorMixin is now part of the private API.

        Reduce dataset to the selected features.

        Parameters
        ----------
        dataset : array of shape [n_samples, n_features]
            The input samples.

        Returns
        -------
        dataset_r : array of shape [n_samples, n_selected_features]
            The input samples with only the selected features.
        """

        dataset = check_array(dataset, dtype=None, accept_sparse='csr',
                              force_all_finite=True)
        mask = self.get_support()
        if not mask.any():
            warn("No features were selected: either the data is"
                 " too noisy or the selection test too strict.",
                 UserWarning)
            return np.empty(0).reshape((dataset.shape[0], 0))
        if len(mask) != dataset.shape[1]:
            raise ValueError("dataset has a different shape than during fitting.")
        return dataset[:, safe_mask(dataset, mask)]


    def get_support(self, indices=False):
        """
        Instead of from sklearn.feature_selection.base import SelectorMixin.
        SelectorMixin is now part of the private API.

        Get a mask, or integer index, of the features selected

        Parameters
        ----------
        indices : boolean (default False)
            If True, the return value will be an array of integers, rather
            than a boolean mask.

        Returns
        -------
        support : array
            An index that selects the retained features from a feature vector.
            If `indices` is False, this is a boolean array of shape
            [# input features], in which an element is True iff its
            corresponding feature is selected for retention. If `indices` is
            True, this is an integer array of shape [# output features] whose
            values are indices into the input feature vector.
        """
        mask = self._get_support_mask()
        return mask if not indices else np.where(mask)[0]
