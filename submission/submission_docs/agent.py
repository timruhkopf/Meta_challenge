import os
import random
import warnings
from abc import abstractmethod
from typing import List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.distributions as td
import torch.nn as nn
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as QuantileRegressor
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from torch.utils.data import DataLoader
from torch.utils.data import Dataset


# import seaborn as sns


def check_diversity(representation, title, epsilon=0.01):
    """

    :param representation: ndarray.
    :param title: name of the matrix
    :param epsilon: float: the value needed to exceed (should be close to zero)
    :raises: Warning if representation is not diverse
    """
    # Check for (naive) representation collapse by checking sparsity after
    # translation by 90% quantile
    translated = representation - np.quantile(representation, 0.9, axis=0)
    sparsity = (translated < epsilon).sum() / np.product(representation.shape)
    if sparsity >= 0.95:
        warnings.warn(f'The {title} representation is not diverse.')

        # Warning(f'The {title} representation is not diverse.')
        print(representation)


def check_or_create_dir(dir):
    # If folder doesn't exist, then create it.
    if not os.path.isdir(dir):
        os.makedirs(dir)
        print("created folder : ", dir)

    else:
        print(dir, "folder already exists.")


class Dataset_Gravity(Dataset):
    # Dataset_columns
    columns_descriptive = ["usage", "name"]
    columns_categorical = ["task", "target_type", "feat_type", "metric"]
    columns_binary = ['has_categorical', 'has_missing', 'is_sparse']
    columns_numerical = ['time_budget', 'feat_num', 'target_num', 'label_num', 'train_num', 'valid_num', 'test_num']

    # Encoder must be available across instances
    enc_cat = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_num = MinMaxScaler()

    # ensure that we will not ignore any observations for categorical variables
    enc_cat.fit(
        pd.concat([
            pd.DataFrame.from_dict(data={'task': {'carlo': 'binary.classification',
                                                  'christine': 'binary.classification',
                                                  'digits': 'multiclass.classification',
                                                  'dilbert': 'multiclass.classification',
                                                  'dorothea': 'binary.classification',
                                                  'evita': 'binary.classification',
                                                  'flora': 'regression',
                                                  'grigoris': 'multilabel.classification',
                                                  'newsgroups': 'multiclass.classification',
                                                  'robert': 'multiclass.classification',
                                                  'tania': 'multilabel.classification',
                                                  'waldo': 'multiclass.classification',
                                                  'wallis': 'multiclass.classification',
                                                  'yolanda': 'regression',
                                                  'adult': 'multilabel.classification',
                                                  'albert': 'binary.classification',
                                                  'alexis': 'multilabel.classification',
                                                  'arturo': 'multiclass.classification'},
                                         'target_type': {'carlo': 'Binary',
                                                         'christine': 'Binary',
                                                         'digits': 'Categorical',
                                                         'dilbert': 'Categorical',
                                                         'dorothea': 'Binary',
                                                         'evita': 'Categorical',
                                                         'flora': 'Numerical',
                                                         'grigoris': 'Categorical',
                                                         'newsgroups': 'Numerical',
                                                         'robert': 'Binary',
                                                         'tania': 'Binary',
                                                         'waldo': 'Categorical',
                                                         'wallis': 'Categorical',
                                                         'yolanda': 'Numerical',
                                                         'adult': 'Binary',
                                                         'albert': 'Categorical',
                                                         'alexis': 'Binary',
                                                         'arturo': 'Categorical'},
                                         'feat_type': {'carlo': 'Numerical',
                                                       'christine': 'Numerical',
                                                       'digits': 'Numerical',
                                                       'dilbert': 'Numerical',
                                                       'dorothea': 'Binary',
                                                       'evita': 'Numerical',
                                                       'flora': 'Numerical',
                                                       'grigoris': 'Numerical',
                                                       'newsgroups': 'Numerical',
                                                       'robert': 'Numerical',
                                                       'tania': 'Numerical',
                                                       'waldo': 'Mixed',
                                                       'wallis': 'Numerical',
                                                       'yolanda': 'Numerical',
                                                       'adult': 'Mixed',
                                                       'albert': 'Numerical',
                                                       'alexis': 'Numerical',
                                                       'arturo': 'Numerical'},
                                         'metric': {'carlo': 'pac_metric',
                                                    'christine': 'bac_metric',
                                                    'digits': 'bac_metric',
                                                    'dilbert': 'pac_metric',
                                                    'dorothea': 'auc_metric',
                                                    'evita': 'auc_metric',
                                                    'flora': 'a_metric',
                                                    'grigoris': 'auc_metric',
                                                    'newsgroups': 'pac_metric',
                                                    'robert': 'bac_metric',
                                                    'tania': 'pac_metric',
                                                    'waldo': 'bac_metric',
                                                    'wallis': 'auc_metric',
                                                    'yolanda': 'r2_metric',
                                                    'adult': 'f1_metric',
                                                    'albert': 'f1_metric',
                                                    'alexis': 'auc_metric',
                                                    'arturo': 'f1_metric'}}),
            pd.DataFrame.from_dict({'task': {'24': 'multilabel.classification',
                                             '25': 'multiclass.classification',
                                             '26': 'multilabel.classification',
                                             '27': 'binary.classification',
                                             '28': 'multilabel.classification',
                                             '29': 'multilabel.classification',
                                             '3': 'regression',
                                             '30': 'multiclass.classification',
                                             '31': 'multilabel.classification',
                                             '32': 'multilabel.classification',
                                             '33': 'binary.classification',
                                             '34': 'multiclass.classification',
                                             '35': 'multiclass.classification',
                                             '36': 'regression',
                                             '37': 'multilabel.classification',
                                             '38': 'regression',
                                             '39': 'multilabel.classification',
                                             '4': 'multilabel.classification',
                                             '41': 'multilabel.classification',
                                             '42': 'binary.classification',
                                             '43': 'binary.classification',
                                             '44': 'binary.classification',
                                             '45': 'multilabel.classification',
                                             '46': 'binary.classification',
                                             '47': 'binary.classification',
                                             '48': 'multiclass.classification',
                                             '5': 'multilabel.classification',
                                             '50': 'binary.classification',
                                             '51': 'multiclass.classification',
                                             '52': 'regression',
                                             '53': 'multiclass.classification',
                                             '54': 'multiclass.classification',
                                             '55': 'regression',
                                             '57': 'multilabel.classification',
                                             '58': 'multilabel.classification',
                                             '59': 'binary.classification',
                                             '6': 'binary.classification',
                                             '60': 'binary.classification',
                                             '63': 'multiclass.classification',
                                             '64': 'multilabel.classification',
                                             '65': 'regression',
                                             '66': 'multilabel.classification',
                                             '67': 'multiclass.classification',
                                             '68': 'binary.classification',
                                             '69': 'regression',
                                             '7': 'binary.classification',
                                             '70': 'multilabel.classification',
                                             '71': 'regression',
                                             '72': 'multiclass.classification',
                                             '73': 'multiclass.classification',
                                             '74': 'multilabel.classification',
                                             '75': 'multiclass.classification',
                                             '76': 'binary.classification',
                                             '77': 'multiclass.classification',
                                             '78': 'multiclass.classification',
                                             '79': 'multilabel.classification',
                                             '80': 'multilabel.classification',
                                             '81': 'multilabel.classification',
                                             '82': 'multilabel.classification',
                                             '85': 'multilabel.classification',
                                             '86': 'multiclass.classification',
                                             '87': 'multiclass.classification',
                                             '88': 'regression',
                                             '89': 'regression',
                                             '9': 'binary.classification',
                                             '90': 'binary.classification',
                                             '91': 'regression',
                                             '92': 'multilabel.classification',
                                             '93': 'binary.classification',
                                             '94': 'binary.classification',
                                             '95': 'binary.classification',
                                             '96': 'regression',
                                             '97': 'multiclass.classification',
                                             '98': 'binary.classification',
                                             '99': 'regression'},
                                    'target_type': {'24': 'Numerical',
                                                    '25': 'Binary',
                                                    '26': 'Categorical',
                                                    '27': 'Categorical',
                                                    '28': 'Numerical',
                                                    '29': 'Numerical',
                                                    '3': 'Categorical',
                                                    '30': 'Categorical',
                                                    '31': 'Numerical',
                                                    '32': 'Numerical',
                                                    '33': 'Binary',
                                                    '34': 'Numerical',
                                                    '35': 'Categorical',
                                                    '36': 'Categorical',
                                                    '37': 'Numerical',
                                                    '38': 'Binary',
                                                    '39': 'Numerical',
                                                    '4': 'Categorical',
                                                    '41': 'Binary',
                                                    '42': 'Categorical',
                                                    '43': 'Binary',
                                                    '44': 'Numerical',
                                                    '45': 'Categorical',
                                                    '46': 'Categorical',
                                                    '47': 'Binary',
                                                    '48': 'Numerical',
                                                    '5': 'Binary',
                                                    '50': 'Numerical',
                                                    '51': 'Binary',
                                                    '52': 'Binary',
                                                    '53': 'Categorical',
                                                    '54': 'Binary',
                                                    '55': 'Numerical',
                                                    '57': 'Categorical',
                                                    '58': 'Categorical',
                                                    '59': 'Numerical',
                                                    '6': 'Categorical',
                                                    '60': 'Binary',
                                                    '63': 'Categorical',
                                                    '64': 'Binary',
                                                    '65': 'Binary',
                                                    '66': 'Binary',
                                                    '67': 'Binary',
                                                    '68': 'Binary',
                                                    '69': 'Categorical',
                                                    '7': 'Numerical',
                                                    '70': 'Binary',
                                                    '71': 'Categorical',
                                                    '72': 'Categorical',
                                                    '73': 'Numerical',
                                                    '74': 'Numerical',
                                                    '75': 'Binary',
                                                    '76': 'Numerical',
                                                    '77': 'Binary',
                                                    '78': 'Categorical',
                                                    '79': 'Numerical',
                                                    '80': 'Numerical',
                                                    '81': 'Categorical',
                                                    '82': 'Binary',
                                                    '85': 'Categorical',
                                                    '86': 'Categorical',
                                                    '87': 'Categorical',
                                                    '88': 'Binary',
                                                    '89': 'Categorical',
                                                    '9': 'Binary',
                                                    '90': 'Numerical',
                                                    '91': 'Categorical',
                                                    '92': 'Categorical',
                                                    '93': 'Categorical',
                                                    '94': 'Numerical',
                                                    '95': 'Numerical',
                                                    '96': 'Categorical',
                                                    '97': 'Numerical',
                                                    '98': 'Numerical',
                                                    '99': 'Numerical'},
                                    'feat_type': {'24': 'Numerical',
                                                  '25': 'Categorical',
                                                  '26': 'Binary',
                                                  '27': 'Binary',
                                                  '28': 'Binary',
                                                  '29': 'Binary',
                                                  '3': 'Binary',
                                                  '30': 'Numerical',
                                                  '31': 'Categorical',
                                                  '32': 'Numerical',
                                                  '33': 'Categorical',
                                                  '34': 'Categorical',
                                                  '35': 'Categorical',
                                                  '36': 'Categorical',
                                                  '37': 'Binary',
                                                  '38': 'Categorical',
                                                  '39': 'Categorical',
                                                  '4': 'Binary',
                                                  '41': 'Binary',
                                                  '42': 'Mixed',
                                                  '43': 'Mixed',
                                                  '44': 'Mixed',
                                                  '45': 'Binary',
                                                  '46': 'Categorical',
                                                  '47': 'Categorical',
                                                  '48': 'Categorical',
                                                  '5': 'Numerical',
                                                  '50': 'Mixed',
                                                  '51': 'Binary',
                                                  '52': 'Binary',
                                                  '53': 'Numerical',
                                                  '54': 'Binary',
                                                  '55': 'Numerical',
                                                  '57': 'Numerical',
                                                  '58': 'Numerical',
                                                  '59': 'Numerical',
                                                  '6': 'Binary',
                                                  '60': 'Mixed',
                                                  '63': 'Categorical',
                                                  '64': 'Categorical',
                                                  '65': 'Mixed',
                                                  '66': 'Binary',
                                                  '67': 'Mixed',
                                                  '68': 'Mixed',
                                                  '69': 'Numerical',
                                                  '7': 'Numerical',
                                                  '70': 'Numerical',
                                                  '71': 'Numerical',
                                                  '72': 'Categorical',
                                                  '73': 'Categorical',
                                                  '74': 'Mixed',
                                                  '75': 'Numerical',
                                                  '76': 'Binary',
                                                  '77': 'Binary',
                                                  '78': 'Binary',
                                                  '79': 'Binary',
                                                  '80': 'Mixed',
                                                  '81': 'Mixed',
                                                  '82': 'Mixed',
                                                  '85': 'Categorical',
                                                  '86': 'Mixed',
                                                  '87': 'Categorical',
                                                  '88': 'Categorical',
                                                  '89': 'Numerical',
                                                  '9': 'Numerical',
                                                  '90': 'Numerical',
                                                  '91': 'Binary',
                                                  '92': 'Mixed',
                                                  '93': 'Mixed',
                                                  '94': 'Numerical',
                                                  '95': 'Categorical',
                                                  '96': 'Binary',
                                                  '97': 'Binary',
                                                  '98': 'Binary',
                                                  '99': 'Numerical'},
                                    'metric': {'24': 'auc_metric',
                                               '25': 'auc_metric',
                                               '26': 'f1_metric',
                                               '27': 'r2_metric',
                                               '28': 'f1_metric',
                                               '29': 'a_metric',
                                               '3': 'auc_metric',
                                               '30': 'r2_metric',
                                               '31': 'bac_metric',
                                               '32': 'r2_metric',
                                               '33': 'pac_metric',
                                               '34': 'bac_metric',
                                               '35': 'bac_metric',
                                               '36': 'a_metric',
                                               '37': 'auc_metric',
                                               '38': 'bac_metric',
                                               '39': 'auc_metric',
                                               '4': 'pac_metric',
                                               '41': 'f1_metric',
                                               '42': 'f1_metric',
                                               '43': 'a_metric',
                                               '44': 'a_metric',
                                               '45': 'r2_metric',
                                               '46': 'bac_metric',
                                               '47': 'f1_metric',
                                               '48': 'f1_metric',
                                               '5': 'r2_metric',
                                               '50': 'f1_metric',
                                               '51': 'f1_metric',
                                               '52': 'bac_metric',
                                               '53': 'bac_metric',
                                               '54': 'auc_metric',
                                               '55': 'a_metric',
                                               '57': 'f1_metric',
                                               '58': 'r2_metric',
                                               '59': 'auc_metric',
                                               '6': 'auc_metric',
                                               '60': 'f1_metric',
                                               '63': 'pac_metric',
                                               '64': 'f1_metric',
                                               '65': 'auc_metric',
                                               '66': 'bac_metric',
                                               '67': 'bac_metric',
                                               '68': 'bac_metric',
                                               '69': 'r2_metric',
                                               '7': 'bac_metric',
                                               '70': 'f1_metric',
                                               '71': 'f1_metric',
                                               '72': 'f1_metric',
                                               '73': 'r2_metric',
                                               '74': 'r2_metric',
                                               '75': 'f1_metric',
                                               '76': 'f1_metric',
                                               '77': 'bac_metric',
                                               '78': 'f1_metric',
                                               '79': 'bac_metric',
                                               '80': 'auc_metric',
                                               '81': 'a_metric',
                                               '82': 'r2_metric',
                                               '85': 'a_metric',
                                               '86': 'f1_metric',
                                               '87': 'auc_metric',
                                               '88': 'pac_metric',
                                               '89': 'auc_metric',
                                               '9': 'bac_metric',
                                               '90': 'r2_metric',
                                               '91': 'r2_metric',
                                               '92': 'bac_metric',
                                               '93': 'auc_metric',
                                               '94': 'f1_metric',
                                               '95': 'auc_metric',
                                               '96': 'a_metric',
                                               '97': 'auc_metric',
                                               '98': 'pac_metric',
                                               '99': 'pac_metric'}})
        ]))

    n_features = 0
    deselected = {}

    def __init__(self, dataset_meta_features, learning_curves, algorithms_meta_features,
                 no_competitors=11,
                 deselect=0, topk=10, deselection_metric='skew', seed=123456, ):
        """

        :param dataset_meta_features:
        :param learning_curves:
        :param algorithms_meta_features:
        :param no_competitors: number of datasets that are compared against. Notice that no_competitors = 2
        :param deselect: int number of algorithms to deselect using backward selection
        :param topk: int in deselection: number of topk performing algorithms to consider for selection
        :param deselection_metric: str. 'skew' or '0threshold'. Skewness based selection looks at how the skewness
        of the change in performance of topk across all datasets is improved towards improving the avg topk performance.
        0threshold removes the algorithm with minimal density for decreased avg-topk performance across datasets (i.e.
        the integral from -inf to 0 of that ecdf.
        is pairwise comparisons
        """
        self.no_competitors = no_competitors
        self.deselect = deselect
        self.preprocess(dataset_meta_features, learning_curves, algorithms_meta_features)

        if deselect > 0:
            self.deselected = self._reduce_algo_space(removals=deselect, k=topk, mode=deselection_metric)
            # redoit all again, so that the getitem never sees these algos
            self.preprocess_with_known_deselection(self.deselected, dataset_meta_features, learning_curves,
                                                   algorithms_meta_features)

        self.nA = len(self.algo_final_performances.columns)

        # needed for plotting
        self.raw_learning_curves = learning_curves
        self.raw_dataset_meta_features = dataset_meta_features

        # seeding
        random.seed(seed)
        np.random.seed(seed)

    def __len__(self):
        return len(self.datasets_meta_features)

    def __getitem__(self, item):
        """
        Compareset get logic:
        :param item: int. index in the range(0, len(self)) of a dataset that is to be optimized.
        :return: D0, D1, A0, A1
        D0: meta features of the dataset that we want to optimize
        A0: algo performances for D0
        D1: set of datasets' meta features to compare against (of length k)
        A1: respective algo performances for D1
        """
        # get the dataset & its performances
        D0 = self.datasets_meta_features[item]
        A0 = self.algo_thresholded_performances[item]

        # generate a random compare set
        # TODO move k to init
        # TODO seedit
        item_compareset = random.choices(list(set(range(self.nD)) - {item}), k=self.no_competitors)
        D1 = self.datasets_meta_features[item_compareset]
        A1 = self.algo_thresholded_performances[item_compareset]

        return D0, D1, A0, A1

    def preprocess(
            self,
            dataset_meta_features,
            learning_curves,
            algorithms_meta_features,
            k=3,
    ):
        """
        (1) Dataset Meta Features are selected based on being variable and transformed to tensor + normalized
        (2) Learning Curves: for each LC particular properties are read out such as e.g. final performance.
        (3) Algorithm Features: LC properties aggregated by algorithm
        (4) Dataset Features LC properties aggregated by dataset

        :param dataset_meta_features:
        :param learning_curves:
        :param algorithms_meta_features:
        :param k: (optional) set the k least performing algorithm performances to 0
        :return:
        """
        self.nD = len(dataset_meta_features.keys())

        # changing keys to int
        algorithms_meta_features = {k: v for k, v in algorithms_meta_features.items()}
        dataset_meta_features = {k: v for k, v in dataset_meta_features.items()}
        learning_curves = {k: {int(k1): v1 for k1, v1 in v.items()}
                           for k, v in learning_curves.items()}

        self._preprocess_meta_features(dataset_meta_features)
        self._preprocess_learning_curves(algorithms_meta_features, learning_curves)
        self._preporcess_scalar_properties(self.algo_learning_curves)
        self._preprocess_dataset_properties(learning_curves, dataset_meta_features)
        self._preprocess_thresholded_algo_performances(k=10)

    def preprocess_with_known_deselection(self, deselected, dataset_meta_features, learning_curves,
                                          algorithms_meta_features):
        """
        wrapper around preprocess to allow user to algin validation and test datasets in terms of algorithm
        deselection.
        """
        self.preprocess(dataset_meta_features,
                        {d: {a: curve for a, curve in algos.items() if int(a) not in deselected}
                         for d, algos in learning_curves.items()},
                        {k: v for k, v in algorithms_meta_features.items() if int(k) not in deselected})

    def _preprocess_learning_curves(self, algorithms_meta_features, learning_curves):
        """Enrich the learning curve objects with properties computed on the learning curve
        such as what the final performance is or when it reached the 90% convergence. etc."""
        # compute algorithm properties -----------------------------------------
        # compute learning curve properties.

        # find the 90% quantile in performance
        # ---> learn how fast an algo given a dataset complexity will approx converge?

        self.algo_learning_curves = {
            k: {} for k in algorithms_meta_features.keys()
        }
        for ds_id in learning_curves.keys():
            for algo_id, curve in learning_curves[ds_id].items():
                self.algo_learning_curves[str(algo_id)][ds_id] = curve

                # final performance
                curve.final_performance = curve.scores[-1]

                # how much in % of the final performance
                curve.convergence_share = curve.scores / curve.final_performance

                # stamp at which at least 90% of final performance is reached
                threshold = 0.9
                curve.convergence90_step = np.argmax(
                    curve.convergence_share >= threshold
                )
                curve.convergence90_time = curve.timestamps[curve.convergence90_step]
                curve.convergence90_performance = curve.scores[curve.convergence90_step]

    def _preporcess_scalar_properties(self, algo_curves):
        """
        calcualte scalar values for each algorithm, dataset combination

        :attributes: must be a single matrix [datasets, algorithms]
            -algo_convergences90_time: the time at which the algo reached its 90% performance
            on the respective dataset.
            -algo_90_performances: the algorithms observed performance, when
            it first surpassed the 90% threshold.
            -algo_final_performances: the final performance of algo on dataset

        """
        # read out the time of convergence 90%
        algo_convergences90_time = dict()
        algo_final_performances = dict()
        algo_90_performance = dict()
        for algo_id, curve_dict in algo_curves.items():
            algo_convergences90_time[algo_id] = pd.Series(
                {k: curve.convergence90_time for k, curve in curve_dict.items()})
            algo_final_performances[algo_id] = pd.Series(
                {k: curve.final_performance for k, curve in curve_dict.items()})
            algo_90_performance[algo_id] = pd.Series(
                {k: curve.convergence90_performance for k, curve in curve_dict.items()})

        # time at which the algo converged on the respective dataset
        self.algo_convergences90_time = pd.DataFrame(algo_convergences90_time)
        self.algo_convergences90_time.columns.name = 'Algorithms'
        self.algo_convergences90_time.index.name = 'Dataset'

        # how well did the algo perform on the respective dataset
        self.algo_final_performances = pd.DataFrame(algo_final_performances)
        self.algo_final_performances.columns.name = 'Algorithms'
        self.algo_final_performances.index.name = 'Dataset'

        # how was the performance at the timestamp that marks the at least 90% convergence
        self.algo_90_performances = pd.DataFrame(algo_90_performance)
        self.algo_90_performances.columns.name = 'Algorithms'
        self.algo_90_performances.index.name = 'Dataset'

    def _preprocess_meta_features(self, dataset_meta_features):
        """create a df with meta features of each dataset"""

        # Preprocess dataset meta data (remove the indiscriminative string variables)
        df = pd.DataFrame(
            list(dataset_meta_features.values()),
            index=dataset_meta_features.keys())

        binary_df = df[self.columns_binary].astype(float)

        # min-max normalization of numeric features
        numerical_df = df[self.columns_numerical].astype(float)
        numerical_df = self.enc_num.fit_transform(numerical_df)
        numerical_df = pd.DataFrame(
            numerical_df,
            columns=self.columns_numerical,
            index=dataset_meta_features.keys())

        categorical_df = df[self.columns_categorical]
        categorical_df = self.enc_cat.transform(categorical_df)
        categorical_df = pd.DataFrame(categorical_df, columns=self.enc_cat.get_feature_names(),
                                      index=dataset_meta_features.keys())

        self.datasets_meta_features_df = pd.concat([numerical_df, categorical_df, binary_df], axis=1)
        self.n_features = len(self.datasets_meta_features_df.columns)
        self.datasets_meta_features = torch.tensor(
            self.datasets_meta_features_df.values, dtype=torch.float32)

    def _preprocess_dataset_properties_meta_testing(self, dataset_meta_features):
        """create a dataset, that is of the exact same shape as the preprocessing"""
        # TODO check if dataset_meta_features is a single example : else: make it
        #  pd.Dataframe rather than a Series (and do the exact same as in preprocessing)
        df = pd.Series(dataset_meta_features).to_frame().T
        df = df[set(df.columns) - set(self.columns_descriptive)]

        binary_df = df[self.columns_binary].astype(float)

        # min-max normalization of numeric features
        numerical_df = df[self.columns_numerical].astype(float)
        numerical_df = self.enc_num.transform(numerical_df)
        numerical_df = pd.DataFrame(numerical_df, columns=self.columns_numerical)

        categorical_df = df[self.columns_categorical]
        categorical_df = self.enc_cat.transform(categorical_df)
        categorical_df = pd.DataFrame(categorical_df, columns=self.enc_cat.get_feature_names())

        self.datasets_meta_features_df = pd.concat([numerical_df, categorical_df, binary_df], axis=1)
        self.n_features = len(self.datasets_meta_features_df.columns)
        self.datasets_meta_features = torch.tensor(
            self.datasets_meta_features_df.values, dtype=torch.float32)

        return self.datasets_meta_features_df, self.datasets_meta_features

    def _preprocess_dataset_properties(self, learning_curves, dataset_meta_features):
        """
        Computes an object for each dataset and aggregates information for it.
        :param learning_curves:
        :return:
        """

        class Datum:
            # placeholder object to keep track of aggregation properties (instead of
            # all of the dictionaries flying around.
            pass

        self.dataset_learning_properties = {
            d: Datum() for d in learning_curves.keys()
        }
        b = 3  # number of points to konsider in the topk_available trials
        for d in learning_curves.keys():
            datum = self.dataset_learning_properties[d]
            datum.final_scores = []  # unconditional distribution
            datum.convergence_step = []
            datum.ranking = []
            datum.meta = dataset_meta_features[d]

            for a, curve in learning_curves[d].items():
                datum.convergence_step.append(curve.convergence90_step)
                # datum.convergence_avg_topk =
                datum.final_scores.append(curve.final_performance)

                datum.ranking.append(
                    (
                        a,
                        curve.final_performance,
                        curve.convergence90_step,
                        curve.convergence90_time,
                    )
                )

            datum.ranking = sorted(datum.ranking, key=lambda x: -x[1])

            datum.convergence_avg_topb = np.mean(
                [algotup[3] for algotup in datum.ranking[:b]]
            )
            datum.convergence_avg = np.mean(datum.convergence_step)

            datum.topk_available_trials = (
                    float(datum.meta["time_budget"]) / datum.convergence_avg_topb
            )

    def _preprocess_thresholded_algo_performances(self, k):
        """

        :param k: topk (how many algorithm performances are used for comparison in cosine similarity)
        all others that are not in topk performing on a dataset will be set to 0
        :return:
        """
        # order = self.datasets_meta_features_df.index
        # algo_performances = pd.DataFrame(
        #     [self.dataset_learning_properties[i].final_scores for i in order],
        #     index=order,
        # )  # index = dataset_id, column = algo_id
        #
        # self.algo_final_performances.columns

        # optional thresholding: remove k least performing
        algo_performances = torch.tensor(self.algo_final_performances.values, dtype=torch.float32)
        _, ind = torch.topk(algo_performances, k=k, largest=False, dim=1)
        for i_row, row in zip(ind, algo_performances):
            row[i_row] = 0

        self.algo_thresholded_performances = algo_performances

    def _reduce_algo_space(self, k=10, removals=5, mode='skew'):
        """
        Using backward elimination, remove the least performing algorithms.
        This helps reduce the complexity of the estimation problem.

        Core idea is to iteratively remove columns from self.algo_final_performances
        by choosing the ones that reduce the overall row score the least.
        """

        def calc_ecdf(data, n_bins=100):
            count, bins_count = np.histogram(data, bins=n_bins)

            # finding the PDF of the histogram using count values
            pdf = count / sum(count)
            cdf = np.cumsum(pdf)
            return bins_count[1:], cdf

        def calc_pdf_integral0(data, n_bins):
            """

            :param data:
            :param n_bins:
            :return: probability of below x=0
            """
            x, cdf = calc_ecdf(data, n_bins)
            return cdf[np.argmax(x >= 0)]

        df = self.algo_final_performances
        k_inv = len(df.columns) - k

        # init for selection
        deselected = set()
        remaining = set(df.columns)

        performances = {k: None for k in df.loc[:, remaining]}
        performance_baseline = df.max(axis=1) - \
                               df[np.argsort(df, axis=1) >= k_inv].mean(axis=1, )
        skew_baseline = performance_baseline.skew(axis=0)
        for r in range(removals):

            # compute which algo hurts the least to remove:
            for algo in remaining:
                temp_remaining = set(remaining) - {algo}
                current_df = df.loc[:, temp_remaining]

                # METRIC for DESELECTION
                # compute average performance increase (on topk algos) on a dataset (value) deselecting this algo (key)
                # An improvement can come from deselecting a below mean performing algo
                # A deterioration comes from removing above mean performing algo on that dataset
                consider = k_inv - len(deselected)  # rank threshold (minimum rank to get into topk)
                absolute_perf = current_df[np.argsort(current_df, axis=1) >= consider].mean(axis=1)
                performances[algo] = performance_baseline - (current_df.max(axis=1) - absolute_perf)

            if mode == 'skew':
                # remove the algo, that reduces the skewness of the baseline the most
                skew = pd.DataFrame(performances).skew(axis=0)
                deselected_algo = skew.index[np.argmin(skew_baseline - skew)]
                skew_baseline = skew[deselected_algo]

            elif mode == '0threshold':
                # remove the algo that has least density mass on reducing the overall
                # performance of datasets (compared to current baseline)
                deselected_algo = pd.Series({k: calc_pdf_integral0(v, n_bins=100)
                                             for k, v in performances.items()}).argmin()

            deselected.add(deselected_algo)
            remaining.remove(deselected_algo)
            performance_baseline = performances.pop(deselected_algo)

            # # plotting the change in baseline performances across datasets:
            # # this is the performance profile across remaining algos & datasets removing an algo.
            # for algo in performances.keys():
            #     # performances[algo].hist()
            #
            #     performances[algo].hist(density=True, histtype='step',
            #                             cumulative=True, label=str(algo), bins=100, )
            # plt.legend()
            # plt.title(f'Leave this algo out decrease in top-{k} avg. performance on a dataset')
            # plt.show()

        return deselected

    def plot_learning_curves(self, dataset_id=9):

        for id, curve in self.raw_learning_curves[str(dataset_id)].items():
            plt.plot(curve.timestamps, curve.scores, marker='o', linestyle='dashed', label=id)

        plt.title('Dataset {}'.format(dataset_id))
        plt.xlabel('Time')
        plt.ylabel('Performance')
        plt.legend(loc='upper right')
        plt.show()

    def plot_convergence90_time(self, normalized=True):

        for dataset_id, dataset_curves in self.raw_learning_curves.items():

            if normalized:
                # fixme: apparently time budget can be smaller than timestamp
                # unconditional_90time = [
                #     curve.convergence90_time / int(self.raw_dataset_meta_features[str(dataset_id)]['time_budget'])
                #     for curve in dataset_curves.values()]

                unconditional_90time = [
                    curve.convergence90_time / curve.timestamps[-1]
                    for curve in dataset_curves.values()]
                title = 'Unconditional Time Budget until 90% Convergence\n' \
                        'normalized by available budget'
            else:
                unconditional_90time = [curve.convergence90_time for curve in dataset_curves.values()]
                title = 'Unconditional Time Budget until 90% Convergence'
            sns.kdeplot(unconditional_90time, label=dataset_id)

        plt.title(title)
        plt.legend(loc='upper right')
        plt.show()


class BaseEncoder(nn.Module):
    """
    Base class for encoders thtat are used to get latent representations
    of the data and the algorithms.
    """

    def __init__(self) -> None:
        super(BaseEncoder, self).__init__()

    def _build_network(self) -> None:
        """
        Bulid the network.
        """
        raise NotImplementedError

    def encode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Encodes the context.
        """
        raise NotImplementedError

    def decode(self, input: torch.Tensor) -> torch.Tensor:
        """
        Decodes the context.
        """
        raise NotImplementedError

    @abstractmethod
    def forward(self, D) -> torch.Tensor:
        """
        Forward path through the network to get the encoding
        """
        pass

    @abstractmethod
    def loss_gravity(self) -> torch.Tensor:
        """
        Loss function for gravity based training.
        """
        pass

    @abstractmethod
    def predict_algorithms(self) -> torch.Tensor:
        """
        Predict the algorithms
        """
        pass


class AE(BaseEncoder):
    # TODO allow for algo meta features
    def __init__(
            self,
            input_dim: int = 10,
            hidden_dims: List[int] = [8, 4],
            embedding_dim: int = 2,
            weights=[1.0, 1.0, 1.0, 1.0],
            repellent_share=0.33,
            n_algos=20,
            device=None,
    ):
        """

        :param nodes: list of number of nodes from input to output
        :param weights: list of floats indicating the weights in the loss:
        reconstruction, algorithm pull towards datasets, data-similarity-attraction,
        data-dissimilarity-repelling.
        :param n_algos: number of algorithms to place in the embedding space
        """
        super().__init__()
        self.device = device
        weights = [weights[0], *weights[2:], weights[1]]  # fixme: change the doc instead!
        self.weights = torch.tensor(weights).to(device)
        self.repellent_share = repellent_share

        # construct the autoencoder
        self.latent_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self._build_network()
        self.cossim = torch.nn.CosineSimilarity(dim=1, eps=1e-08)

        # initialize the algorithms in embedding space
        self.n_algos = n_algos
        self.embedding_dim = self.latent_dim
        self.Z_algo = nn.Parameter(
            td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim])
        )

        self.to(self.device)

    def _build_network(self) -> None:
        """
        Builds the encoder and decoder networks
        """
        # Make the Encoder
        modules = []

        hidden_dims = self.hidden_dims
        input_dim = self.input_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.latent_dim))
        modules.append(nn.BatchNorm1d(self.latent_dim))
        modules.append(nn.Dropout(p=0.5))
        modules.append(nn.ReLU())

        self.encoder = torch.nn.Sequential(*modules)

        # Make the decoder
        modules = []

        hidden_dims.reverse()
        input_dim = self.latent_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.input_dim))
        modules.append(nn.Sigmoid())  # input_dim, self.input_dim

        self.decoder = nn.Sequential(*modules)

    def encode(self, x):
        return self.encoder(x)

    def decode(self, x):
        return self.decoder(x)

    def forward(self, D):
        """
        Forward path through the meta-feature autoencoder
        :param D: input tensor
        :return: tuple: output tensor
        """
        return self.decode(self.encode(D))

    def _loss_reconstruction(self, D0, D0_fwd, *args, **kwargs):
        # reconstruction loss (Autoencoder loss)
        # its purpose is to avoid simple single point solution with catastrophic
        # information loss - in the absence of a repelling force.
        return torch.nn.functional.mse_loss(D0, D0_fwd)

    def _loss_datasets(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):

        reconstruction = self._loss_reconstruction(D0, D0_fwd)
        # batch similarity order + no_repellents
        no_comparisons = A1.shape[1]
        similarity_order_ind = torch.stack([torch.argsort(self.cossim(a0, a1)) for a0, a1 in zip(A0, A1)])
        no_repellent = int(no_comparisons * self.repellent_share)

        # find the repellent forces
        repellents = similarity_order_ind[:, :no_repellent]
        Z1_repellents = torch.stack([z1[r] for z1, r in zip(Z1_data, repellents)])
        A1_repellents = torch.stack([a1[r] for a1, r in zip(A1, repellents)])
        mutual_weighted_dist = [(1 - self.cossim(a0, a1)) @ torch.linalg.norm((z0 - z1), dim=1)
                                for z0, z1, a0, a1 in zip(Z0_data, Z1_repellents, A0, A1_repellents)]
        data_repellent = (len(Z1_data) * len(Z1_repellents[0])) ** -1 * sum(mutual_weighted_dist)

        # find the attracting forces
        attractors = similarity_order_ind[:, no_repellent:]
        Z1_attractors = torch.stack([z1[att] for z1, att in zip(Z1_data, attractors)])
        A1_attractors = torch.stack([a1[att] for a1, att in zip(A1, attractors)])
        mutual_weighted_dist = [self.cossim(a0, a1) @ torch.linalg.norm((z0 - z1), dim=1)
                                for z0, z1, a0, a1 in zip(Z0_data, Z1_attractors, A0, A1_attractors)]
        data_attractor = (len(Z1_data) * len(Z1_attractors[0])) ** -1 * sum(mutual_weighted_dist)

        return torch.stack([data_attractor, (-1) * data_repellent, reconstruction]) @ self.weights[:3]

    def _loss_algorithms(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):
        # Algorithm performance "gravity" towards dataset (D0: calcualted batchwise)
        # TODO check that direction (sign) is correct!
        # compute the distance between algos and D0 (batch) dataset in embedding space
        # and weigh the distances by the algorithm's performances
        # --> pull is normalized by batch size & number of algorithms
        # Fixme: make list comprehension more pytorch style by apropriate broadcasting
        # TODO use torch.cdist for distance matrix calculation!
        dataset_algo_distance = [a @ torch.linalg.norm((z - Z_algo), dim=1) for z, a in zip(Z0_data, A0)]
        return (len(Z_algo) * len(Z0_data)) ** -1 * sum(dataset_algo_distance)

    def loss_gravity(self, D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo):
        """
        Creates a pairwise (dataset-wise) loss that
        a) enforces a reconstruction of the datasets meta features (i.e. we
        have a meaningful embedding)
        b) ensure, that algorithms that perform good on datasets are drawn towards
        those datasets in embedding space.
        c) pull together datasets, if similar algorithms performed well on them.

        # Consider: use of squared/linear/learned exponential (based on full
        # prediction: top_k selection) algo performance for weighing?
        # Consider: that the 90% performance should also be part of the loss
        # this might allow to get a ranking immediately from the distances in
        # embedding space!

        :param D0: Dataset 0 meta features
        :param D0_fwd: autoencoder reconstruction of Dataset 0 meta features
        :param Z0_data: embedding of dataset 0 meta features
        :param Z1_data: embedding of dataset 1 meta features
        :param A0: vector of algorithm performances on dataset 0
        :param A1: vector of algorithm performances on dataset 1
        :param Z_algo: algorithm embedding vector of same dim as Z_data


        :return: scalar.
        """

        algo_pull = self._loss_algorithms(D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo)

        gravity = self._loss_datasets(
            D0, D0_fwd, Z0_data, Z1_data, A0, A1, Z_algo)

        return torch.stack([gravity, self.weights[-1] * algo_pull, ])

    def train_gravity(self, train_dataloader, test_dataloader, epochs, lr=0.001):
        """
        Two step training:
        1) Pretraining using reconstruction loss
        2) Training using gravity loss (reconstruction + attraction + repellent +
        :param train_dataloader:
        :param test_dataloader:
        :param epochs: list of int (len 2): epochs for step 1) and 2) respectively
        :param lr:
        :return:
        """
        name = self.__class__.__name__
        print(f'\nPretraining {name} with reconstruction loss: ')
        self._train(self._loss_reconstruction, train_dataloader, test_dataloader, epochs[0])

        print(f'\nTraining {name} with gravity loss:')
        return self._train(self.loss_gravity, train_dataloader, test_dataloader, epochs[1], lr=lr)

    def train_schedule(self, train_dataloader, test_dataloader, epochs=[100, 100, 100], lr=0.001):
        # Consider Marius idea to first find a reasonable data representation
        #  and only than train with the algorithms

        # pretrain
        name = self.__class__.__name__
        print(f'\nPretraining {name} with reconstruction loss:')
        self._train(self._loss_reconstruction, train_dataloader, test_dataloader, epochs[0], lr)

        # train datasets
        print(f'\nTraining {name} with dataset loss:')
        self._train(self._loss_datasets, train_dataloader, test_dataloader, epochs[1], lr)

        # train algorithms
        print(f'\nTraining {name} with algorithm:')
        return self._train(self._loss_algorithms, train_dataloader, test_dataloader, epochs[2], lr)

    def _train(self, loss_fn, train_dataloader, test_dataloader, epochs, lr=0.001):
        losses = []
        test_losses = []

        tracking = []
        optimizer = torch.optim.Adam(self.parameters(), lr)
        for e in range(epochs):
            for i, data in enumerate(train_dataloader):
                D0, D1, A0, A1 = data

                D0 = D0.to(self.device)
                D1 = D1.to(self.device)
                A0 = A0.to(self.device)
                A1 = A1.to(self.device)
                optimizer.zero_grad()

                # calculate embedding
                D0_fwd = self.forward(D0)

                # todo not recalculate the encoding
                Z0_data = self.encode(D0)
                Z1_data = torch.stack([self.encode(d) for d in D1])

                # look if there is representation collapse:

                # calculate "attracting" forces.
                loss = loss_fn(D0, D0_fwd, Z0_data, Z1_data, A0, A1, self.Z_algo)

                # gradient step
                loss.backward()
                optimizer.step()
                # TODO check convergence: look if neither Z_algo nor Z_data move anymore! ( infrequently)

            losses.append(loss)

            # validation every e epochs
            test_timer = 10
            test_losses = []
            # if e % test_timer == 0:
            #     # look at the gradient step's effects on validation data
            #     D_test = train_dataloader.dataset.datasets_meta_features
            #     D_test = D_test.to(self.device)
            #     Z_data = self.encode(D_test)
            #
            #     tracking.append((self.Z_algo.data.clone(), Z_data))

            # TODO validation procedure

        return tracking, losses, test_losses

    def predict_algorithms(self, D, topk):
        """
        Find the topk performing algorithm candidates.

        :param D: meta features of dataset D
        :param topk: number of candidate algorithms to return.
        :return: set of indicies representing likely good performing algorithms based on their
        distance in embedding space.
        """
        # embed dataset.
        self.eval()

        with torch.no_grad():
            Z_data = self.encode(D)

            # find k-nearest algorithms.
            # sort by distance in embedding space.
            dist_mat = torch.cdist(Z_data, self.Z_algo)
            _, top_algo = torch.topk(dist_mat, largest=False, k=topk)  # find minimum distance

        self.train()

        return top_algo


class VAE(AE):
    # TODO allow for algo meta features
    def __init__(
            self,
            input_dim: int = 10,
            hidden_dims: List[int] = [8, 4],
            embedding_dim: int = 2,
            weights: List[float] = [1.0, 1.0, 1.0, 1.0],
            repellent_share: float = 0.33,
            n_algos: int = 20,
            device=None,
    ):
        """

        :param nodes: list of number of nodes from input to output
        :param weights: list of floats indicating the weights in the loss:
        reconstruction, algorithm pull towards datasets, data-similarity-attraction,
        data-dissimilarity-repelling.
        :param n_algos: number of algorithms to place in the embedding space
        """
        super().__init__()
        self.device = device
        self.weights = torch.tensor(weights).to(device)
        self.repellent_share = repellent_share

        # construct the autoencoder
        self.embedding_dim = embedding_dim
        self.input_dim = input_dim
        self.hidden_dims = hidden_dims

        self._build_network()

        # initialize the algorithms in embedding space
        self.n_algos = n_algos
        self.embedding_dim = self.embedding_dim
        self.Z_algo = nn.Parameter(
            td.Uniform(-10, 10).sample([self.n_algos, self.embedding_dim])
        )

        self.to(self.device)

    def _build_network(self) -> None:
        """
        Builds the encoder and decoder networks
        """
        # Make the Encoder
        modules = []

        hidden_dims = self.hidden_dims
        input_dim = self.input_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        self.encoder = torch.nn.Sequential(*modules)

        # Mean and std_dev for the latent distribution
        self.fc_mu = torch.nn.Linear(hidden_dims[-1], self.embedding_dim)
        self.fc_var = torch.nn.Linear(hidden_dims[-1], self.embedding_dim)

        # modules.append(nn.Linear(input_dim, self.latent_dim))
        # modules.append(nn.BatchNorm1d(self.latent_dim))
        # modules.append(nn.Dropout(p=0.5))
        # modules.append(nn.ReLU())

        # Make the decoder
        modules = []

        hidden_dims.reverse()
        input_dim = self.embedding_dim

        for h_dim in hidden_dims:
            modules.append(nn.Linear(input_dim, h_dim))
            modules.append(nn.BatchNorm1d(h_dim))
            modules.append(nn.Dropout(p=0.5))
            modules.append(nn.ReLU())
            input_dim = h_dim

        modules.append(nn.Linear(input_dim, self.input_dim))
        modules.append(nn.Sigmoid())

        self.decoder = nn.Sequential(*modules)

    def reparameterize(
            self,
            mu: torch.Tensor,
            logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick to sample from N(mu, var)
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)

        return eps * std + mu

    def encode(self, x):

        # Forward pass the input through the network
        result = self.encoder(x)

        # Get the mean and standard deviation from the output
        mu = self.fc_mu(result)
        log_var = self.fc_var(result)

        # TODO: Plot latent distributions

        # Sample a latent vector using the reparameterization trick
        z = self.reparameterize(mu, log_var)

        return z

    def decode(self, x):
        return self.decoder(x)


class Agent:
    encoder_class = {'AE': AE, 'VAE': VAE}

    def __init__(
            self,
            number_of_algorithms,
            encoder: str = "AE",
            seed=123546,
            root_dir='',
            suggest_topk=5
    ):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        encoder : str
            The encoder to use.
            'AE' for the vanilla autoencoder
            'VAE' for the variational autoencoder

        seed : int
            The seed for the random number generator
        """

        self.nA = number_of_algorithms
        self.times = [0.] * self.nA
        self.encoder = encoder
        self.seed = seed

        self.root_dir = root_dir
        self.suggest_topk = suggest_topk
        self.counter = 0
        self.zero_flag = False


        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)

    def reset(
            self,
            dataset_meta_features,
            algorithms_meta_features
    ):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                usage = 'AutoML challenge 2014'
                name = name of the dataset
                task = ’binary.classification’, ’multiclass.classification’, ’multilabel.classification’, ’regression’
                target_type = ’Binary’, ’Categorical’, ’Numerical’
                feat_type = ’Binary’, ’Categorical’, ’Numerical’
                metric = ’bac’, ’auc’, ’f1’, ’pac’, ’abs’, ’r2’
                time_budget = total time budget for running algorithms on the dataset
                feat_num = number of features
                target_num = number of columns of target file (one, except for multi-label problems)
                label_num = number of labels (number of unique values of the targets)
                train_num = number of training examples
                valid_num = number of validation examples
                test_num = number of test examples
                has_categorical = whether there are categorical variable (yes=1, no=0)
                has_missing = whether there are missing values (yes=1, no=0)
                is_sparse = whether this is a sparse dataset (yes=1, no=0)

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of each algorithm:
                meta_feature_0 = 1 or 0
                meta_feature_1 = 0.1, 0.2, 0.3,…, 1.0

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'AutoML challenge 2014', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'f1_metric',
        'time_budget': '600', 'feat_num': '9', 'target_num': '6', 'label_num': '10',
        'train_num': '17', 'valid_num': '87', 'test_num': '72', 'has_categorical': '1',
        'has_missing': '0', 'is_sparse': '1'}

        >>> algorithms_meta_features
        {'0': {'meta_feature_0': '0', 'meta_feature_1': '0.1'},
         '1': {'meta_feature_0': '1', 'meta_feature_1': '0.2'},
         '2': {'meta_feature_0': '0', 'meta_feature_1': '0.3'},
         '3': {'meta_feature_0': '1', 'meta_feature_1': '0.4'},
         ...
         '18': {'meta_feature_0': '1', 'meta_feature_1': '0.9'},
         '19': {'meta_feature_0': '0', 'meta_feature_1': '1.0'},
         }
        """

        # preprocess the newly arriving dataset/algo features
        self.algorithms_meta_features = algorithms_meta_features
        dataset_meta_features_df_testing, dataset_meta_feature_tensor_testing = \
            self.valid_dataset._preprocess_dataset_properties_meta_testing(dataset_meta_features)

        dataset_meta_feature_tensor_testing = dataset_meta_feature_tensor_testing.to(self.model.device)

        # actual resetting
        self.times = {k: 0. for k in algorithms_meta_features.keys()}
        self.obs_performances = {k: 0. for k in algorithms_meta_features.keys()}

        # NOTE: Is this required in the RL setting?
        # set delta_t's (i.e. budgets for each algo we'd like to inquire)
        self.budgets = self.predict_convergence_speed(dataset_meta_features_df_testing)

        # predict the ranking of algorithms for this dataset
        self.learned_rankings = self.model.predict_algorithms(
            dataset_meta_feature_tensor_testing,
            topk=self.nA
        )[0].tolist()

        self.total_budget = int(dataset_meta_features['time_budget'])

    def meta_train(self,
                   dataset_meta_features,
                   algorithms_meta_features,
                   validation_learning_curves,
                   test_learning_curves,
                   # set up the encoder architecture
                   epochs=1000,
                   pretrain_epochs=1000,
                   batch_size=9,
                   n_compettitors=19,
                   lr=0.01527,
                   embedding_dim=5,
                   weights=[2.98744, 4.52075, 8.11511, 2.53756],
                   repellent_share=0.65758,
                   deselect=0, 
                   topk=17, 
                   deselection_metric='skew',
                   training='schedule'):
        """
        Start meta-training the agent with the validation and test learning curves

        :param datasets_meta_features : dict of dict of {str: str}
            Meta-features of meta-training datasets
        :param algorithms_meta_features : dict of dict of {str: str}
            The meta_features of all algorithms
        :param validation_learning_curves : dict of dict of {int : Learning_Curve}
            VALIDATION learning curves of meta-training datasets
        :param test_learning_curves : dict of dict of {int : Learning_Curve}
            TEST learning curves of meta-training datasets
        :param epochs:
        :param pretrain_epochs:
        :param batch_size:
        :param n_compettitors:
        :param lr:
        :param embedding_dim:
        :param weights:
        :param repellent_share:
        :param training: str. 'schedule': uses model.train_schedual
        'gravity' uses model.train_gravity

        Examples:
        To access the meta-features of a specific dataset:
        >>> datasets_meta_features['Erik']
        {'name':'Erik', 'time_budget':'1200', ...}

        To access the validation learning curve of Algorithm 0 on the dataset 'Erik' :

        >>> validation_learning_curves['Erik']['0']
        <learning_curve.Learning_Curve object at 0x9kwq10eb49a0>

        >>> validation_learning_curves['Erik']['0'].timestamps
        [196, 319, 334, 374, 409]

        >>> validation_learning_curves['Erik']['0'].scores
        [0.6465293662860659, 0.6465293748988077, 0.6465293748988145, 0.6465293748988159, 0.6465293748988159]

        """

        # validation dataloader
        self.valid_dataset = Dataset_Gravity(
            dataset_meta_features,
            validation_learning_curves,
            algorithms_meta_features,
            n_compettitors,
            deselect, topk, deselection_metric)

        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            shuffle=True,
            batch_size=batch_size
        )

        self.test_dataset = Dataset_Gravity(
            dataset_meta_features,
            test_learning_curves,
            algorithms_meta_features,
            n_compettitors)

        if len(self.valid_dataset.deselected) > 0:
            print(f'The algorithms {self.valid_dataset.deselected} have been deselected')
            # ensure test_data has exactly the same deselection of algorithms
            self.test_dataset.preprocess_with_known_deselection(
                self.valid_dataset.deselected,
                dataset_meta_features,
                test_learning_curves,
                algorithms_meta_features)
            self.nA = self.valid_dataset.nA

        self.test_dataloader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size)

        # meta_learn convergence speed
        self.meta_train_convergence_speed(confidence=0.2)

        # Training (algo-ranking) procedure
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = self.encoder_class[self.encoder](
            input_dim=self.valid_dataset.n_features,
            embedding_dim=embedding_dim,
            hidden_dims=[8, 4],
            weights=weights,
            repellent_share=repellent_share,
            n_algos=self.valid_dataset.nA,
            device=device,

        )

        if training == 'gravity':
            tracking, losses, test_losses = self.model.train_gravity(
                self.valid_dataloader, self.test_dataloader, epochs=[pretrain_epochs, epochs], lr=lr)

        elif training == 'schedule':
            tracking, losses, test_losses = self.model.train_schedule(
                self.valid_dataloader, self.test_dataloader, epochs=[pretrain_epochs, epochs, epochs], lr=lr)

        # TODO: checkpointing the model
        # run_id = hash()
        # TODO: append hashtable
        # self.output_dir = f'{self.root_dir}/output/{self.encoder}{run_id}'
        # check_or_create_dir(self.output_dir)
        #
        # torch.save(self.model, f'{self.output_dir}')

        # self.plot_encoder_training(losses)
        # self.plot_current_embedding()

        # TODO: Add Bandit exploration
        print()

    def meta_train_convergence_speed(self, confidence=0.9):
        """
        Using Quantile Regression, find the time at which the algorithm is suspected (
        with confidence (in %)) that the algorithm will have converged to 90% of
        its final performance on the given dataset.
        :param confidence: float: quantile of the 90% convergence distribution

        :attributes:
        :param qr_models: dict: {algo_id: QuantileRegression}
        independent models for each individual algorithm's convergence90 time at the
        confidence quantile.
        """

        # TODO: also use test dataset info for convergence speed learning
        print('Training 90% convergence speed.')
        X = self.valid_dataset.datasets_meta_features_df
        Y = self.valid_dataset.algo_convergences90_time

        # independent (algo-wise) quantile regression models
        self.qr_models = {k: None for k in Y.columns}
        for algo in Y.columns:
            self.qr_models[algo] = QuantileRegressor(loss='quantile', alpha=confidence)
            self.qr_models[algo].fit(X, Y[algo])

    def predict_convergence_speed(self, df):
        """Predict the 90% convergence time budget we would like to allocate for each algorithm
        requires meta_train_convergence_speed"""
        if not hasattr(self, 'qr_models'):
            raise ValueError('meta_train_convergence_speed must be executed beforehand')

        prediction_convergence_speed = {}
        for algo in range(self.nA):
            prediction_convergence_speed[int(algo)] = self.qr_models[str(algo)].predict(df)

        return prediction_convergence_speed

    @property
    def incumbent(self):
        inc = max(self.obs_performances, key=self.obs_performances.get)
        inc_perf = self.obs_performances[inc]

        return inc, inc_perf

    def suggest(self, observation):
        """
        Return a new suggestion based on the observation

        Parameters
        ----------
        observation : tuple of (int, float, float)
            The last observation returned by the environment containing:
                (1) A: the explored algorithm,
                (2) C_A: time has been spent for A
                (3) R_validation_C_A: the validation score of A given C_A

        Returns
        ----------
        action : tuple of (int, int, float)
            The suggested action consisting of 3 things:
                (1) A_star: algorithm for revealing the next point on its test learning curve
                            (which will be used to compute the agent's learning curve)
                (2) A:  next algorithm for exploring and revealing the next point
                       on its validation learning curve
                (3) delta_t: time budget will be allocated for exploring the chosen algorithm in (2)

        Examples
        ----------
        >>> action = agent.suggest((9, 151.73, 0.5))
        >>> action
        (9, 9, 80)
        """

        #trials = sum(1 if t != 0 else 0 for t in self.times.values())
                

        # If the counter is 0, initialize A_star = A
        if self.counter == 0:
            self.A_star = self.learned_rankings[self.counter]
            self.A = self.learned_rankings[self.counter]
        
        #Otherwise, only update A
        else: 

            if observation is not None:  # initial observation is None
                A, C_A, R = observation
                self.times[str(A)] += C_A
                self.obs_performances[str(A)] = R
                self.A = A
            # If A performed better than A_star, update A_star
            if self.obs_performances[str(self.A)] > self.obs_performances[str(self.A_star)]:
                self.A_star = self.A

            # Get the new value of A
            


        # Assign the time budget for the chosen algorithm
        if not self.zero_flag:
            self.A = self.learned_rankings[self.counter]
            delta_t = self.budgets[self.A][0] 
            #self.zero_flag = True
            action = (self.A_star, self.A, delta_t)
            self.counter += 1

        else:
            x = self.learned_rankings[self.counter]
            action = (self.A_star, x, 0.0)
            #self.zero_flag = False
       
        

        if self.counter == self.suggest_topk:
            self.counter = 0
            self.zero_flag = not self.zero_flag

        return action

    def plot_encoder_training(self, losses, ):
        # plot pretrain loss at each epoch.
        plt.figure()
        plt.plot(torch.tensor(losses).numpy(), label="gravity")
        plt.legend()

        plt.savefig(
            f'{self.root_dir}/output/{self.encoder}_training_loss.png',
        )

    def plot_current_embedding(self, normalize=False):
        """
        Plot the current embedding (both dataset & algorithms) based on the validation dataset.
        In the case of self.model.embedding_dim > 2, a projection is used.
        :param normalize: bool. Whether or not the 2D embeddings should be normalized
        using the dataset's mean and standard deviation.

        """
        # plot the dataset-algo embeddings 2D
        D_test = self.valid_dataloader.dataset.datasets_meta_features.data.to(self.model.device)
        d_test = self.model.encode(D_test)

        d_test = d_test.cpu().detach().numpy()
        z_algo = self.model.Z_algo.cpu().detach().numpy()

        check_diversity(d_test, 'Dataset')
        check_diversity(z_algo, 'Algorithm')

        if self.model.embedding_dim == 2:
            if normalize:
                d_test = (d_test - d_test.mean(axis=0)) / d_test.std(axis=0)
                # z_algo = (z_algo - z_algo.mean(axis=0)) / z_algo.std(axis=0)
                z_algo = (z_algo - d_test.mean(axis=0)) / d_test.std(axis=0)  # normalization based on datasets

            self._plot_embedding2d(d_test, z_algo)
            return None

        else:
            # fixme : fix umap collapse? -- when the representation is not diverse
            try:
                self._plot_embedding_projection(d_test, z_algo)
            except:
                pass  # if umap throghs an error ignore it!

    def _plot_embedding2d(self, d_test, z_algo):

        plt.figure()
        plt.scatter(d_test[:, 0], d_test[:, 1], label="datasets")
        plt.scatter(z_algo[:, 0], z_algo[:, 1], label="algorithms")
        plt.legend()
        plt.savefig(
            f'{self.root_dir}/output/{self.encoder}_training_embeddings.png',
        )

    def _plot_embedding_projection(self, d_test, z_algo, projection='umap'):
        """
        higher dimensional embeddings must be projected before being plotted.
        To do so, the datasets are used to learn a projection, in which the algorithm
        embeddings are projected. This way, this visualization can be used to see whether or not
        the learned embedding is a sensible projection.
        """

        if projection == 'umap':
            import umap

            trans = umap.UMAP(densmap=True).fit(d_test)

            # plot the current embedding
            # TODO add colouring c?
            plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s=5, cmap='Spectral')  # c=y_train,
            plt.title('Embedding of the training set by densMAP')

            # embedd the z_algos accordingly:
            # Consider: This embedding may not be sensible for z_algo!
            test_embedding = trans.fit_transform(z_algo)
            plt.scatter(test_embedding[:, 0], test_embedding[:, 1], s=5, cmap='Spectral')

            plt.savefig(
                f'{self.root_dir}/output/{self.encoder}_embedding_umap.png',
            )

        elif projection == 'pca':
            from sklearn.decomposition import PCA

            pca = PCA(n_components=2)
            d_test_pca = pca.fit_transform(d_test)
            z_algo_pca = pca.transform(z_algo)

            plt.scatter(d_test_pca[:, 0], d_test_pca[:, 1], s=5, cmap='Spectral')  # c=y_train,
            plt.scatter(z_algo_pca[:, 0], z_algo_pca[:, 1], s=5, cmap='Spectral')
            plt.title('Embedding of the training set by densMAP')

            plt.savefig(
                f'{self.root_dir}/output/{self.encoder}_embedding_pca.png',
            )


