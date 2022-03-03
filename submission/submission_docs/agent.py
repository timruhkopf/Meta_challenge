import matplotlib.pyplot as plt
import pandas as pd
import torch
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as QuantileRegressor
from torch.utils.data import DataLoader
import numpy as np

from sklearn.metrics import label_ranking_average_precision_score  as LRAP

import torch
import torch.nn as nn

import random

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats as ss
import seaborn as sns
import torch
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset


class Dataset_Gravity(Dataset):
    # Dataset_columns
    columns_descriptive = ["usage", "name"]
    columns_categorical = ["task", "target_type", "feat_type", "metric"]
    columns_binary = ['has_categorical', 'has_missing', 'is_sparse']
    columns_numerical = ['time_budget', 'feat_num', 'target_num', 'label_num', 'train_num', 'valid_num', 'test_num']

    # Encoder must be available across instances
    enc_cat = OneHotEncoder(sparse=False, handle_unknown='ignore')
    enc_num = StandardScaler()

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
    deselected = set()
    nA = None  # number of algorithms

    def __init__(self,
                 no_competitors=11,
                 seed=123456, ):
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

        # self.preprocess(dataset_meta_features, learning_curves, algorithms_meta_features)

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

    def preprocess_learning_curves(self, learning_curves):
        """
        facilitate later calculations on learning curves:
        For scalar calculations use self.lc and assign a new attribute for a calculated
        scalar property
        for algorithm aggregations use self.lc_algos
        for dataset aggregations use self.lc_datasets
        :param learning_curves:

        """
        ids_algos = learning_curves[list(learning_curves.keys())[0]].keys()
        self.ids_algos = [int(i) for i in ids_algos]
        self.ids_datasets = list(learning_curves.keys())

        self.lc = {(d, int(a)): lc for d, algos in learning_curves.items()
                   for a, lc in algos.items()}

        # Depreciate EDA prints
        # pd.set_option("display.max_rows", None, "display.max_columns", None)
        # print('\ntimestamp distribution')
        # print(pd.Series(len(lc.timestamps) for k, lc in self.lc.items()).value_counts().sort_index())
        #
        # print('\nsingle timestamp scores')
        # print(pd.Series(lc.scores for k, lc in self.lc.items() if len(lc.timestamps) == 1))

        # ensure correct ordering
        dataset_major = sorted(self.lc.keys(), key=lambda tup: tup[0])
        algo_major = sorted(self.lc.keys(), key=lambda tup: tup[1])

        # lookup object ids based on aggregation
        self.algos = {a: [k for k in dataset_major if k[1] == a]
                      for a in self.ids_algos}
        self.datasets = {d: [k for k in algo_major if k[0] == d]
                         for d in self.ids_datasets}

        # lookup actual object based on aggregation
        self.lc_algos = {a: [self.lc[k] for k in dataset_major if k[1] == a]
                         for a in self.ids_algos}
        self.lc_datasets = {d: [self.lc[k] for k in algo_major if k[0] == d]
                            for d in self.ids_datasets}

        # Depreciate remove this EDA
        # print('\nper algo distribution of timestamps')
        # print(pd.DataFrame({a_id: pd.Series([len(lc.timestamps) for lc in algo_lcs]).value_counts().sort_index()
        #  for a_id, algo_lcs in self.lc_algos.items()}))
        #
        # print('\nper dataset distribution of timestamps')
        # print(pd.DataFrame({a_id: pd.Series([len(lc.timestamps) for lc in algo_lcs]).value_counts().sort_index()
        #        for a_id, algo_lcs in self.lc_datasets.items()}))
        #
        # raise ValueError('debugging EDA')

        # add identifier to lc object
        for k, lc in self.lc.items():
            lc.id = k

    def _aggregate_scalars_to_df(self, scalar: str):
        """
        read out the attribute from all the learning curves and create a dataframe
        from it.
        :param scalar:str. must be existing attribute to the learning curves.
        :return: pd.DataFrame (datasets, algorithms)
        """
        # create a dataframe out of it
        dataset_dict = {}
        for d, algos in self.lc_datasets.items():
            dataset_dict[d] = [lc.__dict__[scalar] for lc in algos]

        return pd.DataFrame.from_dict(dataset_dict, orient='index')

    def _calc_timestamp_distribution(self, stamp: int):

        for lc in self.lc.values():
            # to avoid errors due to single timestmaps
            if len(lc.timestamps) > stamp:
                lc.tmp = lc.timestamps[stamp]
            else:
                lc.tmp = lc.timestamps[-1]
        df = self._aggregate_scalars_to_df('tmp')

        # clean up
        for lc in self.lc.values():
            delattr(lc, 'tmp')

        return df

    # DEPRECIATED ---------------------------------------------------------------
    # instead use dataset_g2.py's methods
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

        # fixme: remove this duplicate line: (debugging purpose only)
        self.preprocess_learning_curves(learning_curves)

        # changing keys to int
        # algorithms_meta_features = {k: v for k, v in algorithms_meta_features.items()}
        # dataset_meta_features = {k: v for k, v in dataset_meta_features.items()}
        learning_curves = {k: {k1: v1 for k1, v1 in v.items()}
                           for k, v in learning_curves.items()}
        # Depreciate
        self._preprocess_meta_features(dataset_meta_features)
        self._preprocess_learning_curves(algorithms_meta_features, learning_curves)
        self._preporcess_scalar_properties(self.algo_learning_curves)
        self._preprocess_dataset_properties(learning_curves, dataset_meta_features)
        self._preprocess_thresholded_algo_performances(k=10)

        # Consider: this is the new version of preprocessing
        Dataset_Gravity.nA = len(self.algo_final_performances.columns)
        self.preprocess_learning_curves(learning_curves)
        self._calc_timestamp_distribution(1)

        # needed for plotting
        self.raw_learning_curves = learning_curves
        self.raw_dataset_meta_features = dataset_meta_features

    def preprocess_with_known_deselection(self,
                                          dataset_meta_features, learning_curves,
                                          algorithms_meta_features):
        """
        wrapper around preprocess to allow user to algin validation and test datasets in terms of algorithm
        deselection. (previous call to _reduce algo space resulted in deselection of algos)
        """
        # deselect the algorithms that had been deselected
        learning_curves = {d: {a: curve for a, curve in algos.items()
                               if a not in Dataset_Gravity.deselected}
                           for d, algos in learning_curves.items()}
        algorithms_meta_features = {k: v for k, v in algorithms_meta_features.items()
                                    if k not in Dataset_Gravity.deselected}

        self.preprocess(dataset_meta_features, learning_curves, algorithms_meta_features)

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
                # guarding against 0 devision error
                curve.final_performance = curve.scores[-1] if curve.scores[-1] > 0. else 0.0001

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

        df = self.algo_final_performances
        k_inv = len(df.columns) - k
        
        # init for selection
        deselected = set()
        remaining = set(df.columns)
        # FIXME: argsort is behaving erradically
        performance_baseline = df[np.argsort(df, axis=1) >= k].mean(axis=1)
        performances = {k: None for k in df.loc[:, remaining]}
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
                consider = k_inv - len(deselected)  # rank threshold (minimum rank to get into topk) - this changes
                # once we start removing algorithms

                # tie breaking by order in array
                ranks = pd.DataFrame(ss.rankdata(current_df, axis=1, method='ordinal'))
                ranks.columns, ranks.index = current_df.columns, current_df.index

                avg_remaining_topk_perf = current_df[ranks >= consider].mean(axis=1)
                performances[algo] = performance_baseline - avg_remaining_topk_perf

            if mode == 'skew':
                # remove the algo, that reduces the skewness of the baseline the least
                # i.e. it has the least value to it
                assert all(pd.DataFrame(performances) < 0)
                skew = pd.DataFrame(performances).skew(axis=0)
                deselected_algo = skew.index[np.argmin(skew_baseline - skew)]
                skew_baseline = skew[deselected_algo]

            deselected.add(deselected_algo)
            remaining.remove(deselected_algo)
            performance_baseline = performances.pop(deselected_algo)

        Dataset_Gravity.deselected = deselected

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


class SUR(nn.Module):
    # TODO regularization on the coefficients?
    def __init__(self, epsilon, X_dim, y_dim, lam=10):
        """

        :param epsilon:
        :param X_dim:
        :param y_dim:
        :param lam: L2 regularization weight
        """
        super().__init__()
        self.epsilon = epsilon
        self.X_dim = X_dim
        self.y_dim = y_dim
        self.lam = lam

        self.loss = lambda X: sum(X ** 2)
        self.coef = nn.Parameter(torch.Tensor(X_dim * y_dim, X_dim * y_dim))

    def fit(self, x, y, lr=0.01, budget=10_000):
        """
        Fit a regression model to the data

        :param x: meta features
        :param y: labels
        :param lr: learning rate
        :param budget: number of iterations for the fit

        """
        self.n, self.n_algos = y.shape  # n being a single regressions' no. of obs.

        # Kroneckerize the inputs and format the output as a block diagonal matrix
        X = torch.tensor(np.kron(np.eye(self.n_algos), x), dtype=torch.float32)
        Y = torch.block_diag(*[y[:, i].view(-1, 1) for i in range(self.n_algos)])

        # (1) initialize with the independent coefficients
        self.cov = torch.eye(self.n)
        np_W = np.kron(torch.linalg.inv(self.cov).detach().numpy(),
                       np.eye(self.n_algos))
        self.W = torch.tensor(np_W, dtype=torch.float32)
        self.coef.data = self.gls_beta(X, Y)

        self.coef_lagged = torch.ones_like(self.coef)
        epoch = 0

        diffs = []
        while not self.converged and epoch < budget:
            # for i in tqdm(range(budget)):
            # todo update coef_lagged
            self.coef_lagged = self.coef.clone()
            self.update_cov(X, Y)
            diff_vec = self.gls_beta(X, Y) - self.coef

            diffs.append(torch.norm(diff_vec))

            self.coef.data += lr * (diff_vec)

            epoch += 1
            #print(epoch)

        else:
            if epoch == budget:
                print('convergence was not reached, but budget depleted')

    @property
    def converged(self):
        return torch.allclose(self.coef_lagged, self.coef, rtol=self.epsilon)

    def gls_beta(self, X, Y):
        """
        L2 regularized Generalized Linear Coefficent.
        """
    
        K = self.lam * torch.eye(X.shape[1])
        return torch.linalg.inv(X.t() @ self.W @ X + K) @ X.t() @ self.W @ Y

    def update_cov(self, X, Y):
        """
        Update the cov ariance matrix
        """
        resid = self.residuals(X, Y)
        cov = resid.t() @ resid / self.n
        cov = torch.linalg.inv(cov)
        self.W = torch.tensor(
                            np.kron(
                            cov.detach().numpy(), 
                            np.eye(self.n)
                        ), 
                        dtype=torch.float32
                    )

    def predict(self, X):
        """
        Predict the values of the regressions
        """
        return X @ self.coef

    def residuals(self, X, Y):
        """
        Calculate the residuals
        """
        return Y - X @ self.coef

    def rank(self, X):
        """
        Get hte rank from a single observation
        """
        # given some new dataset meta features rank the algorithms:
        X = torch.tensor(np.kron(np.eye(self.n_algos), X), dtype=torch.float32)
        Y = torch.diag(self.predict(X))

        # based on the predicted values use
        _, rank = torch.unique(Y, sorted=True, return_inverse=True)
        inverted = torch.abs(rank - torch.max(rank))
        return [x for _, x in sorted(zip(inverted, range(20)))]


class Agent:
    nA = None  # number of algorihtms used

    def __init__(
            self,
            number_of_algorithms,
            seed=123546,
            suggest_topk=10
    ):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithm
        
        seed : int
            The seed for the random number generator
        """

        Agent.nA = number_of_algorithms
        self.seed = seed

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

        # fixme: reactivate with encoder!
        # dataset_meta_feature_tensor_testing = dataset_meta_feature_tensor_testing.to(self.model.device)

        int(dataset_meta_features['time_budget'])

        # actual resetting
        self.visits = {k: 0 for k in algorithms_meta_features.keys()}
        self.times = {k: 0. for k in algorithms_meta_features.keys()}
        self.obs_performances = {k: 0. for k in algorithms_meta_features.keys()}

        self.budgets = self.predict_initial_speed(dataset_meta_features_df_testing)

        print('Test_data: predicting ranking based on SUR')

        self.learned_rankings = self.model.rank(dataset_meta_feature_tensor_testing)

    def meta_train(self,
                   dataset_meta_features,
                   algorithms_meta_features,
                   validation_learning_curves,
                   test_learning_curves,
                   # set up the encoder architecture
                   batch_size=9,
                   n_compettitors=11 ):
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

        validation_data = dataset_meta_features, validation_learning_curves, algorithms_meta_features
        test_data = dataset_meta_features, test_learning_curves, algorithms_meta_features

        # validation dataloader
        self.valid_dataset = Dataset_Gravity(n_compettitors)
        self.test_dataset = Dataset_Gravity(n_compettitors)

        if len(Dataset_Gravity.deselected) == 0:
            self.valid_dataset.preprocess(*validation_data)
            self.test_dataset.preprocess(*test_data)
        else:
            print(f'The algorithms {Dataset_Gravity.deselected} have been deselected')
            self.valid_dataset.preprocess_with_known_deselection(*validation_data)
            self.test_dataset.preprocess_with_known_deselection(*test_data)
            self.test_dataset.preprocess_with_known_deselection(*test_data)

        Y = pd.DataFrame.from_dict({i: [lc.final_performance for lc in lc_list] for i, lc_list in
                                    self.valid_dataset.lc_datasets.items()}, orient='index')
        Y = torch.tensor(Y.__array__(), dtype=torch.float32)
        X = self.valid_dataset.datasets_meta_features

        print('Training SUR model for ranking.')
        self.model = SUR(epsilon=0.001, X_dim=27, y_dim=20)
        self.model.fit(X, Y, lr=0.005)  # currently learning rate is not working

        
        observed_rankings = {k: [int(tup[0]) for tup in
                                 self.valid_dataset.dataset_learning_properties[str(k)].ranking]
                             for k in self.valid_dataset.ids_datasets}


        x = torch.tensor(np.kron(np.eye(self.model.n_algos), X),
                         dtype=torch.float32)
        
        order = self.valid_dataset.datasets_meta_features_df.index

        obs_mat = np.array([
            observed_rankings[k] for k in order
        ])

        first_axis = np.shape(obs_mat)[0]
        temp = np.asarray(self.model.predict(x).detach(), dtype=np.float32)
        M = temp.T

        relevant = np.array([M[i, first_axis * i:first_axis * (i + 1)] for i in range(20)]).reshape(
            first_axis, -1)

        mat = [list(sorted([(i, score) for i, score in enumerate(row)], key=lambda x: -x[1]))
               for row in relevant]
        true_ranks = [[tup[0] for tup in row] for row in mat]

        print('\ntimestamp distribution')
        print(
            pd.Series(len(lc.timestamps) for k, lc in self.valid_dataset.lc.items()).value_counts())

        print('\nsingle timestamp scores')
        print(pd.Series(
            lc.scores for k, lc in self.valid_dataset.lc.items() if len(lc.timestamps) == 1))

        self.nA = self.valid_dataset.nA

        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            shuffle=True,
            batch_size=batch_size)

        self.test_dataloader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size)

        # meta_learn convergence speed
        print('Training 90% convergence speed.')
        self.meta_train_convergence_speed(confidence=0.2)

        print('Training initial budget based on timestamps.')
        self.meta_train_initial_budgets(confidence=0.8, stamp=1)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

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
        X = self.valid_dataset.datasets_meta_features_df
        Y = self.valid_dataset.algo_convergences90_time

        # independent (algo-wise) quantile regression models
        self.qr_models = {k: None for k in Y.columns}
        for algo in Y.columns:
            self.qr_models[algo] = QuantileRegressor(loss='quantile', alpha=confidence)
            self.qr_models[algo].fit(X, Y[algo])

    def meta_train_initial_budgets(self, confidence=0.8, stamp=1):
        """
        Since the enviroment penalizes allocating to little budget for the first step,
        we calculate the second timestamp's distribution to ensure we have hit the first one
        """

        X = self.valid_dataset.datasets_meta_features_df
        Y = self.valid_dataset._calc_timestamp_distribution(stamp)

        self.qr_models_init = {k: None for k in Y.columns}
        for algo in Y.columns:
            self.qr_models_init[algo] = QuantileRegressor(loss='quantile', alpha=confidence)
            self.qr_models_init[algo].fit(X, Y[algo])

    def predict_convergence_speed(self, df):
        """Predict the 90% convergence time budget we would like to allocate for each algorithm
        requires meta_train_convergence_speed"""
        if not hasattr(self, 'qr_models'):
            raise ValueError('meta_train_convergence_speed must be executed beforehand')

        prediction_convergence_speed = {}
        for algo in range(self.nA):
            prediction_convergence_speed[int(algo)] = self.qr_models[str(algo)].predict(df)[0]

        return prediction_convergence_speed

    def predict_initial_speed(self, df):
        """Predict the 90% convergence time budget we would like to allocate for each algorithm
        requires meta_train_convergence_speed"""
        if not hasattr(self, 'qr_models'):
            raise ValueError('meta_train_convergence_speed must be executed beforehand')

        prediction_convergence_speed = {}
        for algo in range(self.nA):
            prediction_convergence_speed[int(algo)] = self.qr_models_init[algo].predict(df)

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
            self.zero_flag = True
            self.A = self.learned_rankings[self.counter]
            delta_t = self.budgets[self.A][0]*0.5
            action = (self.A_star, self.A, delta_t)
            self.counter += 1
        else:
            x = self.learned_rankings[self.counter]
            action = (self.A_star, x, 0.0)
            self.zero_flag = False        

        if self.counter == self.suggest_topk:
            self.counter = 0

        return action
