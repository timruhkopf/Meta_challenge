import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset
from itertools import product

class Dataset_Gravity(Dataset):
    def __init__(self, dataset_meta_features, learning_curves, algorithms_meta_features):
        self.preprocess(dataset_meta_features, learning_curves, algorithms_meta_features)

        # create pairwise datasets
        # indicies = dataset_meta_features.keys()
        self.data_indicies = list(tup for tup in product(range(self.nD), range(self.nD)) if tup[0] != tup[1])

    def __len__(self):
        return len(self.datasets_meta_features)

    def __getitem__(self, item):
        # TODO one vs all (subset of other datasets)
        # select comparison partners
        idx = self.data_indicies[item]
        D0, D1 = self.datasets_meta_features[idx[0]], self.datasets_meta_features[idx[1]]



        # learning_properties
        A0, A1 = self.algo_performances[idx[0]], self.algo_performances[idx[1]]

        return D0, D1, A0, A1

    def preprocess(self, dataset_meta_features, validation_learning_curves, algorithms_meta_features):
        """
        (1) Dataset Meta Features are selected based on being variable and transformed to tensor + normalized
        (2) Learning Curves: for each LC particular properties are read out such as e.g. final performance.
        (3) Algorithm Features: LC properties aggregated by algorithm
        (4) Dataset Features LC properties aggregated by dataset

        :param dataset_meta_features:
        :param validation_learning_curves:
        :param algorithms_meta_features:
        :return:
        """
        self.nD = len(dataset_meta_features.keys())

        # Preprocess dataset meta data (remove the indiscriminative string variables)
        self.df_data_meta_features = pd.DataFrame(
            list(dataset_meta_features.values()),
            index=dataset_meta_features.keys())
        string_typed_variables = ['usage', 'name', 'task', 'target_type', 'feat_type', 'metric']
        other_columns = list(set(self.df_data_meta_features.columns) - set(string_typed_variables))
        self.df_data_meta_features[other_columns] = self.df_data_meta_features[other_columns].astype(float)

        # min-max normalization of numeric features
        df = self.df_data_meta_features[other_columns]
        self.df_data_meta_features[other_columns] = (df - df.min()) / (df.max() - df.min())
        self.datasets_meta_features = torch.tensor(self.df_data_meta_features[other_columns].values,
                                                   dtype=torch.float32)

        # compute learning curve properties. -----------------------------------
        # find the 90% quantile in performance
        # ---> learn how fast an algo given a dataset complexity will approx converge?
        curve_set = validation_learning_curves
        self.algo_valid_learning_curves = {k: {} for k in algorithms_meta_features.keys()}
        for ds_id in curve_set.keys():
            for algo_id, curve in curve_set[ds_id].items():
                self.algo_valid_learning_curves[algo_id][ds_id] = curve

                # final performance
                curve.final_performance = curve.scores[-1]

                # how much in % of the final performance
                curve.convergence_share = curve.scores / curve.final_performance

                # stamp at which at least 90% of final performance is reached
                threshold = 0.9
                curve.convergence90_step = np.argmax(curve.convergence_share >= threshold)
                curve.convergence90_time = curve.timestamps[curve.convergence90_step]

        class Datum:
            # placeholder object to keep track of aggregation properties (instead of
            # all of the dictionaries flying around.
            pass

        # compute algorithm properties -----------------------------------------

        # compute dataset properties  ------------------------------------------
        self.dataset_learning_properties = {d: Datum() for d in dataset_meta_features.keys()}
        k = 3
        for d in validation_learning_curves.keys():
            datum = self.dataset_learning_properties[d]
            datum.final_scores = []  # unconditional distribution
            datum.convergence_step = []
            datum.ranking = []
            datum.meta = dataset_meta_features[d]

            for a, curve in validation_learning_curves[d].items():
                datum.convergence_step.append(curve.convergence90_step)
                # datum.convergence_avg_topk =
                datum.final_scores.append(curve.final_performance)

                datum.ranking.append((a, curve.final_performance, curve.convergence90_step, curve.convergence90_time))

            datum.ranking = sorted(datum.ranking, key=lambda x: -x[1])

            datum.convergence_avg_topk = np.mean([algotup[3] for algotup in datum.ranking[:k]])
            datum.convergence_avg = np.mean(datum.convergence_step)

            datum.topk_available_trials = float(datum.meta['time_budget']) / datum.convergence_avg_topk

        # # amount of time on the datasets to find out which algo works best:
        # plt.hist(self.df_data_meta_features['time_budget'])
        # plt.show()
        #
        # plt.hist([datum.topk_available_trials for datum in self.dataset_learning_properties.values()])
        # plt.show()
        order = self.df_data_meta_features.index

        algo_performances = pd.DataFrame(
            [self.dataset_learning_properties[i].final_scores for i in order],
            index=order
        )  # index = dataset_id, column = algo_id

        self.algo_performances = torch.tensor(algo_performances.values)

