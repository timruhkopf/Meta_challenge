import random

import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset

import matplotlib.pyplot as plt
import seaborn as sns


class Dataset_Gravity(Dataset):
    def __init__(self, dataset_meta_features, learning_curves, algorithms_meta_features, no_competitors=11):
        """

        :param dataset_meta_features:
        :param learning_curves:
        :param algorithms_meta_features:
        :param no_competitors: number of datasets that are compared against. Notice that no_competitors = 2
        is pairwise comparisons
        """
        self.no_competitors = no_competitors
        self.preprocess(dataset_meta_features, learning_curves, algorithms_meta_features)

        # needed for plotting
        self.raw_learning_curves = learning_curves
        self.raw_dataset_meta_features = dataset_meta_features

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
        A0 = self.algo_performances[item]

        # generate a random compare set
        # TODO move k to init
        # TODO seedit
        item_compareset = random.choices(list(set(range(self.nD)) - {item}), k=self.no_competitors)
        D1 = self.datasets_meta_features[item_compareset]
        A1 = self.algo_performances[item_compareset]

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
        algorithms_meta_features = {int(k): v for k, v in algorithms_meta_features.items()}
        dataset_meta_features = {int(k): v for k, v in dataset_meta_features.items()}
        learning_curves = {int(k): {int(k1): v1 for k1, v1 in v.items()}
                           for k, v in learning_curves.items()}

        self._preprocess_meta_features(dataset_meta_features)
        self._preprocess_learning_curves(algorithms_meta_features, learning_curves)
        self._preporcess_scalar_properties(self.algo_learning_curves)
        self._preprocess_dataset_properties(learning_curves, dataset_meta_features)
        self._preprocess_thresholded_algo_performances(k=10)
        # # amount of time on the datasets to find out which algo works best:
        # plt.hist(self.df_data_meta_features['time_budget'])
        # plt.show()
        #
        # plt.hist([datum.topk_available_trials for datum in self.dataset_learning_properties.values()])
        # plt.show()

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
                self.algo_learning_curves[algo_id][ds_id] = curve

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
        datasets_meta_features_df = pd.DataFrame(
            list(dataset_meta_features.values()), index=dataset_meta_features.keys()
        )
        string_typed_variables = [
            "usage", "name", "task", "target_type", "feat_type", "metric",
        ]
        other_columns = list(set(datasets_meta_features_df.columns) - set(string_typed_variables))
        datasets_meta_features_df = datasets_meta_features_df[other_columns].astype(float)

        # min-max normalization of numeric features
        df = datasets_meta_features_df

        self.normalizations = df.min(), df.max()
        datasets_meta_features_df = (df - df.min()) / (df.max() - df.min())

        self.datasets_meta_features_df = datasets_meta_features_df
        self.datasets_meta_features = torch.tensor(
            self.datasets_meta_features_df.values, dtype=torch.float32)

    @staticmethod
    def _preprocess_dataset_properties_meta_testing(dataset_meta_features, normalizations):
        df = pd.Series(dataset_meta_features).to_frame().T
        df = df[['train_num', 'target_num', 'has_categorical', 'valid_num', 'feat_num',
                 'time_budget', 'label_num', 'is_sparse', 'has_missing', 'test_num']]
        df = df.astype(float)
        minimum, maximum = normalizations
        df = (df - minimum) / (maximum - minimum)

        return df, torch.tensor(df.values, dtype=torch.float32)

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

        self.algo_performances = algo_performances

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
