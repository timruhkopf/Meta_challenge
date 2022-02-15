import random
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


# TODO Reproducability: Setting seeds!


def ecdf(data):
    """ Compute the empirical distribution function """
    x = np.sort(data)
    n = x.size
    y = np.arange(1, n + 1) / n
    return x, y


class Agent:
    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        ### TO BE IMPLEMENTED ###
        self.nA = number_of_algorithms

    def reset(self, dataset_meta_features, algorithms_meta_features):
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

        self.dataset_meta_features = dataset_meta_features
        self.algorithms_meta_features = algorithms_meta_features

        # keep track of the enviroment values
        self.algo_time_spent = {k: 0.0 for k in algorithms_meta_features.keys()}
        self.algo_performance = {k: 0.0 for k in algorithms_meta_features.keys()}
        self.trajectory = []

    def meta_train(
        self,
        dataset_meta_features,
        algorithms_meta_features,
        validation_learning_curves,
        test_learning_curves,
    ):
        """
        Start meta-training the agent with the validation and test learning curves

        Parameters
        ----------
        datasets_meta_features : dict of dict of {str: str}
            Meta-features of meta-training datasets

        algorithms_meta_features : dict of dict of {str: str}
            The meta_features of all algorithms

        validation_learning_curves : dict of dict of {int : Learning_Curve}
            VALIDATION learning curves of meta-training datasets

        test_learning_curves : dict of dict of {int : Learning_Curve}
            TEST learning curves of meta-training datasets

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

        # obtain self.df_data_meta_features & algo_valid_learning_curves
        # TODO add preprocessing
        # self.preprocess(dataset_meta_features, validation_learning_curves, algorithms_meta_features)

        # ToDo Learn Delta_t policy -------------------------------------------------
        # algorithm characteristics for delta_t estimation
        # these need to be conditioned on the dataset clusters to estimate how much budget we
        # should invest to get an informative point
        algo_id = "9"
        # [curve.convergence90_step for curve in self.algo_valid_learning_curves[algo_id].values()]
        # unconditional distribution (general algo speed):
        unconditional_dist_temp = [
            curve.convergence90_time
            for curve in self.algo_valid_learning_curves[algo_id].values()
        ]
        # plt.hist(unconditional_dist_temp)
        # plt.show()

        x, y = ecdf(unconditional_dist_temp[20:])
        plt.scatter(x=x, y=y)
        plt.xlabel("x", fontsize=16)
        plt.ylabel("y", fontsize=16)
        plt.show()

        k = "9"  # fixme actual suggestion & move to suggestion
        lower_bound = y[np.argmin(x[::-1] >= self.algo_time_spent[k])]

        # inverse sampling with preference for smaller values?
        # --> no inquiry cost for asking suggestion. (unrealistic because of the validation time)
        # Maybe some skewed beta distribution?

        # Transform Meta to learning dataset -----------------------------------
        # Find best performing algorithms for each dataset
        # TODO either remove or use for simple supervised model.
        d_finalist = {}
        for d, algolist in validation_learning_curves.items():
            incumbent_algo = None
            incumbent_score = 0.0
            for a, curve in algolist.items():
                score = curve.scores[-1]
                if score >= incumbent_score:
                    incumbent_algo, incumbent_score = a, score

            d_finalist[d] = (incumbent_algo, incumbent_score)

        # find validation set final performances on respective dataset for each algo
        d_final_performances = {d: [] for d in dataset_meta_features.keys()}
        for d, algolist in validation_learning_curves.items():
            for a, curve in algolist.items():
                d_final_performances[d].append((a, curve.scores[-1]))

            d_final_performances[d] = sorted(
                d_final_performances[d], key=lambda x: -x[1]
            )

        # ToDo Find set of best performing algorithms for each dataset using multiple comparisons
        # Friedman test to check if there is a significant difference between candidate
        # algorithms. no diff --> all are candidates. Otherwise, Holm's procedure
        # to identify the appropriate from the rest.

        # ToDo Find k-best ranking algorithms for each dataset
        # consider: rather than k, find those that are top performing, but are insignificantly different?
        k = 3
        k_highest_performing_on_d = {
            d: ranklist[:k] for d, ranklist in d_final_performances.items()
        }

        # exploratory: investingate the similarity in terms of inductive bias
        peak = pd.DataFrame(
            np.zeros((self.nA, self.nD)),
            columns=sorted(dataset_meta_features.keys(), key=lambda x: int(x)),
            index=sorted(algorithms_meta_features.keys(), key=lambda x: int(x)),
        )
        peak_d = pd.DataFrame(
            np.zeros((self.nA, self.nD)),
            columns=sorted(dataset_meta_features.keys(), key=lambda x: int(x)),
            index=sorted(algorithms_meta_features.keys(), key=lambda x: int(x)),
        )
        for d, ranks in k_highest_performing_on_d.items():
            for tup in ranks:
                peak[d][tup[0]] += 1
                peak_d[d][tup[0]] += tup[1]

        # todo require normalization
        # The diagonal represents the amount of times the algo was in the top-k performing,
        # indicating how much of a "hammer" the algorithm is
        # off_diagonal elements show how often
        algo_corr = peak.__array__() @ peak.__array__().transpose()

        # clustering the datasets based on top-k algorithm performances
        from sklearn.cluster import spectral_clustering

        dataset_corr = peak_d.__array__().transpose() @ peak_d.__array__()
        label = spectral_clustering(dataset_corr, n_clusters=4, eigen_solver="arpack")

        # todo clustering to find similarities between datasets / algorithms respectively?

        # Meta-inference -------------------------------------------------------
        # ToDo Use single best algorithm

        # ToDo Use Ranking based algorithms

        # ToDo Use Multi-label based algorithms

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
        # save trajectory
        self.trajectory.append(observation)

        # TODO Predict to which class of dataset the new one belongs

        # ToDo Predict its best ranking algorithm(s) & consider in the set of
        #  candidates which might perform well but are cheap (converge presumably faster)
        #  --> LANDMARKING. should they succeed, we can query more expensive ones?
        #  --> given that our cheap algo performed well, we can select a more expensive one
        #      if it performed bad however, we might want to consider another

        # ToDo Decide on delta_t.
        # exploiting the discrete nature of the curve; we want to inquire at least 90% of final performance
        # so we need to formulate an expectation towards when this amount will be reached and check
        # if a) we might have achieved it already and b) if we haven't seen new info - we should allocate
        # more budget until we see a new observation which is in the vacinity of the allegedly
        # expected performance.

        # TODO currently max available budget:
        # remaining time = self.dataset_meta_features['total_budget'] - consumed_time

        # ToDo save the observation to history
        A_star, A, delta_t = None, None, None
        self.algo_time_spent_algo[A_star] += delta_t

        return (A_star, A, delta_t)

    def _exploratory_meta_feature_analysis(
        self,
        dataset_meta_features,
        algorithms_meta_features,
        validation_learning_curves,
        test_learning_curves,
    ):
        # FIXME: remove me: exploratory analysis of dataset meta features

        # (0) dataset_meta_features ------------------------------------------------
        # Notice, that index AND name are unique identifiers. Name is dataset name
        # String_typed_variables are useless variables.
        import seaborn as sns
        import matplotlib.pyplot as plt
        import scipy

        df_meta_features = pd.DataFrame(
            list(dataset_meta_features.values()), index=dataset_meta_features.keys()
        )
        string_typed_variables = [
            "usage",
            "name",
            "task",
            "target_type",
            "feat_type",
            "metric",
        ]
        df_meta_features[string_typed_variables].describe()

        other_columns = list(
            set(df_meta_features.columns) - set(string_typed_variables)
        )
        df_meta_features[other_columns] = df_meta_features[other_columns].astype(float)
        df_meta_features[other_columns].describe()

        sns.pairplot(data=df_meta_features, vars=other_columns)
        plt.show()

        # (1) algorithm_meta_features ----------------------------------------------
        df_algo_meta_features = pd.DataFrame(
            list(algorithms_meta_features.values()),
            index=algorithms_meta_features.keys(),
        )
        df_algo_meta_features["meta_feature_1"] = df_algo_meta_features[
            "meta_feature_1"
        ].astype(float)
        # exactly the same overall distribution if grouped by the binary meta feature
        df_algo_meta_features.groupby("meta_feature_0").describe()

        # (2) validation / test_learning_curves
        def plot_curves(dataset):
            for ds_id in dataset.keys():
                for algo_id, curve in dataset[ds_id].items():
                    plt.plot(
                        curve.timestamps,
                        curve.scores,
                        linestyle="--",
                        marker="o",
                        label=algo_id,
                    )

                plt.title("Dataset {}".format(ds_id))
                plt.legend(loc="upper right")
                plt.show()

        plot_curves(validation_learning_curves)
        plot_curves(test_learning_curves)

        # (3) Informational content of discretized LC
        # The 2nd / 3rd observation usually suffices to get a good performance indication.
        # the question is, at which timestamp do they occur? and how informative are they really?
        n_stamps = 5
        curve_set = test_learning_curves  # validation_learning_curves
        stamp = {k: [] for k in range(n_stamps)}
        share_in_performance = {k: [] for k in range(n_stamps)}

        for ds_id in curve_set.keys():
            for algo_id, curve in curve_set[ds_id].items():

                final_performance = curve.scores[-1]

                for k in range(n_stamps):
                    stamp[k].append(curve.timestamps[k])
                    share_in_performance[k].append(curve.scores[k] / final_performance)

        [sns.kdeplot(stamp[k]) for k in range(n_stamps)]
        plt.legend()
        plt.show()

        # peak hight indicates how much information is contained in the respective
        # stamp on average. Apparently it should generally suffice to get info from
        # the 2nd or 3rd.
        [sns.kdeplot(share_in_performance[k], label=k) for k in range(n_stamps)]
        plt.legend()
        plt.show()

        # TODO dive deeper into these curves: for each step: how do the curves look
        #  like for the respective algorithm? & dataset? This should be some kind
        #  of a complexity measure, that could be used to predict the informational
        #  content of the curve.

        # sns.jointplot(x=df["sepal_length"], y=df["sepal_width"], kind='kde')

        # fit log curve to the data
        # !!! The fit is terrible though !!!
        dataset = validation_learning_curves
        for ds_id in dataset.keys():
            for algo_id, curve in dataset[ds_id].items():
                plt.plot(
                    curve.timestamps,
                    curve.scores,
                    linestyle="--",
                    marker="o",
                    label=algo_id,
                )

                def func(x, a, b, c):
                    return a * np.log(b * x) + c

                x = curve.timestamps
                y = curve.scores

                popt, pcov = scipy.optimize.curve_fit(func, x, y)

                plt.plot(
                    curve.timestamps,
                    func(curve.timestamps, *popt),
                    linestyle="-",
                    marker="x",
                    label=algo_id,
                )

            plt.show()

        # alorithm on the respective datasets
        algo_datasets = {
            a: {d: [] for d in dataset_meta_features.keys()}
            for a in algorithms_meta_features.keys()
        }
        dataset = test_learning_curves
        for ds_id in dataset.keys():
            for algo_id, curve in dataset[ds_id].items():
                algo_datasets[algo_id][ds_id] = curve

        # plot the algo for all datasets
        for algo_id, datasets in algo_datasets.items():
            for ds_id, curve in datasets.items():
                plt.plot(
                    curve.timestamps,
                    curve.scores,
                    linestyle="--",
                    marker="o",
                    label=algo_id,
                )

            plt.title("Algorithm {}".format(ds_id))
            plt.legend(loc="upper right")
            plt.show()

        # algo meta features are unique identifiers: and useless for this challenge
        df_algo_meta_features = pd.DataFrame(
            list(algorithms_meta_features.values()),
            index=algorithms_meta_features.keys(),
        )

        sns.set(style="ticks", context="talk")

        sns.swarmplot(
            x="meta_feature_0", y="meta_feature_1", data=df_algo_meta_features
        )
        sns.despine()
        plt.show()

        sorted(
            [
                (v["meta_feature_0"], v["meta_feature_1"])
                for k, v in algorithms_meta_features.items()
            ]
        )
        # [('0', '0.1'), ('0', '0.2'), ('0', '0.3'), ('0', '0.4'), ('0', '0.5'), ('0', '0.6'), ('0', '0.7'), ('0', '0.8'),
        #  ('0', '0.9'), ('0', '1.0'), ('1', '0.1'), ('1', '0.2'), ('1', '0.3'), ('1', '0.4'), ('1', '0.5'), ('1', '0.6'),
        #  ('1', '0.7'), ('1', '0.8'), ('1', '0.9'), ('1', '1.0')]
