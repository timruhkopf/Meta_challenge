from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as QuantileRegressor
import torch
import matplotlib.pyplot as plt
import pandas as pd
import torch
# from networks# import *
#from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as QuantileRegressor
from torch.utils.data import DataLoader

# FIXME: remove this block for agent_submission
from gravitas.autoencoder import AE
from gravitas.dataset_gravitas import Dataset_Gravity
from gravitas.utils import check_diversity  #
from gravitas.vae import VAE


class Agent:
    encoder_class = {'AE': AE, 'VAE': VAE}

    def __init__(
            self,
            number_of_algorithms,
            encoder: str = "AE",
            seed=123546,
            root_dir='',
            suggest_topk=2
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

    def meta_train(self,
                   dataset_meta_features,
                   algorithms_meta_features,
                   validation_learning_curves,
                   test_learning_curves,
                   # set up the encoder architecture
                   epochs=10,
                   pretrain_epochs=50,
                   batch_size=9,
                   n_compettitors=11,
                   lr=0.001,
                   embedding_dim=2,
                   weights=[1., 1., 1., 1.],
                   repellent_share=0.33,
                   deselect=5, topk=10, deselection_metric='skew',
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

        # # fixme: move following eda to Dataset_Gravity:
        # # (0) find out if there are algorithms that perform bad across all tasks
        # self.valid_dataset.algo_final_performances
        #
        # # (1) find if out if there is a good backup algorithm
        # import numpy as np
        # import seaborn as sns
        # import pandas as pd
        # algo_order = [9, 2, 3, 13, 15, 4, 17, 1, 12, 7, 19, 14, 6, 5, 16, 18, 11, 8, 0, 10]
        # for k in range(10):
        #     self.valid_dataset._preprocess_thresholded_algo_performances(k=k)
        #     (self.valid_dataset.algo_thresholded_rankings != 0).sum(axis=0)
        #
        # # ranking = np.argsort(self.valid_dataset.algo_final_performances, axis=1)
        # # ranking[9].value_counts().sort_index().plot.bar() # plot a single algorithm's performances
        #
        # # algorithm performances across datasets
        # for algo in self.valid_dataset.algo_90_performances.columns:
        #     sns.kdeplot(self.valid_dataset.algo_90_performances[algo], cut=0, label=str(algo))
        #
        # plt.xlabel('relative performance')
        # plt.title('Density of algorithms\'s 90% performance across datasets')
        # plt.legend()
        # plt.show()
        #
        # # (2) Dataset meta_feature projection
        # import umap
        # trans = umap.UMAP(densmap=True, n_neighbors=5, random_state=42).fit(self.valid_dataset.datasets_meta_features_df)
        # plt.scatter(trans.embedding_[:, 0], trans.embedding_[:, 1], s=5,
        #             c=self.valid_dataset.algo_final_performances[9] ,cmap='Spectral')
        # plt.title('Embedding of the training set by densMAP')
        #
        # plt.show()
        #
        # import umap.plot
        # ranking = np.argsort(self.valid_dataset.algo_final_performances, axis=1)
        # umap.plot.points(trans, labels=ranking[9], width=500, height=500)
        # FIXME: start here to find the complementary set.
        #  1) look at the count of top-k k = 3 and see which are the most promising across all
        #  2) see how much performance we'd loose if we removed the across all least performing.
        #     what is the measure here though

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

        # TODO: disable: some data exploration
        # self.test_dataset.plot_convergence90_time()
        #
        # for d in dataset_meta_features.keys():
        #     self.test_dataset.plot_learning_curves(dataset_id=int(d))

        # meta_learn convergence speed
        self.meta_train_convergence_speed(confidence=0.9)

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
            prediction_convergence_speed[int(algo)] = self.qr_models[str(algo)].predict(df)[0]

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
        # TODO predict which algorithms are likely too succeed: (ONCE)  <-- maybe in self.reset?

        # FIXME: Query: C_A: we cannot only log the delta_t, because the actual time
        #  spent wont be incremented by delta_t - but only by the timestamp.
        #  in fact we need to track C_A to know how much budget should
        # keep track of spent budget & observed performances
        if observation is not None:  # initial observation is None
            A, C_A, R = observation
            self.times[str(A)] += C_A
            self.obs_performances[str(A)] = R

        trials = sum(1 if t != 0 else 0 for t in self.times.values())
        A = self.learned_rankings[trials % self.suggest_topk]
        A_star = A

        delta_t = self.budgets[A]

        # Fixme: Negative values of delta_t encountered
        # in some cases, need to be fixed
        if delta_t < 0:
            delta_t = 10

        action = (A, A, delta_t)

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

    def measure_embedding_diversity(self):
        """
        Calculate the diversity based on euclidiean minimal spanning tree
        :return:  diversity for datasets, diversity for algos
        """

        D_test = self.valid_dataloader.dataset.datasets_meta_features.data.to(self.model.device)
        d_test = self.model.encode(D_test)
        z_algo = self.model.Z_algo

        d_test_tree = self._calc_min_eucl_spanning_tree(d_test)
        z_algo_tree = self._calc_min_eucl_spanning_tree(z_algo)

        d_diversity = sum([tup[2]['weight'] for tup in d_test_tree])
        z_diversity = sum([tup[2]['weight'] for tup in z_algo_tree])

        # sum of weighted edges
        return d_diversity, z_diversity

    def _calc_min_eucl_spanning_tree(self, d_test: torch.tensor):
        dist_mat = torch.cdist(d_test, d_test)
        dist_mat = dist_mat.cpu().detach().numpy()

        nodes = list(range(len(dist_mat)))
        d = [(src, dst, dist_mat[src, dst]) for src in nodes for dst in nodes if src != dst]

        df = pd.DataFrame(data=d, columns=['src', 'dst', 'eucl'])

        g = Graph()
        for index, row in df.iterrows():
            g.add_edge(row['src'], row['dst'], weight=row['eucl'])

        return list(minimum_spanning_edges(g))  # fixme: VAE produces NAN sometimes
