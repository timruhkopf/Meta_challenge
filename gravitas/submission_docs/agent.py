# Packages in the requiremnets file
import random
import pandas as pd
import numpy as np
from typing import List, Tuple, Any

import subprocess

# def uninstall(name):
#     subprocess.call(['pip', 'uninstall', '-y', name])

# def install(name):
#     subprocess.call(['pip', 'install', name])

# #uninstall('scikit-learn')
# install('scikit-learn==1.0.2')


# Packasges that need to be additionally installed
from sklearn.ensemble.gradient_boosting import GradientBoostingRegressor as QuantileRegressor
#from sklearn.linear_model import QuantileRegressor
import torch
from torch.utils.data import DataLoader
import torch
from torch.utils.data import Dataset
import torch.nn.functional as F
import torch.nn as nn
import torch.distributions as td

import pdb



#==============AUTOENCODERS==============
class AE(nn.Module):
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
        weights = [weights[0], *weights[2:], weights[1]] # fixme: change the doc instead!
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

        return torch.stack([gravity, self.weights[-1] * algo_pull,])

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
        hidden_dims: List[int] = [8,4],
        embedding_dim: int = 2,
        weights: List[float]=[1.0, 1.0, 1.0, 1.0],
        repellent_share: float =0.33,
        n_algos: int =20,
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

#==============Pre-Processing==============
class Dataset_Gravity(Dataset):
    def __init__(
            self, 
            dataset_meta_features, 
            learning_curves, 
            algorithms_meta_features, 
            no_competitors=11, 
            seed=123456
        ):
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

        # optional thresholding: remove k least performing
        algo_performances = torch.tensor(self.algo_final_performances.values, dtype=torch.float32)
        _, ind = torch.topk(algo_performances, k=k, largest=False, dim=1)
        for i_row, row in zip(ind, algo_performances):
            row[i_row] = 0

        self.algo_thresholded_performances = algo_performances





#==============AGENT Class==============
class Agent:

    def __init__(
            self,
            number_of_algorithms,
            seed=123546,
            encoder: str = "VAE",
            suggest_topk = 2
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
            'VAE' for the variational autoencoder -- To 
        
        seed : int
            The seed for the random number generator
        """

        self.nA = number_of_algorithms
        self.times = [0.] * self.nA
        self.encoder = encoder
        self.seed = seed

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
            Dataset_Gravity._preprocess_dataset_properties_meta_testing(
                dataset_meta_features,
                self.valid_dataset.normalizations
            )

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
                   epochs=1000,
                   pretrain_epochs=500,
                   batch_size=9,
                   n_compettitors=11,
                   lr=0.001,
                   embedding_dim=2,
                   weights=[1., 1., 1., 1.],
                   repellent_share=0.33,
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
            n_compettitors)
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
        self.test_dataloader = DataLoader(
            self.test_dataset,
            shuffle=True,
            batch_size=batch_size)

        # meta_learn convergence speed
        self.meta_train_convergence_speed(confidence=0.9)

        # Training (algo-ranking) procedure
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = eval(self.encoder)(
            input_dim=10,
            embedding_dim=embedding_dim,
            hidden_dims=[8, 4],
            weights=weights,
            repellent_share=repellent_share,
            n_algos=self.nA,
            device=device,

        )

        if training == 'gravity':
            tracking, losses, test_losses = self.model.train_gravity(
                self.valid_dataloader, self.test_dataloader, epochs=[pretrain_epochs, epochs], lr=lr)

        elif training == 'schedule':
            tracking, losses, test_losses = self.model.train_schedule(
                self.valid_dataloader, self.test_dataloader, epochs=[pretrain_epochs, epochs, epochs], lr=lr)


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
            prediction_convergence_speed[algo] = self.qr_models[algo].predict(df)

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

        if observation is not None:  # initial observation is None
            A, C_A, R = observation
            self.times[str(A)] += C_A
            self.obs_performances[str(A)] = R

        trials = sum(1 if t != 0 else 0 for t in self.times.values())
        A = self.learned_rankings[trials%self.suggest_topk]
        A_star = A
        
        delta_t = self.budgets[A][0]

        # Fixme: Negative values of delta_t encountered
        # in some cases, need to be fixed
        if delta_t < 0:
            delta_t = 10

        action = (A, A, delta_t)

        return action
