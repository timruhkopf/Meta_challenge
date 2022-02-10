from sklearn.linear_model import QuantileRegressor
import torch
from torch.utils.data import DataLoader

from gravitas.autoencoder import AE
from gravitas.base_encoder import BaseEncoder
from gravitas.vae import VAE 
from gravitas.dataset_gravitas import Dataset_Gravity

import numpy as np
import pdb


class Agent_Gravitas:
    def __init__(
            self, 
            number_of_algorithms,
            encoder: str = "VAE", 
            seed=123546):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.nA = number_of_algorithms
        self.times = [0.] * self.nA
        self.encoder = encoder
        self.seed = seed

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
                                    topk=20
                                )[0].tolist()

    def meta_train(self,
                   dataset_meta_features,
                   algorithms_meta_features,
                   validation_learning_curves,
                   test_learning_curves,
                   epochs=100,
                   pretrain_epochs=100,
                   batch_size=9):
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

        # validation dataloader
        self.valid_dataset = Dataset_Gravity(
            dataset_meta_features,
            validation_learning_curves,
            algorithms_meta_features)
        self.valid_dataloader = DataLoader(
            self.valid_dataset,
            shuffle=True,
            batch_size=batch_size
            )

        self.test_dataset = Dataset_Gravity(
            dataset_meta_features,
            test_learning_curves,
            algorithms_meta_features
        )

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
        self.model = eval(self.encoder)(
            input_dim=10,
            latent_dim=2,
            hidden_dims=[8, 4, 3],
            n_algos=20,
            device=device
        )

        print(f'\nPretraining {str(self.encoder)} with reconstruction loss: ')
        tracking_pre, losses_pre, test_losses_pre = self.model.pretrain(
            self.valid_dataloader, self.test_dataloader, epochs=pretrain_epochs)

        print(f'\nTraining {str(self.encoder)} with gravity loss:')
        tracking, losses, test_losses = self.model.trainer(
            self.valid_dataloader, self.test_dataloader, epochs=epochs)

        self.plot_encoder_training(losses, losses_pre)

        # TODO: Add Bandit exploration

    def plot_encoder_training(self, losses, losses_pre):
        # fixme: remove the below plotting method
        import matplotlib.pyplot as plt

        # plot pretrain loss at each epoch.
        plt.figure()
        plt.plot(torch.tensor(losses).numpy(), label="gravity")
        plt.plot(torch.tensor(losses_pre).numpy(), label="pre")
        plt.legend()
        plt.savefig(
            f'output/{self.encoder}_training_loss.png',
        )
        # len(tracking_pre)
        # len(losses_pre)

        # plot the dataset-algo embeddings 2D
        D_test = self.valid_dataloader.dataset.datasets_meta_features.data.to(self.model.device)
        d_test = self.model.encode(D_test)
        d_test = d_test.cpu().detach().numpy()

        z_algo = self.model.Z_algo.cpu().detach().numpy()
        d_test = (d_test - d_test.mean(axis=0)) / d_test.std(axis=0)
        # z_algo = (z_algo - z_algo.mean(axis=0)) / z_algo.std(axis=0)
        z_algo = (z_algo - d_test.mean(axis=0)) / d_test.std(axis=0) # normalization based on datasets

        plt.figure()
        plt.scatter(d_test[:, 0], d_test[:, 1], label="datasets")
        plt.scatter(z_algo[:, 0], z_algo[:, 1], label="algorithms")
        plt.legend()
        plt.savefig(
            f'output/{self.encoder}_training_reps.png',
        )

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
            self.qr_models[algo] = QuantileRegressor(quantile=confidence, alpha=0)
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
        # TODO predict which algorithms are likely too succeed: (ONCE)  <-- maybe in self.reset?

        # keep track of spent budget & observed performances
        if observation is not None: # initial observation is None
            A, C_A, R = observation
            self.times[str(A)] += C_A
            self.obs_performances[str(A)] = R

        trials = sum(1 if t!= 0 else 0 for t in self.times.values())
        
        print(f'Trial Type: {trials}')
        print(f'Learned Rankings: {np.shape(self.learned_rankings)}')
        pdb.set_trace()
        
        A = self.learned_rankings[trials]
        A_star  = A # FIXME: what is the difference?
        delta_t = self.budgets[trials]

        # TODO suggest based on bandit policy
        return A_star, A, delta_t

