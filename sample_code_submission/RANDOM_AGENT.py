import random
import numpy as np


class Agent():
    """
    RANDOM SEARCH AGENT
    """

    def __init__(self, number_of_algorithms):
        """
        Initialize the agent

        Parameters
        ----------
        number_of_algorithms : int
            The number of algorithms

        """
        self.nA = number_of_algorithms

    def reset(self, dataset_meta_features, algorithms_meta_features):
        """
        Reset the agents' memory for a new dataset

        Parameters
        ----------
        dataset_meta_features : dict of {str : str}
            The meta-features of the dataset at hand, including:
                'usage' : name of the competition
                'name' : name of the dataset
                'task' : type of the task
                'target_type' : target type
                'feat_type' : feature type
                'metric' : evaluatuon metric used
                'time_budget' : time budget for training and testing
                'feat_num' : number of features
                'target_num' : number of targets
                'label_num' : number of labels
                'train_num' : number of training examples
                'valid_num' : number of validation examples
                'test_num' : number of test examples
                'has_categorical' : presence or absence of categorical variables
                'has_missing' : presence or absence of missing values
                'is_sparse' : full matrices or sparse matrices

        algorithms_meta_features : dict of dict of {str : str}
            The meta_features of all algorithms

        Examples
        ----------
        >>> dataset_meta_features
        {'usage': 'Meta-learningchallenge2022', 'name': 'Erik', 'task': 'regression',
        'target_type': 'Binary', 'feat_type': 'Mixed', 'metric': 'f1_metric',
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
        self.validation_last_scores = [0.0 for i in range(self.nA)]

    def meta_train(self, datasets_meta_features, algorithms_meta_features, validation_learning_curves,
                   test_learning_curves):
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

        self.validation_learning_curves = validation_learning_curves
        self.test_learning_curves = test_learning_curves
        self.datasets_meta_features = datasets_meta_features
        self.algorithms_meta_features = algorithms_meta_features

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
        # === Uniformly sampling
        next_algo_to_reveal = random.randint(0, self.nA - 1)
        delta_t = random.randrange(10, 100, 10)

        if observation == None:
            best_algo_for_test = None
        else:
            A, C_A, R_validation_C_A = observation
            self.validation_last_scores[A] = R_validation_C_A
            best_algo_for_test = np.argmax(self.validation_last_scores)

        action = (best_algo_for_test, next_algo_to_reveal, delta_t)
        return action
