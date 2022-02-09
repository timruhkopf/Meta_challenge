import os
from sys import argv, path
from ingestion_program.environment import Meta_Learning_Environment
import random
import os
from sklearn.model_selection import KFold
import time
import datetime
import shutil

# === Verbose mode
verbose = True

# === Set RANDOM SEED
random.seed(208)

# === Setup input/output directories
root_dir = "/home/ruhkopf/PycharmProjects/Meta_challenge/"  # os.getcwd()
default_input_dir = os.path.join(root_dir, "sample_data/")
default_output_dir = os.path.join(root_dir, "output/")
default_program_dir = os.path.join(root_dir, "ingestion_program/")
default_submission_dir = os.path.join(root_dir, "sample_code_submission/")


def vprint(mode, t):
    """
    Print to stdout, only if in verbose mode.

    Parameters
    ----------
    mode : bool
        True if the verbose mode is on, False otherwise.

    Examples
    --------
    >>> vprint(True, "hello world")
    hello world

    >>> vprint(False, "hello world")

    """

    if mode:
        print(str(t))


def clear_output_dir(output_dir):
    """
    Delete previous output files.

    Parameters
    ----------
    output_dir : str
        Path to the output directory.
    """
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    else:
        shutil.rmtree(output_dir)
        os.makedirs(output_dir)

    # === Delete all .DS_Store
    os.system("find . -name '.DS_Store' -type f -delete")


def meta_training(agent, D_tr):
    """
    Meta-train an agent on a set of datasets.

    Parameters
    ----------
    agent : Agent
        The agent before meta-training.
    D_tr : list of str
        List of dataset indices used for meta-training

    Returns
    -------
    agent : Agent
        The meta-trained agent.
    """

    vprint(verbose, "[+]Start META-TRAINING phase...")
    vprint(verbose, "meta-training datasets = ")
    vprint(verbose, D_tr)

    # === Gather all meta_features and learning curves on meta-traning datasets
    validation_learning_curves = {}
    test_learning_curves = {}
    datasets_meta_features = {}
    algorithms_meta_features = {}

    for d_tr in D_tr:
        dataset_name = list_datasets[d_tr]
        datasets_meta_features[dataset_name] = env.meta_features[dataset_name]
        validation_learning_curves[dataset_name] = env.validation_learning_curves[
            dataset_name
        ]
        test_learning_curves[dataset_name] = env.test_learning_curves[dataset_name]

    # === Get algorithms_meta_features of algorithms
    algorithms_meta_features = env.algorithms_meta_features

    # === Start meta-traning the agent
    vprint(verbose, datasets_meta_features)
    vprint(verbose, algorithms_meta_features)
    agent.meta_train(
        datasets_meta_features,
        algorithms_meta_features,
        validation_learning_curves,
        test_learning_curves,
    )

    vprint(verbose, "[+]Finished META-TRAINING phase")

    return agent


def meta_testing(trained_agent, D_te):
    """
    Meta-test the trained agent on a set of datasets.

    Parameters
    ----------
    trained_agent : Agent
        The agent after meta-training.
    D_te : list of str
        List of dataset indices used for meta-testing
    """

    vprint(verbose, "\n[+]Start META-TESTING phase...")
    vprint(verbose, "meta-testing datasets = " + str(D_te))

    for d_te in D_te:
        dataset_name = list_datasets[d_te]
        meta_features = env.meta_features[dataset_name]

        # === Reset both the environment and the trained_agent for a new task
        dataset_meta_features, algorithms_meta_features = env.reset(
            dataset_name=dataset_name
        )
        trained_agent.reset(dataset_meta_features, algorithms_meta_features)
        vprint(
            verbose,
            "\n#===================== Start META-TESTING on dataset: "
            + dataset_name
            + " =====================#",
        )
        vprint(verbose, "\n#---Dataset meta-features = " + str(dataset_meta_features))
        vprint(
            verbose, "\n#---Algorithms meta-features = " + str(algorithms_meta_features)
        )

        # === Start meta-testing on a dataset step by step until the given total_time_budget is exhausted (done=True)
        done = False
        observation = None
        while not done:
            # === Get the agent's suggestion
            action = trained_agent.suggest(observation)

            # === Execute the action and observe
            observation, done = env.reveal(action)

            vprint(verbose, "------------------")
            vprint(verbose, "A_star = " + str(action[0]))
            vprint(verbose, "A = " + str(action[1]))
            vprint(verbose, "delta_t = " + str(action[2]))
            vprint(verbose, "remaining_time_budget = " + str(env.remaining_time_budget))
            vprint(verbose, "observation = " + str(observation))
            vprint(verbose, "done=" + str(done))
    vprint(verbose, "[+]Finished META-TESTING phase")


#################################################
################# MAIN FUNCTION #################
#################################################

if __name__ == "__main__":
    # === Get input and output directories
    if (
            len(argv) == 1
    ):  # Use the default input and output directories if no arguments are provided
        input_dir = default_input_dir
        output_dir = default_output_dir
        program_dir = default_program_dir
        submission_dir = default_submission_dir
        validation_data_dir = os.path.join(input_dir, "valid")
        test_data_dir = os.path.join(input_dir, "test")
        meta_features_dir = os.path.join(input_dir, "dataset_meta_features")
        algorithms_meta_features_dir = os.path.join(
            input_dir, "algorithms_meta_features"
        )
    else:
        input_dir = os.path.abspath(argv[1])
        output_dir = os.path.abspath(argv[2])
        program_dir = os.path.abspath(argv[3])
        submission_dir = os.path.abspath(argv[4])
        validation_data_dir = os.path.join(input_dir, "valid")
        test_data_dir = os.path.join(input_dir, "test")
        meta_features_dir = os.path.join(input_dir, "dataset_meta_features")
        algorithms_meta_features_dir = os.path.join(
            input_dir, "algorithms_meta_features"
        )

    vprint(verbose, "Using input_dir: " + input_dir)
    vprint(verbose, "Using output_dir: " + output_dir)
    vprint(verbose, "Using program_dir: " + program_dir)
    vprint(verbose, "Using submission_dir: " + submission_dir)
    vprint(verbose, "Using validation_data_dir: " + validation_data_dir)
    vprint(verbose, "Using test_data_dir: " + test_data_dir)
    vprint(verbose, "Using meta_features_dir: " + meta_features_dir)
    vprint(
        verbose, "Using algorithms_meta_features_dir: " + algorithms_meta_features_dir
    )

    # === List of dataset names
    list_datasets = os.listdir(validation_data_dir)
    if ".DS_Store" in list_datasets:
        list_datasets.remove(".DS_Store")
    list_datasets.sort()

    # === List of algorithms
    list_algorithms = os.listdir(os.path.join(validation_data_dir, list_datasets[0]))
    if ".DS_Store" in list_algorithms:
        list_algorithms.remove(".DS_Store")
    list_algorithms.sort()

    # === Import the agent submitted by the participant ----------------------------------------------------------------
    path.append(submission_dir)
    from gravitas.agent_gravitas import (
        Agent_Gravitas as Agent,
    )  # fixme: for debugging: replace with my own Agent script

    # === Clear old output
    clear_output_dir(output_dir)

    # TODO SMAC HPO the Agent's meta training Autoencoder --------------------------------------------------------------
    #  !! make Meta_training dependent on the hyperparameters !!
    #  set up hyperband with multiple epoch-fidelities
    import logging

    logging.basicConfig(level=logging.INFO)
    import numpy as np
    import torch
    from sklearn.metrics import ndcg_score, label_ranking_loss
    from gravitas.dataset_gravitas import Dataset_Gravity
    import ConfigSpace as CS
    from ConfigSpace.hyperparameters import \
        CategoricalHyperparameter, UniformFloatHyperparameter, UniformIntegerHyperparameter

    from smac.configspace import ConfigurationSpace
    from smac.facade.smac_mf_facade import SMAC4MF
    from smac.scenario.scenario import Scenario

    # (0) ConfigSpace ----------------------------------------------------------
    cs = ConfigurationSpace()
    n_compettitors = UniformIntegerHyperparameter('n_compettitors', 1, 19, default_value=10)
    embedding_dim = UniformIntegerHyperparameter('embedding_dim', 2, 5, default_value=3)
    repellent_threshold = UniformFloatHyperparameter('repellent_threshold', 0., .8, default_value=0.33)
    lossweight1 = UniformFloatHyperparameter('lossweight1', 0., 10., default_value=1.)
    lossweight2 = UniformFloatHyperparameter('lossweight2', 0., 10., default_value=1.)
    lossweight3 = UniformFloatHyperparameter('lossweight3', 0., 10., default_value=1.)
    lossweight4 = UniformFloatHyperparameter('lossweight4', 0., 10., default_value=1.)
    learning_rate_init = UniformFloatHyperparameter(
        'learning_rate_init', 0.0001, 1.0, default_value=0.001, log=True)

    # Add all hyperparameters at once:
    cs.add_hyperparameters(
        [n_compettitors, embedding_dim, repellent_threshold, lossweight1,
         lossweight2, lossweight3, lossweight4, learning_rate_init])


    # (1) TAE ------------------------------------------------------------------
    # TODO make TAE config dependent
    def tae(cfg):
        # === Init K-folds cross-validation
        kf = KFold(n_splits=6, shuffle=False)

        ################## MAIN LOOP ##################
        # === Init a meta-learning environment
        env = Meta_Learning_Environment(
            validation_data_dir,
            test_data_dir,
            meta_features_dir,
            algorithms_meta_features_dir,
            output_dir,
        )

        # === Start iterating, each iteration involves a meta-training step and a meta-testing step
        iteration = 0
        holdout_losses_ndcg = []
        holdout_losses_ranking = []
        for D_tr, D_te in kf.split(list_datasets):
            vprint(verbose, "\n********** ITERATION " + str(iteration) + " **********")

            # Init a new agent instance in each iteration
            agent = Agent(number_of_algorithms=len(list_algorithms))

            # META-TRAINING-----------
            trained_agent = meta_training(agent, D_tr)

            # precompute ground trugh labels:
            # get ground truth label
            meta_test_dataset = [list_datasets[d] for d in D_te]
            meta_features = {k: env.meta_features[k] for k in meta_test_dataset}
            learning_curves = {k: env.test_learning_curves[k] for k in meta_test_dataset}
            meta_test_dataset = Dataset_Gravity(meta_features, learning_curves, env.algorithms_meta_features)

            ground_truth_rankings = np.argsort(meta_test_dataset.algo_final_performances, axis=1)

            # META-TESTING-------------
            # meta_testing(trained_agent, D_te)  # <<<--- intercepting the results on the heldout fold
            holdout_rankings = []
            holdout_rankings_truth = []
            holdout_embedding_distances = []  # distances of algorithms to the dataset. (base for ranking vector)
            for d_te in D_te:
                dataset_name = list_datasets[d_te]
                meta_features = env.meta_features[dataset_name]

                # === Reset both the environment and the trained_agent for a new task
                dataset_meta_features, algorithms_meta_features = env.reset(dataset_name=dataset_name)
                # agent.reset will already trigger a prediction on the new dataset
                trained_agent.reset(dataset_meta_features, algorithms_meta_features)
                holdout_rankings.append(trained_agent.learned_rankings)
                holdout_rankings_truth.append(ground_truth_rankings.loc[int(dataset_name), :])

                # get the embedding distances (score for ndcg loss)
                d_new = meta_test_dataset.datasets_meta_features[d_te].reshape(1, -1)
                d_new = d_new.to(agent.model.device)
                agent.model.eval()
                # fixme: encoding seems not to work
                Z_data = agent.model.encode(d_new)
                dist_mat = torch.cdist(Z_data, agent.model.Z_algo)

            holdout_rankings_truth = np.array(holdout_rankings)

            # compute the fold's loss
            # Fixme: choose a K?? NDCG@K loss!
            holdout_losses_ndcg.append(
                ndcg_score(holdout_rankings_truth, dist_mat.numpy(),
                           k=None, sample_weight=None, ignore_ties=False))

            holdout_losses_ranking.append(
                label_ranking_loss(holdout_rankings_truth, trained_agent.learned_rankings))

            iteration += 1

        # average across holdout_scores
        return 1 - (sum(holdout_losses_ndcg) / (len(D_te) * 6))  # no_splits=6

    # (2) Set up SMAC-MF with budget schedual
    tae(None)
    # (3) run & analyse the performances.