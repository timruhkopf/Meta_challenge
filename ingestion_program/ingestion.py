import os
import sys
from sys import argv, path
import random
import os
from sklearn.model_selection import KFold
import shutil
import pdb
import inspect

currentdir = os.path.dirname(os.path.abspath(inspect.getfile(inspect.currentframe())))
parentdir = os.path.dirname(currentdir)
sys.path.insert(0, parentdir)

from ingestion_program import Meta_Learning_Environment

# === Verbose mode
verbose = True

# === seeding
seed = 123456

# === Set RANDOM SEED : TODO see how this translates to the whole pipeline
random.seed(seed)

# === Setup input/output directories
root_dir = '/'.join(os.getcwd().split('/')[:-1])  # fixing the root to project root and not ingestion_program
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


def meta_training(agent, D_tr, encoder_config, epochs, pretrain_epochs):
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
        validation_learning_curves[dataset_name] = env.validation_learning_curves[dataset_name]
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
        epochs=epochs,
        pretrain_epochs=pretrain_epochs,
        **encoder_config
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

        # 

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
    from gravitas.agent_gravitas import Agent  # fixme: for debugging: replace with my own Agent script

    # === Clear old output
    clear_output_dir(output_dir)

    # === Init K-folds cross-validation
    kf = KFold(
        n_splits=6,
        shuffle=False
    )

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
    for D_tr, D_te in kf.split(list_datasets):
        vprint(verbose, "\n********** ITERATION " + str(iteration) + " **********")

        # Init a new agent instance in each iteration to prevent
        # leakage between folds
        agent = Agent(
            number_of_algorithms=len(list_algorithms),
            seed=seed,
            encoder='AE'
        )

        encoder_config = {}

        # === META-TRAINING
        trained_agent = meta_training(agent, D_tr, encoder_config=encoder_config, epochs=10, pretrain_epochs=10)

        # === META-TESTING
        meta_testing(trained_agent, D_te)

        iteration += 1

        # TODO get the performance metric, log it        

        # break
    ################################################

# === For debug only
# tmp = os.listdir(os.path.join(root_dir, 'program'))
# # tmp2 = os.listdir(root_dir)
# tmp = ' '.join([str(item) for item in tmp])
# tmp_3 = os.listdir(tmp)
#
#
# training = {'carlo': {'usage': 'AutoMLchallenge2014', 'name': 'carlo', 'task': 'binary.classification', 'target_type':
#     'Binary', 'feat_type': 'Numerical', 'metric': 'pac_metric', 'time_budget': '1200', 'feat_num': '1070',
#                       'target_num': '1', 'label_num': '2', 'train_num': '50000', 'valid_num': '10000',
#                       'test_num': '10000', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '0'},
#             'christine': {'usage': 'AutoMLchallenge2015', 'name': 'christine', 'task': 'binary.classification',
#                           'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                           'time_budget': '1200', 'feat_num': '1636', 'target_num': '1', 'label_num': '2',
#                           'train_num': '5418', 'valid_num': '834', 'test_num': '2084', 'has_categorical': '0',
#                           'has_missing': '0', 'is_sparse': '0'},
#             'digits': {'usage': 'AutoMLchallenge2014', 'name': 'digits', 'task': 'multiclass.classification',
#                        'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                        'feat_num': '1568', 'target_num': '10', 'label_num': '10', 'train_num': '15000',
#                        'valid_num': '20000', 'test_num': '35000', 'has_categorical': '0', 'has_missing': '0',
#                        'is_sparse': '0', 'time_budget': '300'},
#             'dilbert': {'usage': 'AutoMLchallenge2014', 'name': 'dilbert', 'task': 'multiclass.classification',
#                         'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'pac_metric',
#                         'time_budget': '1200', 'feat_num': '2000', 'target_num': '5', 'label_num': '5',
#                         'train_num': '10000', 'valid_num': '4860', 'test_num': '9720', 'has_categorical': '0',
#                         'has_missing': '0', 'is_sparse': '0'},
#             'dionis': {'usage': 'AutoMLchallenge2014', 'name': 'dionis', 'task': 'multiclass.classification',
#                        'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'bac_metric', 'feat_num': '60',
#                        'time_budget': '1200', 'target_num': '355', 'label_num': '355', 'train_num': '416188',
#                        'valid_num': '6000', 'test_num': '12000', 'has_categorical': '0', 'has_missing': '0',
#                        'is_sparse': '0'},
#             'dorothea': {'usage': 'AutoMLchallenge2014', 'name': 'dorothea', 'task': 'binary.classification',
#                          'target_type': 'Binary', 'feat_type': 'Binary', 'metric': 'auc_metric', 'feat_num': '100000',
#                          'target_num': '1', 'label_num': '2', 'train_num': '800', 'valid_num': '350', 'test_num': '800',
#                          'has_categorical': '0', 'has_missing': '0', 'is_sparse': '1', 'time_budget': '100'},
#             'evita': {'usage': 'AutoMLchallenge2014', 'name': 'evita', 'task': 'binary.classification',
#                       'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'auc_metric',
#                       'feat_num': '3000', 'target_num': '1', 'label_num': '2', 'train_num': '20000',
#                       'valid_num': '8000', 'test_num': '14000', 'has_categorical': '0', 'has_missing': '0',
#                       'is_sparse': '1', 'time_budget': '1200'},
#             'fabert': {'usage': 'AutoMLchallenge2014', 'name': 'fabert', 'task': 'multiclass.classification',
#                        'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'pac_metric',
#                        'time_budget': '1200', 'feat_num': '800', 'target_num': '7', 'label_num': '7',
#                        'train_num': '8237', 'valid_num': '1177', 'test_num': '2354', 'has_categorical': '0',
#                        'has_missing': '0', 'is_sparse': '0'},
#             'flora': {'usage': 'AutoMLchallenge2014', 'name': 'flora', 'task': 'regression', 'target_type': 'Numerical',
#                       'feat_type': 'Numerical', 'metric': 'a_metric', 'time_budget': '1200', 'feat_num': '200000',
#                       'target_num': '1', 'label_num': '0', 'train_num': '15000', 'valid_num': '2000',
#                       'test_num': '2000', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '1'},
#             'grigoris': {'usage': 'AutoMLchallenge2014', 'name': 'grigoris', 'task': 'multilabel.classification',
#                          'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'auc_metric',
#                          'time_budget': '1200', 'feat_num': '301561', 'target_num': '91', 'label_num': '91',
#                          'train_num': '45400', 'valid_num': '6486', 'test_num': '9920', 'has_categorical': '0',
#                          'has_missing': '0', 'is_sparse': '1'},
#             'helena': {'usage': 'AutoMLchallenge2014', 'name': 'helena', 'task': 'multiclass.classification',
#                        'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                        'time_budget': '1200', 'feat_num': '27', 'target_num': '100', 'label_num': '100',
#                        'train_num': '65196', 'valid_num': '9314', 'test_num': '18628', 'has_categorical': '0',
#                        'has_missing': '0', 'is_sparse': '0'},
#             'jannis': {'usage': 'AutoMLchallenge2014', 'name': 'jannis', 'task': 'multiclass.classification',
#                        'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                        'time_budget': '1200', 'feat_num': '54', 'target_num': '4', 'label_num': '4',
#                        'train_num': '83733', 'valid_num': '4926', 'test_num': '9851', 'has_categorical': '0',
#                        'has_missing': '0', 'is_sparse': '0'},
#             'jasmine': {'usage': 'AutoMLchallenge2015', 'name': 'jasmine', 'task': 'binary.classification',
#                         'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                         'time_budget': '1200', 'feat_num': '144', 'target_num': '1', 'label_num': '2',
#                         'train_num': '2984', 'valid_num': '526', 'test_num': '1756', 'has_categorical': '0',
#                         'has_missing': '0', 'is_sparse': '0'},
#             'madeline': {'usage': 'AutoMLchallenge2015', 'name': 'madeline', 'task': 'binary.classification',
#                          'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                          'time_budget': '1200', 'feat_num': '259', 'target_num': '1', 'label_num': '2',
#                          'train_num': '3140', 'valid_num': '1080', 'test_num': '3240', 'has_categorical': '0',
#                          'has_missing': '0', 'is_sparse': '0'},
#             'marco': {'usage': 'AutoMLchallenge2014', 'name': 'marco', 'task': 'multilabel.classification',
#                       'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'auc_metric',
#                       'time_budget': '1200', 'feat_num': '15299', 'target_num': '180', 'label_num': '180',
#                       'train_num': '163860', 'valid_num': '20482', 'test_num': '20482', 'has_categorical': '0',
#                       'has_missing': '0', 'is_sparse': '1'},
#             'newsgroups': {'usage': 'AutoMLchallenge2014', 'name': 'newsgroups', 'task': 'multiclass.classification',
#                            'target_type': 'Numerical', 'feat_type': 'Numerical', 'metric': 'pac_metric',
#                            'feat_num': '61188', 'target_num': '20', 'label_num': '20', 'train_num': '13142',
#                            'valid_num': '1877', 'test_num': '3755', 'has_categorical': '0', 'has_missing': '0',
#                            'is_sparse': '1', 'time_budget': '300'},
#             'pablo': {'usage': 'AutoMLchallenge2014', 'name': 'pablo', 'task': 'regression', 'target_type': 'Numerical',
#                       'feat_type': 'Numerical', 'metric': 'a_metric', 'time_budget': '1200', 'feat_num': '120',
#                       'target_num': '1', 'label_num': '0', 'train_num': '188524', 'valid_num': '23565',
#                       'test_num': '23565', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '0'},
#             'philippine': {'usage': 'AutoMLchallenge2015', 'name': 'philippine', 'task': 'binary.classification',
#                            'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                            'time_budget': '1200', 'feat_num': '308', 'target_num': '1', 'label_num': '2',
#                            'train_num': '5832', 'valid_num': '1166', 'test_num': '4664', 'has_categorical': '0',
#                            'has_missing': '0', 'is_sparse': '0'},
#             'robert': {'usage': 'AutoMLchallenge2014', 'name': 'robert', 'task': 'multiclass.classification',
#                        'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'bac_metric', 'feat_num': '7200',
#                        'target_num': '10', 'label_num': '10', 'train_num': '10000', 'valid_num': '2000',
#                        'test_num': '5000', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '0',
#                        'time_budget': '1200'},
#             'sylvine': {'usage': 'AutoMLchallenge2015', 'name': 'sylvine', 'task': 'binary.classification',
#                         'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'bac_metric',
#                         'time_budget': '1200', 'feat_num': '20', 'target_num': '1', 'label_num': '2',
#                         'train_num': '5124', 'valid_num': '5124', 'test_num': '10244', 'has_categorical': '0',
#                         'has_missing': '0', 'is_sparse': '0'},
#             'tania': {'usage': 'AutoMLchallenge2014', 'name': 'tania', 'task': 'multilabel.classification',
#                       'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'pac_metric', 'feat_num': '47236',
#                       'target_num': '95', 'label_num': '95', 'train_num': '157599', 'valid_num': '22514',
#                       'test_num': '44635', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '1',
#                       'time_budget': '1200'},
#             'volkert': {'usage': 'AutoMLchallenge2014', 'name': 'volkert', 'task': 'multiclass.classification',
#                         'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'pac_metric',
#                         'time_budget': '1200', 'feat_num': '180', 'target_num': '10', 'label_num': '10',
#                         'train_num': '58310', 'valid_num': '3500', 'test_num': '7000', 'has_categorical': '0',
#                         'has_missing': '0', 'is_sparse': '0'},
#             'waldo': {'usage': 'AutoMLchallenge2014', 'name': 'waldo', 'task': 'multiclass.classification',
#                       'target_type': 'Categorical', 'feat_type': 'Mixed', 'metric': 'bac_metric', 'time_budget': '1200',
#                       'feat_num': '270', 'target_num': '4', 'label_num': '4', 'train_num': '19439', 'valid_num': '2430',
#                       'test_num': '2430', 'has_categorical': '1', 'has_missing': '0', 'is_sparse': '0'},
#             'wallis': {'usage': 'AutoMLchallenge2014', 'name': 'wallis', 'task': 'multiclass.classification',
#                        'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'auc_metric',
#                        'time_budget': '1200', 'feat_num': '193731', 'target_num': '11', 'label_num': '11',
#                        'train_num': '10000', 'valid_num': '4098', 'test_num': '8196', 'has_categorical': '0',
#                        'has_missing': '0', 'is_sparse': '1'},
#             'yolanda': {'usage': 'AutoMLchallenge2014', 'name': 'yolanda', 'task': 'regression',
#                         'target_type': 'Numerical', 'feat_type': 'Numerical', 'metric': 'r2_metric',
#                         'time_budget': '1200', 'feat_num': '100', 'target_num': '1', 'label_num': '0',
#                         'train_num': '400000', 'valid_num': '30000', 'test_num': '30000', 'has_categorical': '0',
#                         'has_missing': '0', 'is_sparse': '0'}}
#
# testing = {'adult': {'usage': 'AutoMLchallenge2014', 'name': 'adult', 'task': 'multilabel.classification',
#                      'target_type': 'Binary', 'feat_type': 'Mixed', 'metric': 'f1_metric', 'time_budget': '300',
#                      'feat_num': '24', 'target_num': '3', 'label_num': '3', 'train_num': '34190', 'valid_num': '4884',
#                      'test_num': '9768', 'has_categorical': '1', 'has_missing': '1', 'is_sparse': '0'},
#            'albert': {'usage': 'AutoMLchallenge2014', 'name': 'albert', 'task': 'binary.classification',
#                       'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'f1_metric',
#                       'time_budget': '1200', 'feat_num': '78', 'target_num': '1', 'label_num': '2',
#                       'train_num': '425240', 'valid_num': '25526', 'test_num': '51048', 'has_categorical': '1',
#                       'has_missing': '1', 'is_sparse': '0'},
#            'alexis': {'usage': 'AutoMLchallenge2014', 'name': 'alexis', 'task': 'multilabel.classification',
#                       'target_type': 'Binary', 'feat_type': 'Numerical', 'metric': 'auc_metric', 'feat_num': '5000',
#                       'target_num': '18', 'label_num': '18', 'train_num': '54491', 'valid_num': '7784',
#                       'test_num': '15569', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '1',
#                       'time_budget': '1200'},
#            'arturo': {'usage': 'AutoMLchallenge2014', 'name': 'arturo', 'task': 'multiclass.classification',
#                       'target_type': 'Categorical', 'feat_type': 'Numerical', 'metric': 'f1_metric',
#                       'time_budget': '1200', 'feat_num': '400', 'target_num': '20', 'label_num': '20',
#                       'train_num': '9565', 'valid_num': '1366', 'test_num': '2733', 'has_categorical': '0',
#                       'has_missing': '0', 'is_sparse': '0'},
#            'cadata': {'usage': 'AutoMLchallenge2014', 'name': 'cadata', 'task': 'regression',
#                       'target_type': 'Numerical', 'feat_type': 'Numerical', 'metric': 'r2_metric', 'feat_num': '16',
#                       'target_num': '1', 'label_num': '0', 'train_num': '5000', 'valid_num': '5000',
#                       'test_num': '10640', 'has_categorical': '0', 'has_missing': '0', 'is_sparse': '0',
#                       'time_budget': '200'}}
#
# import pandas as pd
#
# meta_data = pd.concat([pd.DataFrame().from_dict(training, orient='index'),
#                        pd.DataFrame().from_dict(testing, orient='index')])
# meta_data[Dataset_Gravity.columns_categorical].drop_duplicates().to_dict()
