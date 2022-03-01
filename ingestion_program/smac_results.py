import json
import os
from itertools import chain

import pandas as pd

from ingestion_program.ingestion_smac import tae, clear_output_dir, output_dir  # , kf, env  # just for explicitness

run_id = 'VAE1645638049.1745791'
# run_id = 'AE1645636558.0056844'

# to iterate through all the files:

run = -1
root_dir = '/'.join(os.getcwd().split('/')[:-1])
dir_smac = f'{root_dir}/output_smac/'

# run_id = os.listdir(dir_smac)[run]
smac_id = os.listdir(dir_smac + run_id)[0]

# fixing the root to project root and not ingestion_program
with open(dir_smac + f'{run_id}/{smac_id}/runhistory.json') as json_file:
    runhistory = json.load(json_file)

# preprocess config dict
configs = runhistory['configs']
config_df = pd.DataFrame.from_dict(configs, orient='index')

# preprocess the rundata dict
rundata = [list(chain(*i)) for i in runhistory['data']]

run_df = pd.DataFrame([[list(item.values()) if isinstance(item, dict) else item
                        for item in step] for step in rundata],
                      columns=(
                          'config_id', 'instance_id', 'seed', 'budget', 'cost', 'time', 'status', 'starttime',
                          'endtime',
                          'additional_info'))
run_df['status'] = run_df['status'].apply(lambda x: x[0].split('.')[-1])

# add the configs in the df
config_df.index = config_df.index.astype(int)
run_config_df = pd.merge(config_df, run_df, left_index=True, right_on='config_id')

# Find the not failed configs on highest fidelity.
budget_success = (run_df['budget'] == 10000) & (run_df['status'] == 'SUCCESS')
final_fidelity = run_config_df[budget_success]
final_fidelity = final_fidelity[['cost', 'budget', 'status', 'additional_info', *config_df.columns]].sort_values('cost')

# Write out frame to csv.
# run_config_df.to_csv(f'{root_dir}/output/run_{run_id}/run_{run_id}.csv')
config_names = configs['1'].keys()
# index is runid!
rerun_config = final_fidelity[config_names].transpose().to_dict()

# single config rerun of TAE:
# TODO create an ID, with which the models are plotted and saved.
clear_output_dir(output_dir)
cfg = configs['6']
tae(cfg, budget=1000)
