#!/bin/bash

cat /home/mclovin/git/temp/Meta_challenge/gravitas/utils.py \
/home/mclovin/git/temp/Meta_challenge/gravitas/base_encoder.py \
/home/mclovin/git/temp/Meta_challenge/gravitas/autoencoder.py \
/home/mclovin/git/temp/Meta_challenge/gravitas/vae.py \
/home/mclovin/git/temp/Meta_challenge/gravitas/dataset_gravitas.py \
/home/mclovin/git/temp/Meta_challenge/gravitas/agent_gravitas.py   \
| grep -Ev  '^from gravitas|^from networkx|^*self\.plot_*|root_dir|import scipy.stats as ss' > /home/mclovin/git/temp/Meta_challenge/submission/submission_docs/agent.py

#grep -v '^from gravitas' \