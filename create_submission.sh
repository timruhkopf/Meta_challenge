#!/bin/bash

cat /home/ruhkopf/PycharmProjects/Meta_challenge/gravitas/utils.py \
/home/ruhkopf/PycharmProjects/Meta_challenge/gravitas/base_encoder.py \
/home/ruhkopf/PycharmProjects/Meta_challenge/gravitas/autoencoder.py \
/home/ruhkopf/PycharmProjects/Meta_challenge/gravitas/vae.py \
/home/ruhkopf/PycharmProjects/Meta_challenge/gravitas/dataset_gravitas.py \
/home/ruhkopf/PycharmProjects/Meta_challenge/gravitas/agent_gravitas.py   \
| grep -Ev  '^from gravitas|^from networkx' > /home/ruhkopf/PycharmProjects/Meta_challenge/submission/submission_docs/agent.py

#grep -v '^from gravitas' \