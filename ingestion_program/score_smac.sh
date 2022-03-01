#!/bin/bash

source ~/miniconda3/bin/activate meta
cd ~/PycharmProjects/Meta_challenge/scoring_program
python score.py

# change back the dir for smac
cd ~/PycharmProjects/Meta_challenge/ingestion_program/
