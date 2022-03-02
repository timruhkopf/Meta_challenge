#!/bin/bash

python ingestion_program/ingestion.py --mode submission --pretrain_epochs 100 --epochs 100

python scoring_program/score.py
