#!/bin/bash

python ingestion_program/ingestion.py --mode submission --pretrain_epochs 10 --epochs 10

python scoring_program/score.py
