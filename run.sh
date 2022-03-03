#!/bin/bash

rm -rf ../output/

mkdir ../output

python ingestion_program/ingestion.py --mode submission 

python scoring_program/score.py