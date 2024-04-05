#!/bin/bash
python train.py --model_dir models/MAXSAT --config configs/MAXSAT/MAXSAT_default.json

# loading from last checkpoint
python train.py --model_dir models/MAXSAT --config configs/MAXSAT/MAXSAT_default.json --from_last
