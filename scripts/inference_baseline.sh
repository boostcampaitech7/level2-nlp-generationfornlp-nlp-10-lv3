#!/bin/bash

export PYTHONPATH=$PYTHONPATH:.
python ./inference.py --config_path "./configs/baseline.yaml"